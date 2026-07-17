"""
Spark-приложение для задания "Работа со Spark".

Выполняет пункты 1–8:
1. SparkSession с master=yarn, 2 executor'а
2. Пустой файл на HDFS /sparkExperiments.txt
3. Чтение ratings/tags, подсчёт строк, стейджей и тасков
4. Уникальные фильмы и юзеры
5. Число оценок >= 4.0
6. Средняя дельта времени между тегом и оценкой
7. Среднее от средних оценок пользователей
8. TF-IDF + SGDRegressor через UDF, RMSE

Все результаты дописываются в /sparkExperiments.txt на HDFS.
"""

import os
import joblib
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

HDFS_URI = "hdfs://192.168.34.2:8020"
YARN_RM = "192.168.34.2:8032"
DATA_DIR = f"{HDFS_URI}/ml-latest-small"
LOCAL_DATA_DIR = "/app/ml-latest-small"
OUTPUT_FILE = f"{HDFS_URI}/sparkExperiments.txt"

VECTORIZER_PATH = "tfidf_vectorizer.pkl"
MODEL_PATH = "sgd_model.pkl"


def get_hadoop_fs(spark):
    """Доступ к HDFS FileSystem API через JVM-мост py4j."""
    hadoop_conf = spark._jsc.hadoopConfiguration()
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
    return fs


def write_all_lines(spark, fs, path_str, lines):
    """Перезаписывает файл на HDFS целиком одним writeм — надёжнее многократного append,
    который на кластерах с одним датанодом часто падает с 'bad datanode' при повторных открытиях."""
    Path = spark._jvm.org.apache.hadoop.fs.Path
    path = Path(path_str)
    if fs.exists(path):
        fs.delete(path, False)
    out_stream = fs.create(path)
    content = "\n".join(lines) + "\n"
    out_stream.write(content.encode("utf-8"))
    out_stream.close()


def count_stages_and_tasks(spark, job_group, action_fn):
    """
    Выполняет action_fn() под меткой job_group и считает, сколько было
    стейджей и тасков именно для запущенных в этой группе джобов.
    Возвращает (результат action_fn, число_стейджей, число_тасков).
    """
    sc = spark.sparkContext
    sc.setJobGroup(job_group, job_group)

    result = action_fn()

    tracker = sc.statusTracker()
    job_ids = tracker.getJobIdsForGroup(job_group) or []

    stage_ids = set()
    task_count = 0
    for jid in job_ids:
        job_info = tracker.getJobInfo(jid)
        if job_info is None:
            continue
        for sid in job_info.stageIds:
            if sid in stage_ids:
                continue
            stage_ids.add(sid)
            stage_info = tracker.getStageInfo(sid)
            if stage_info is not None:
                task_count += stage_info.numTasks

    sc.setJobGroup("", "")
    return result, len(stage_ids), task_count


def ensure_dataset_on_hdfs(spark, fs):
    """Если ml-latest-small отсутствует на HDFS — заливает его из локальной копии внутри образа."""
    Path = spark._jvm.org.apache.hadoop.fs.Path
    remote_dir = Path(DATA_DIR)
    if fs.exists(remote_dir):
        print(f"[INFO] Датасет уже есть на HDFS: {DATA_DIR}")
        return

    print(f"[INFO] Датасет не найден на HDFS ({DATA_DIR}), загружаю из локальной копии в образе...")
    fs.mkdirs(remote_dir)
    for fname in ["ratings.csv", "tags.csv"]:
        local_path = Path(f"file://{os.path.join(LOCAL_DATA_DIR, fname)}")
        remote_path = Path(f"{DATA_DIR}/{fname}")
        fs.copyFromLocalFile(False, True, local_path, remote_path)
        print(f"[INFO] Загружен {fname} на HDFS")


def main():
    # ---------- Пункт 1: SparkSession с YARN, 2 executor'а ----------
    spark = (
        SparkSession.builder.appName("SparkExperiments")
        .master("yarn")
        .config("spark.submit.deployMode", "client")
        .config("spark.executor.instances", "2")
        .config("spark.hadoop.fs.defaultFS", HDFS_URI)
        .config("spark.hadoop.yarn.resourcemanager.address", YARN_RM)
        .config("spark.sql.adaptive.enabled", "false")
        .getOrCreate()
    )
    sc = spark.sparkContext
    fs = get_hadoop_fs(spark)
    Path = spark._jvm.org.apache.hadoop.fs.Path

    # ---------- Пункт 2: пустой файл на HDFS ----------
    out_path = Path(OUTPUT_FILE)
    if fs.exists(out_path):
        fs.delete(out_path, False)
    fs.create(out_path).close()

    output_lines = []

    # ---------- Диагностика: где реально лежат данные на HDFS ----------
    def list_dir(path_str, depth=0, max_depth=2):
        try:
            statuses = fs.listStatus(Path(path_str))
        except Exception as e:
            print(f"[DEBUG] Не удалось прочитать {path_str}: {e}")
            return
        for status in statuses:
            full_path = status.getPath().toString()
            is_dir = status.isDirectory()
            print(f"[DEBUG] {'DIR ' if is_dir else 'FILE'} {full_path}")
            if is_dir and depth < max_depth:
                list_dir(full_path, depth + 1, max_depth)

    print("[DEBUG] ===== Содержимое корня HDFS ('/') =====")
    list_dir("/")
    print("[DEBUG] ===== Конец диагностики =====")

    ensure_dataset_on_hdfs(spark, fs)

    # ---------- Пункт 3: чтение ratings/tags, подсчёт строк + стейджей/тасков ----------
    ratings_df = spark.read.csv(f"{DATA_DIR}/ratings.csv", header=True, inferSchema=True)
    tags_df = spark.read.csv(f"{DATA_DIR}/tags.csv", header=True, inferSchema=True)

    def do_counts():
        return ratings_df.count(), tags_df.count()

    (ratings_count, tags_count), stages, tasks = count_stages_and_tasks(
        spark, "count_ratings_tags", do_counts
    )

    output_lines.append(f"ratingsCount:{ratings_count} tagsCount:{tags_count}")
    output_lines.append(f"stages:{stages} tasks:{tasks}")

    # ---------- Пункт 4: уникальные фильмы и юзеры ----------
    films_unique = ratings_df.select("movieId").distinct().count()
    users_unique = ratings_df.select("userId").distinct().count()
    output_lines.append(f"filmsUnique:{films_unique} usersUnique:{users_unique}")

    # ---------- Пункт 5: число оценок >= 4.0 ----------
    good_rating = ratings_df.filter(F.col("rating") >= 4.0).count()
    output_lines.append(f"goodRating:{good_rating}")

    # ---------- Пункт 6: средняя дельта времени тег/оценка ----------
    joined = ratings_df.alias("r").join(
        tags_df.alias("t"), on=["userId", "movieId"], how="inner"
    )
    joined = joined.withColumn(
        "timeDiff", F.col("t.timestamp") - F.col("r.timestamp")
    )
    avg_time_diff = joined.select(F.avg("timeDiff")).first()[0]
    output_lines.append(f"timeDifference:{avg_time_diff}")

    # ---------- Пункт 7: среднее от средних оценок пользователей ----------
    per_user_avg = ratings_df.groupBy("userId").agg(F.avg("rating").alias("avgRating"))
    avg_of_avg = per_user_avg.select(F.avg("avgRating")).first()[0]
    output_lines.append(f"avgRating:{avg_of_avg}")

    # ---------- Пункт 8: TF-IDF + SGDRegressor, предсказание рейтинга по тегу, RMSE ----------
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)

    def predict_rating_udf(tag_text):
        """UDF-функция: предсказывает рейтинг по тексту тега через TF-IDF + SGDRegressor."""
        if tag_text is None:
            return None
        vec = vectorizer.transform([str(tag_text)])
        pred = model.predict(vec)
        return float(pred[0])

    tags_with_ratings = tags_df.join(
        ratings_df.select("userId", "movieId", "rating"),
        on=["userId", "movieId"],
        how="inner",
    )

    # На executor-узлах этого YARN-кластера нет интерпретатора python3, поэтому
    # ЛЮБОЕ действие, которое Spark пытается выполнить через Python-воркер на
    # executor'е (обычный F.udf, а также spark.createDataFrame() из pandas —
    # он тоже параллелизует Python-объекты через воркер), падает с
    # "Cannot run program python3". toPandas() эту проблему не имеет, т.к.
    # сериализация Arrow идёт на стороне JVM executor'а, а не через python-воркер.
    # Поэтому дальше всё считаем локально на driver в pandas, не создавая
    # новый распределённый DataFrame из результата.
    pdf = tags_with_ratings.toPandas()
    pdf["predictedRating"] = pdf["tag"].apply(predict_rating_udf)

    # Убеждаемся, что UDF реально работает — печатаем первые 50 строк
    print(pdf.head(50).to_string())

    rmse = float(np.sqrt(np.sum((pdf["predictedRating"] - pdf["rating"]) ** 2)))
    output_lines.append(f"rmse:{rmse}")

    write_all_lines(spark, fs, OUTPUT_FILE, output_lines)

    spark.stop()


if __name__ == "__main__":
    main()