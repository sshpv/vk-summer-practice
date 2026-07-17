# Собирать под x86_64, т.к. кластер тестирования - не ARM:
#   docker build --platform linux/amd64 -t spark-experiments .

FROM eclipse-temurin:11-jdk-jammy

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Hadoop/YARN конфиги с адресами кластера
ENV HADOOP_CONF_DIR=/app/hadoop-conf
RUN mkdir -p $HADOOP_CONF_DIR
COPY core-site.xml $HADOOP_CONF_DIR/core-site.xml
COPY yarn-site.xml $HADOOP_CONF_DIR/yarn-site.xml

# Приложение и заранее обученные артефакты модели
COPY spark_app.py .
COPY tfidf_vectorizer.pkl .
COPY sgd_model.pkl .
COPY ml-latest-small/ratings.csv ml-latest-small/ratings.csv
COPY ml-latest-small/tags.csv ml-latest-small/tags.csv

ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

ENTRYPOINT ["python3", "spark_app.py"]