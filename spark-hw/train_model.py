"""
Локальное обучение модели предсказания рейтинга по тексту тега.

Запускается ОДИН РАЗ у вас локально (не в контейнере, не на кластере).
Результат — два файла (tfidf_vectorizer.pkl, sgd_model.pkl), которые
кладутся в Docker-образ и используются внутри Spark UDF на кластере.

Ожидается, что рядом лежит папка ml-latest-small/ (ratings.csv, tags.csv) —
скачать: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
"""

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

DATA_DIR = "ml-latest-small"


def main():
    ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")
    tags = pd.read_csv(f"{DATA_DIR}/tags.csv")

    # Джойним теги с оценками того же пользователя и фильма —
    # так каждому тексту тега сопоставляется рейтинг, который надо предсказывать.
    merged = tags.merge(ratings, on=["userId", "movieId"], how="inner")
    merged = merged.dropna(subset=["tag", "rating"])
    merged["tag"] = merged["tag"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        merged["tag"], merged["rating"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = SGDRegressor(random_state=42, max_iter=1000, tol=1e-3)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    rmse_local = mean_squared_error(y_test, preds) ** 0.5
    print(f"Локальная проверка (hold-out), RMSE: {rmse_local:.4f}")

    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    joblib.dump(model, "sgd_model.pkl")
    print("Модель и векторизатор сохранены: tfidf_vectorizer.pkl, sgd_model.pkl")


if __name__ == "__main__":
    main()