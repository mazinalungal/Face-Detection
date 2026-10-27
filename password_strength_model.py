from __future__ import annotations

import json
import random
import string
from collections import Counter
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

matplotlib.use("Agg")

DATA_PATH = Path("password_strength_dataset.csv")
PLOTS_DIR = Path("plots")
MODEL_PATH = Path("password_strength_model.joblib")
RESULTS_PATH = Path("evaluation_results.json")

FEATURE_COLUMNS = [
    "length",
    "uppercase",
    "lowercase",
    "digits",
    "symbols",
    "unique_chars",
    "repeated_chars",
    "common_word",
    "has_upper",
    "has_lower",
    "has_digit",
    "has_symbol",
    "is_common_pattern",
    "entropy",
]

CLASS_NAMES = ["weak", "medium", "strong"]
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS = {value: key for key, value in CLASS_TO_INDEX.items()}

COMMON_PASSWORDS = [
    "password",
    "admin",
    "qwerty",
    "welcome",
    "sunshine",
    "dragon",
    "football",
    "monkey",
    "letmein",
    "abc123",
    "iloveyou",
    "princess",
    "passw0rd",
    "happy123",
]


def calculate_entropy(password: str) -> float:
    if not password:
        return 0.0
    counts = Counter(password)
    total = len(password)
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * np.log2(probability)
    return round(float(entropy), 4)


def detect_pattern_family(password: str) -> str:
    lower = password.lower()
    if any(word in lower for word in COMMON_PASSWORDS):
        return "common"
    upper = sum(1 for c in password if c.isupper())
    lower_count = sum(1 for c in password if c.islower())
    digits = sum(1 for c in password if c.isdigit())
    symbols = sum(1 for c in password if not c.isalnum())
    if upper > 0 and lower_count > 0 and digits > 0 and symbols > 0 and len(password) >= 12:
        return "complex"
    if upper > 0 and lower_count > 0 and digits > 0:
        return "mixed"
    if lower_count > 0:
        return "simple"
    return "numeric"


def compute_password_features(password: str) -> dict:
    password = str(password)
    lower_count = sum(1 for c in password if c.islower())
    upper_count = sum(1 for c in password if c.isupper())
    digit_count = sum(1 for c in password if c.isdigit())
    symbol_count = sum(1 for c in password if not c.isalnum())
    unique_chars = len(set(password))
    repeated_chars = sum(1 for char in set(password) if password.count(char) > 1)
    common_word_flag = int(any(word in password.lower() for word in COMMON_PASSWORDS))
    has_upper = int(upper_count > 0)
    has_lower = int(lower_count > 0)
    has_digit = int(digit_count > 0)
    has_symbol = int(symbol_count > 0)
    common_pattern = int(
        any(
            seq in password.lower()
            for seq in ["123", "abc", "qwe", "password", "admin", "welcome", "letmein"]
        )
    )
    return {
        "length": len(password),
        "uppercase": upper_count,
        "lowercase": lower_count,
        "digits": digit_count,
        "symbols": symbol_count,
        "unique_chars": unique_chars,
        "repeated_chars": repeated_chars,
        "common_word": common_word_flag,
        "has_upper": has_upper,
        "has_lower": has_lower,
        "has_digit": has_digit,
        "has_symbol": has_symbol,
        "is_common_pattern": common_pattern,
        "entropy": calculate_entropy(password),
        "pattern_family": detect_pattern_family(password),
    }


def generate_password_for_label(label: str) -> str:
    if label == "weak":
        weak_templates = [
            random.choice(COMMON_PASSWORDS),
            random.choice(["abc123", "welcome1", "admin2024", "sunshine"]),
            ("".join(random.choice(string.ascii_lowercase) for _ in range(random.randint(4, 8)))
             + str(random.randint(10, 99))),
        ]
        return random.choice(weak_templates)

    if label == "medium":
        medium_templates = [
            random.choice(["Summer2024", "BlueSky9", "Tiger#42", "Mango123", "Cloud7!", "River2025"]),
            random.choice(string.ascii_uppercase) + "".join(random.choice(string.ascii_lowercase) for _ in range(random.randint(5, 8))) + str(random.randint(10, 99)),
            random.choice(string.ascii_uppercase) + "".join(random.choice(string.ascii_lowercase) for _ in range(random.randint(6, 9))) + str(random.randint(10, 99)),
        ]
        return random.choice(medium_templates)

    strong_templates = [
        random.choice(["R!ver#42Lake", "C0d3!M@n#7", "7*Sky$Blue6", "N!ce@2025Road", "Z3bra!W@ves"]),
        random.choice(string.ascii_uppercase) + "".join(random.choice(string.ascii_letters + string.digits + "!@#$%^&*()") for _ in range(12, 18)),
        "".join(random.choice(string.ascii_letters + string.digits + "!@#$%^&*()") for _ in range(14, 18)),
    ]
    return random.choice(strong_templates)


def build_dataset(rows_per_class: int = 1200) -> pd.DataFrame:
    records = []
    for label in CLASS_NAMES:
        for _ in range(rows_per_class):
            password = generate_password_for_label(label)
            features = compute_password_features(password)
            records.append({"password": password, "strength": label, **features})

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["password"]).reset_index(drop=True)
    return df


def save_dataset() -> pd.DataFrame:
    df = build_dataset()
    df.to_csv(DATA_PATH, index=False)
    print(f"Dataset saved to {DATA_PATH}")
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["pattern_family"]),
            ("num", StandardScaler(), FEATURE_COLUMNS),
        ],
        remainder="drop",
    )


def evaluate_models(X: pd.DataFrame, y: pd.Series) -> tuple[list[dict], str, Pipeline]:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
        "SVM": SVC(probability=True, kernel="rbf", random_state=42),
    }

    cv_results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, classifier in models.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", classifier),
        ])

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "roc_auc_ovr_weighted"],
            n_jobs=None,
        )

        cv_results.append({
            "Model": name,
            "CV Accuracy": round(float(np.mean(scores["test_accuracy"])), 4),
            "CV Precision": round(float(np.mean(scores["test_precision_weighted"])), 4),
            "CV Recall": round(float(np.mean(scores["test_recall_weighted"])), 4),
            "CV F1": round(float(np.mean(scores["test_f1_weighted"])), 4),
            "CV ROC-AUC": round(float(np.mean(scores["test_roc_auc_ovr_weighted"])), 4),
        })

    model_summary = pd.DataFrame(cv_results).sort_values("CV F1", ascending=False)
    chosen_model_name = model_summary.iloc[0]["Model"]
    final_model = models[chosen_model_name]
    final_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", final_model),
    ])
    return model_summary.to_dict("records"), chosen_model_name, final_pipeline


def plot_class_distribution(df: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="strength", palette="Set2")
    plt.title("Class Distribution of Password Strength")
    plt.xlabel("Password Strength")
    plt.ylabel("Count")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "class_distribution.png", dpi=300)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    corr = df[FEATURE_COLUMNS].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=300)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=300)
    plt.close()


def plot_roc_curve(model_pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_prob = model_pipeline.predict_proba(X_test)
    classes = model_pipeline.classes_
    plt.figure(figsize=(8, 6))

    for label in classes:
        class_name = INDEX_TO_CLASS[int(label)]
        y_true_bin = (y_test == label).astype(int)
        y_score = y_prob[:, int(label)]
        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        auc_score = roc_auc_score(y_true_bin, y_score)
        plt.plot(fpr, tpr, label=f"{class_name} (AUC={auc_score:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_curve.png", dpi=300)
    plt.close()


def plot_model_comparison(model_summary: list[dict]) -> None:
    summary_df = pd.DataFrame(model_summary)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=summary_df, x="Model", y="CV F1", palette="viridis")
    plt.title("Cross-Validation F1 Comparison")
    plt.ylabel("CV F1 Score")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_comparison.png", dpi=300)
    plt.close()


def main() -> None:
    print("Step 1: Loading / generating dataset...")
    if not DATA_PATH.exists():
        df = save_dataset()
    else:
        df = pd.read_csv(DATA_PATH)

    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("Class counts:")
    print(df["strength"].value_counts())

    print("\nStep 2: Checking missing values and duplicates...")
    print(df.isnull().sum())
    print(f"Duplicate rows: {df.duplicated().sum()}")

    df = df.drop_duplicates().dropna().reset_index(drop=True)
    X = df[["pattern_family"] + FEATURE_COLUMNS]
    y = df["strength"].map(CLASS_TO_INDEX)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    print("\nStep 3: Comparing models via 5-fold CV...")
    model_summary, final_model_name, final_pipeline = evaluate_models(X, y)
    print(pd.DataFrame(model_summary))

    print(f"\nChosen final model: {final_model_name}")
    final_pipeline.fit(X_train, y_train)
    y_pred = final_pipeline.predict(X_test)
    y_prob = final_pipeline.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    cm = confusion_matrix(y_test, y_pred, labels=[CLASS_TO_INDEX[name] for name in CLASS_NAMES])

    print("\nStep 4: Metrics on the held-out test set")
    results = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
    }
    print(results)
    print("Confusion matrix:")
    print(cm)

    PLOTS_DIR.mkdir(exist_ok=True)
    plot_class_distribution(df)
    plot_correlation_heatmap(df)
    plot_confusion_matrix(cm)
    plot_roc_curve(final_pipeline, X_test, y_test)
    plot_model_comparison(model_summary)

    joblib.dump(final_pipeline, MODEL_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}")

    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump({
            "chosen_model": final_model_name,
            "metrics": results,
            "confusion_matrix": cm.tolist(),
            "model_summary": model_summary,
        }, f, indent=2)

    print(f"Saved evaluation results to {RESULTS_PATH}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
