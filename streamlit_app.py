from __future__ import annotations

import json
import math
import secrets
import string
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "password_strength_model.joblib"
RESULTS_PATH = APP_DIR / "evaluation_results.json"
PLOTS_DIR = APP_DIR / "plots"

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
CLASS_NAMES = {0: "weak", 1: "medium", 2: "strong"}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_evaluation_results() -> dict:
    with RESULTS_PATH.open(encoding="utf-8") as results_file:
        return json.load(results_file)


def calculate_entropy(password: str) -> float:
    if not password:
        return 0.0
    counts = Counter(password)
    total = len(password)
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return round(entropy, 4)


def determine_pattern_family(password: str) -> str:
    lower = password.lower()
    if any(word in lower for word in COMMON_PASSWORDS):
        return "common"
    upper = sum(1 for char in password if char.isupper())
    lower_count = sum(1 for char in password if char.islower())
    digits = sum(1 for char in password if char.isdigit())
    symbols = sum(1 for char in password if not char.isalnum())
    if upper > 0 and lower_count > 0 and digits > 0 and symbols > 0 and len(password) >= 12:
        return "complex"
    if upper > 0 and lower_count > 0 and digits > 0:
        return "mixed"
    if lower_count > 0:
        return "simple"
    return "numeric"


def build_input_row(password: str) -> pd.DataFrame:
    data = {
        "pattern_family": determine_pattern_family(password),
        "length": len(password),
        "uppercase": sum(1 for char in password if char.isupper()),
        "lowercase": sum(1 for char in password if char.islower()),
        "digits": sum(1 for char in password if char.isdigit()),
        "symbols": sum(1 for char in password if not char.isalnum()),
        "unique_chars": len(set(password)),
        "repeated_chars": sum(1 for char in set(password) if password.count(char) > 1),
        "common_word": int(any(word in password.lower() for word in COMMON_PASSWORDS)),
        "has_upper": int(any(char.isupper() for char in password)),
        "has_lower": int(any(char.islower() for char in password)),
        "has_digit": int(any(char.isdigit() for char in password)),
        "has_symbol": int(any(not char.isalnum() for char in password)),
        "is_common_pattern": int(
            any(
                sequence in password.lower()
                for sequence in ["123", "abc", "qwe", "password", "admin", "welcome", "letmein"]
            )
        ),
        "entropy": calculate_entropy(password),
    }
    return pd.DataFrame([data], columns=["pattern_family", *FEATURE_COLUMNS])


def generate_password(length: int) -> str:
    groups = [string.ascii_uppercase, string.ascii_lowercase, string.digits, string.punctuation]
    characters = [secrets.choice(group) for group in groups]
    alphabet = "".join(groups)
    characters.extend(secrets.choice(alphabet) for _ in range(length - len(characters)))
    secrets.SystemRandom().shuffle(characters)
    return "".join(characters)


def render_password_checks(password: str, row: pd.DataFrame) -> None:
    checks = [
        ("At least 12 characters", len(password) >= 12),
        ("Uppercase and lowercase letters", bool(row.at[0, "has_upper"] and row.at[0, "has_lower"])),
        ("Contains a number", bool(row.at[0, "has_digit"])),
        ("Contains a symbol", bool(row.at[0, "has_symbol"])),
        ("Avoids common words and patterns", not bool(row.at[0, "common_word"] or row.at[0, "is_common_pattern"])),
    ]
    st.subheader("Ways to improve")
    for label, passed in checks:
        st.write(f"{'✅' if passed else '⬜'} {label}")


def render_analyzer() -> None:
    st.subheader("Check a password")
    st.caption("Avoid entering a password you use for a real account.")
    show_password = st.checkbox("Show password while typing")
    with st.form("password_analyzer"):
        password = st.text_input(
            "Enter a password",
            type="default" if show_password else "password",
            placeholder="Type a sample password",
        )
        submitted = st.form_submit_button("Analyze strength", type="primary")

    if not submitted:
        return
    if not password:
        st.warning("Please enter a password first.")
        return

    row = build_input_row(password)
    model = load_model()
    prediction = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    probability_map = {
        CLASS_NAMES[int(class_index)]: float(probability)
        for class_index, probability in zip(model.classes_, probabilities)
    }
    prediction_label = CLASS_NAMES[prediction]
    st.success(f"Predicted strength: {prediction_label.upper()}")

    metrics = st.columns(4)
    metrics[0].metric("Length", f"{len(password)} characters")
    metrics[1].metric("Unique characters", int(row.at[0, "unique_chars"]))
    metrics[2].metric("Character entropy", f"{row.at[0, 'entropy']:.2f} bits")
    metrics[3].metric("Model confidence", f"{probability_map[prediction_label]:.0%}")

    charts = st.columns(2)
    with charts[0]:
        st.subheader("Character composition")
        composition = pd.DataFrame(
            {
                "Characters": [
                    int(row.at[0, "uppercase"]),
                    int(row.at[0, "lowercase"]),
                    int(row.at[0, "digits"]),
                    int(row.at[0, "symbols"]),
                ]
            },
            index=["Uppercase", "Lowercase", "Digits", "Symbols"],
        )
        st.bar_chart(composition, y="Characters", color="#4c78a8")
    with charts[1]:
        st.subheader("Class probabilities")
        probabilities_chart = pd.DataFrame(
            {"Probability": [probability_map[name] for name in CLASS_NAMES.values()]},
            index=[name.title() for name in CLASS_NAMES.values()],
        )
        st.bar_chart(probabilities_chart, y="Probability", color="#59a14f")

    render_password_checks(password, row)


def render_model_insights() -> None:
    results = load_evaluation_results()
    st.subheader("Model evaluation")
    st.caption(f"Selected model: {results['chosen_model']}")

    metric_columns = st.columns(len(results["metrics"]))
    for column, (metric, value) in zip(metric_columns, results["metrics"].items()):
        column.metric(metric.replace("_", " ").title(), f"{value:.1%}")

    summary = pd.DataFrame(results["model_summary"]).set_index("Model")
    st.subheader("Cross-validation model comparison")
    st.bar_chart(summary[["CV Accuracy", "CV Precision", "CV Recall", "CV F1"]])

    graph_tabs = st.tabs(
        ["Class distribution", "Confusion matrix", "ROC curve", "Feature correlations"]
    )
    graph_files = [
        "class_distribution.png",
        "confusion_matrix.png",
        "roc_curve.png",
        "correlation_heatmap.png",
    ]
    captions = [
        "Examples per password-strength class in the dataset",
        "Held-out predictions compared with the true classes",
        "One-vs-rest ROC curves for the three classes",
        "Correlations between the numeric model features",
    ]
    for tab, filename, caption in zip(graph_tabs, graph_files, captions):
        with tab:
            st.image(PLOTS_DIR / filename, caption=caption, width="stretch")


st.set_page_config(page_title="Password Strength Predictor", page_icon="🔐", layout="wide")
st.title("Password Strength Predictor")
st.caption("Explore password strength predictions and model performance.")

analyzer_tab, insights_tab, generator_tab = st.tabs(
    ["Password analyzer", "Model insights", "Password generator"]
)
with analyzer_tab:
    render_analyzer()
with insights_tab:
    render_model_insights()
with generator_tab:
    st.subheader("Generate a random password")
    st.caption("Generated in this app session; copy it to a password manager before navigating away.")
    password_length = st.slider("Password length", min_value=12, max_value=32, value=16)
    if st.button("Generate password", type="primary"):
        st.code(generate_password(password_length), language=None)
