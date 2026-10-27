 
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Prediction System")
st.write("Machine Learning based student marks prediction.")

# ---------------- LOAD DATASET ----------------
DATA_FILE = Path(__file__).parent / "students_dataset_clean.csv"

if not DATA_FILE.exists():
    st.error("Dataset file not found!")
    st.write(
        "Keep students_dataset_clean.csv "
        "in the same folder as app.py."
    )
    st.stop()


@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)


try:
    df = load_data(str(DATA_FILE))
    df.columns = df.columns.astype(str).str.strip()
except Exception as e:
    st.error(f"Dataset loading error: {e}")
    st.stop()

df = df.dropna(axis=1, how="all")

if df.empty:
    st.error("Dataset is empty.")
    st.stop()

# ---------------- SELECT TARGET COLUMN ----------------
target_candidates = [
    "G3", "final_grade", "Final Grade",
    "Final_Marks", "final_marks",
    "Marks", "Score", "score", "target"
]

target = next(
    (col for col in target_candidates if col in df.columns),
    None
)

if target is None:
    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if not numeric_cols:
        st.error("No numeric marks column found.")
        st.write("Available columns:", list(df.columns))
        st.stop()

    target = numeric_cols[-1]

df[target] = pd.to_numeric(df[target], errors="coerce")
df = df.dropna(subset=[target]).copy()

if len(df) < 5:
    st.error("At least 5 valid dataset rows are required.")
    st.stop()

# ---------------- PREPARE FEATURES ----------------
feature_cols = [
    col for col in df.columns
    if col != target and df[col].notna().any()
]

if not feature_cols:
    st.error("No input columns found.")
    st.stop()

X = df[feature_cols].copy()
y = df[target].copy()

# Convert numeric-looking text columns to numbers
for col in X.columns:
    if X[col].dtype == "object":
        converted = pd.to_numeric(X[col], errors="coerce")

        if converted.notna().mean() >= 0.90:
            X[col] = converted

numeric_cols = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_cols = [
    col for col in X.columns
    if col not in numeric_cols
]

# ---------------- BUILD PREPROCESSOR ----------------
transformers = []

if numeric_cols:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    transformers.append((
        "numeric",
        numeric_pipeline,
        numeric_cols
    ))

if categorical_cols:
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers.append((
        "categorical",
        categorical_pipeline,
        categorical_cols
    ))

preprocessor = ColumnTransformer(
    transformers=transformers
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

# ---------------- TRAIN MODEL ----------------
mae = None
r2 = None

try:
    if len(df) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

        model.fit(X_train, y_train)

        test_predictions = model.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            test_predictions
        )

        if len(y_test) > 1:
            r2 = r2_score(y_test, test_predictions)

    else:
        model.fit(X, y)

except Exception as e:
    st.error(f"Model training error: {e}")
    st.stop()

# ---------------- MODEL PERFORMANCE ----------------
st.subheader("📊 Model Performance")

c1, c2, c3 = st.columns(3)

c1.metric("Total Students", len(df))

c2.metric(
    "Mean Absolute Error",
    f"{mae:.2f}" if mae is not None else "N/A"
)

c3.metric(
    "R² Score",
    f"{r2:.3f}" if r2 is not None else "N/A"
)

st.caption(f"Prediction target: {target}")
st.divider()

# ---------------- STUDENT INPUT FORM ----------------
st.subheader("📝 Enter Student Details")

st.write(
    "Enter the student's details and click Predict Marks."
)

with st.form("student_form"):

    student_values = {}
    input_columns = st.columns(2)

    for index, col in enumerate(feature_cols):

        with input_columns[index % 2]:

            label = col.replace("_", " ")

            if col in numeric_cols:

                valid_values = pd.to_numeric(
                    X[col],
                    errors="coerce"
                ).dropna()

                minimum = (
                    float(valid_values.min())
                    if not valid_values.empty else 0.0
                )

                maximum = (
                    float(valid_values.max())
                    if not valid_values.empty else 100.0
                )

                default = (
                    float(valid_values.median())
                    if not valid_values.empty else 0.0
                )

                # Avoid identical minimum and maximum
                if minimum == maximum:
                    maximum = minimum + 1.0

                default = min(
                    max(default, minimum),
                    maximum
                )

                student_values[col] = st.number_input(
                    label,
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                    step=1.0,
                    key=f"input_{index}"
                )

            else:

                options = sorted(
                    X[col].dropna().astype(str).unique().tolist()
                )

                if not options:
                    options = ["Unknown"]

                student_values[col] = st.selectbox(
                    label,
                    options,
                    key=f"input_{index}"
                )

    predict_button = st.form_submit_button(
        "🔮 Predict Student Marks",
        type="primary"
    )

# ---------------- PREDICTION ----------------
if predict_button:

    # Use exactly the same columns and order as training
    input_df = pd.DataFrame(
        [student_values],
        columns=feature_cols
    )

    for col in categorical_cols:
        input_df[col] = input_df[col].astype(str)

    try:
        prediction = float(model.predict(input_df)[0])

        st.success(
            f"Predicted {target}: {prediction:.2f}"
        )

        target_min = float(y.min())
        target_max = float(y.max())

        st.caption(
            f"Observed marks range: "
            f"{target_min:.0f} to {target_max:.0f}"
        )

        # General performance indicator
        if target_max > target_min:
            relative_score = (
                (prediction - target_min)
                / (target_max - target_min)
            )
        else:
            relative_score = 0.5

        if relative_score >= 0.75:
            st.info("Estimated performance: High")
        elif relative_score >= 0.40:
            st.info("Estimated performance: Moderate")
        else:
            st.warning(
                "Estimated performance: Needs improvement"
            )

        st.caption(
            "This is an estimate based on the dataset, "
            "not a guaranteed result."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")

# ---------------- DATASET PREVIEW ----------------
with st.expander("📁 View Dataset"):
    st.dataframe(df.head(20), use_container_width=True)

    st.write("Dataset columns:", list(df.columns))
    st.write("Target column:", target)
