
# ============================================================
# SMARTHEALTHCARE - BIG DATA ANALYTICS PROJECT
# Modern Healthcare Analytics Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import io

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartHealthcare | BDA Dashboard",
    page_icon="🏥",
    layout="wide"
)

# ============================================================
# MODERN UI THEME
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 10% 5%,
            rgba(14, 165, 233, 0.18),
            transparent 35%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(20, 184, 166, 0.14),
            transparent 35%
        ),
        linear-gradient(
            135deg,
            #071426 0%,
            #0b1f35 50%,
            #102d42 100%
        );
    color: #f1f5f9;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

h1, h2, h3 {
    color: #f0fdfa !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}

p, label, .stMarkdown, .stCaption {
    color: #d5e4f0;
}

[data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        rgba(20, 55, 78, 0.95),
        rgba(12, 36, 56, 0.95)
    );
    border: 1px solid rgba(94, 234, 212, 0.25);
    padding: 22px 20px;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
    transition: 0.25s ease;
}

[data-testid="stMetric"]:hover {
    border-color: #2dd4bf;
    transform: translateY(-3px);
}

[data-testid="stMetricValue"] {
    color: #5eead4 !important;
    font-weight: 800;
}

[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
}

.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(90deg, #0d9488, #0284c7);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.3rem;
    font-weight: 600;
    transition: 0.2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: linear-gradient(90deg, #14b8a6, #38bdf8);
    border: none;
    box-shadow: 0 4px 18px rgba(45, 212, 191, 0.25);
}

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"] {
    background-color: #122b40;
    color: #f8fafc;
    border-radius: 9px;
}

[data-testid="stFileUploader"] {
    background: rgba(15, 45, 65, 0.75);
    border: 1px dashed #2dd4bf;
    border-radius: 14px;
    padding: 12px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.2);
}

.stTabs [data-baseweb="tab"] {
    color: #cbd5e1;
    border-radius: 8px 8px 0 0;
}

.stTabs [aria-selected="true"] {
    color: #5eead4 !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #081525,
        #102b40
    );
    border-right: 1px solid rgba(94, 234, 212, 0.18);
}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(94, 234, 212, 0.2);
    border-radius: 12px;
    overflow: hidden;
}

hr {
    border-color: rgba(148, 163, 184, 0.2);
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

FEATURES = [
    "Age",
    "BMI",
    "Glucose",
    "Systolic_BP",
    "Diastolic_BP",
    "Heart_Rate",
    "Cholesterol"
]

REQUIRED_COLUMNS = FEATURES + ["Risk_Label"]

# These are example ranges for synthetic educational data.
# They are NOT validated clinical thresholds.
RISK_FEATURES = FEATURES


# ============================================================
# FUNCTION 1: GENERATE SAMPLE HEALTHCARE DATA
# ============================================================

@st.cache_data
def generate_sample_data(number_of_patients=1000):

    rng = np.random.default_rng(42)

    data = pd.DataFrame({
        "Patient_ID": [
            f"P{i:04d}" for i in range(1, number_of_patients + 1)
        ],
        "Age": rng.integers(18, 85, number_of_patients),
        "BMI": np.round(
            rng.normal(26, 5, number_of_patients).clip(15, 45), 1
        ),
        "Glucose": rng.integers(
            70, 201, number_of_patients
        ),
        "Systolic_BP": rng.integers(
            90, 181, number_of_patients
        ),
        "Diastolic_BP": rng.integers(
            60, 111, number_of_patients
        ),
        "Heart_Rate": rng.integers(
            55, 121, number_of_patients
        ),
        "Cholesterol": rng.integers(
            120, 301, number_of_patients
        )
    })

    # Illustrative labels created from a simple synthetic rule.
    # They are not real patient diagnoses.
    score = (
        (data["Age"] >= 50).astype(int)
        + (data["BMI"] >= 30).astype(int)
        + (data["Glucose"] >= 140).astype(int)
        + (data["Systolic_BP"] >= 140).astype(int)
        + (data["Cholesterol"] >= 240).astype(int)
    )

    data["Risk_Label"] = (score >= 2).astype(int)

    data["Risk_Category"] = data["Risk_Label"].map({
        0: "Lower indicator score",
        1: "Higher indicator score"
    })

    return data


# ============================================================
# FUNCTION 2: CLEAN AND VALIDATE DATA
# ============================================================

def prepare_data(data):

    data = data.copy()

    # Remove extra spaces from column names.
    data.columns = data.columns.astype(str).str.strip()

    # Convert numeric fields into numbers.
    for column in FEATURES:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column], errors="coerce"
            )

    # Missing values are filled with column medians.
    for column in FEATURES:
        if column in data.columns:
            median_value = data[column].median()

            if pd.isna(median_value):
                median_value = 0

            data[column] = data[column].fillna(median_value)

    # Ensure a patient identifier exists.
    if "Patient_ID" not in data.columns:
        data["Patient_ID"] = [
            f"P{i:04d}" for i in range(1, len(data) + 1)
        ]

    # Create illustrative labels if they are absent.
    if "Risk_Label" not in data.columns:
        score = (
            (data["Age"] >= 50).astype(int)
            + (data["BMI"] >= 30).astype(int)
            + (data["Glucose"] >= 140).astype(int)
            + (data["Systolic_BP"] >= 140).astype(int)
            + (data["Cholesterol"] >= 240).astype(int)
        )

        data["Risk_Label"] = (score >= 2).astype(int)

    data["Risk_Label"] = pd.to_numeric(
        data["Risk_Label"], errors="coerce"
    )

    data = data.dropna(subset=["Risk_Label"])

    data["Risk_Label"] = data["Risk_Label"].astype(int)

    # The demo model supports binary labels 0 and 1.
    data = data[data["Risk_Label"].isin([0, 1])].copy()

    data["Risk_Category"] = data["Risk_Label"].map({
        0: "Lower indicator score",
        1: "Higher indicator score"
    })

    return data.reset_index(drop=True)


# ============================================================
# FUNCTION 3: FIND SIMILAR PATIENT RECORDS USING KNN
# ============================================================

def find_similar_patients(data, patient_values, number_of_neighbors=5):

    if len(data) == 0:
        return pd.DataFrame()

    # Standardize features so different units are comparable.
    scaler = StandardScaler()

    X = scaler.fit_transform(data[FEATURES])

    query_df = pd.DataFrame(
        [patient_values],
        columns=FEATURES
    )

    query = scaler.transform(query_df)

    # Do not include the query patient itself.
    k = min(number_of_neighbors, len(data))

    model = NearestNeighbors(
        n_neighbors=k,
        metric="euclidean"
    )

    model.fit(X)

    distances, indices = model.kneighbors(query)

    results = data.iloc[indices[0]].copy()

    results["Distance"] = np.round(distances[0], 3)

    results["Similarity_Score"] = np.round(
        100 / (1 + distances[0]), 2
    )

    return results.sort_values("Distance")


# ============================================================
# FUNCTION 4: TRAIN EDUCATIONAL LOGISTIC REGRESSION MODEL
# ============================================================

def train_risk_model(data):

    X = data[FEATURES]
    y = data["Risk_Label"]

    # Both classes are required for binary classification.
    if y.nunique() < 2:
        return None, None, None

    # A stratified split is used when there are enough records.
    if len(data) >= 10 and y.value_counts().min() >= 2:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)

    return model, scaler, accuracy


# ============================================================
# FUNCTION 5: CREATE CSV REPORT
# ============================================================

def create_csv_report(data):

    buffer = io.StringIO()

    data.to_csv(buffer, index=False)

    return buffer.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("# 🏥 SmartHealthcare")

    st.caption("Big Data Analytics Dashboard")

    st.divider()

    st.markdown("### Navigation")

    page = st.radio(
        "Select a section",
        [
            "Dashboard",
            "Patient Analysis",
            "Data Explorer",
            "Database"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### Data source")

    data_source = st.radio(
        "Choose your data",
        ["Sample Data", "Upload CSV"]
    )

    uploaded_file = None

    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload healthcare CSV",
            type=["csv"]
        )

    st.divider()

    st.caption(
        "Academic demonstration only. "
        "Not for medical diagnosis."
    )


# ============================================================
# LOAD DATA
# ============================================================

if data_source == "Upload CSV" and uploaded_file is not None:

    try:
        raw_data = pd.read_csv(uploaded_file)

        missing_columns = [
            column for column in FEATURES
            if column not in raw_data.columns
        ]

        if missing_columns:
            st.error(
                "Your CSV is missing these required columns: "
                + ", ".join(missing_columns)
            )
            st.stop()

        data = prepare_data(raw_data)

        if data.empty:
            st.error("No valid patient records were found.")
            st.stop()

    except Exception as error:
        st.error(f"Could not read the CSV: {error}")
        st.stop()

elif data_source == "Upload CSV":

    st.info(
        "Upload a CSV file using the sidebar, "
        "or select Sample Data."
    )
    st.stop()

else:

    data = generate_sample_data()

data = prepare_data(data)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div style="
        padding: 26px 30px;
        border-radius: 22px;
        background: linear-gradient(
            110deg,
            rgba(13, 148, 136, 0.22),
            rgba(2, 132, 199, 0.16)
        );
        border: 1px solid rgba(94, 234, 212, 0.30);
        margin-bottom: 25px;
    ">
        <div style="
            color: #5eead4;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 2px;
        ">
            BIG DATA ANALYTICS PROJECT
        </div>
        <h1 style="
            margin: 10px 0 6px 0;
            font-size: 36px;
        ">
            🏥 SmartHealthcare
        </h1>
        <p style="
            margin: 0;
            font-size: 16px;
        ">
            Healthcare Data Analysis, Patient Similarity
            and Interactive Insights
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    f"Dashboard last refreshed: "
    f"{datetime.now().strftime('%d %b %Y, %I:%M %p')}"
)

st.divider()


# ============================================================
# TRAIN MODEL
# ============================================================

model, model_scaler, model_accuracy = train_risk_model(data)


# ============================================================
# PAGE 1: DASHBOARD
# ============================================================

if page == "Dashboard":

    st.subheader("📊 Healthcare Overview")

    # Summary metrics.
    total_patients = len(data)

    average_age = data["Age"].mean()

    average_bmi = data["BMI"].mean()

    higher_indicator_count = int(
        (data["Risk_Label"] == 1).sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Records",
        f"{total_patients:,}",
        help="Number of patient records in the dataset"
    )

    col2.metric(
        "Average Age",
        f"{average_age:.1f} years"
    )

    col3.metric(
        "Average BMI",
        f"{average_bmi:.1f}"
    )

    col4.metric(
        "Higher Indicator Score",
        f"{higher_indicator_count:,}",
        help="Records assigned label 1 by the educational model"
    )

    st.write("")

    # Charts.
    left, right = st.columns(2)

    with left:

        st.markdown("### Risk-label distribution")

        category_counts = (
            data["Risk_Category"]
            .value_counts()
            .rename_axis("Category")
            .reset_index(name="Records")
        )

        fig = px.pie(
            category_counts,
            names="Category",
            values="Records",
            hole=0.55,
            title="Distribution of illustrative labels"
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            legend_title_text="Label"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:

        st.markdown("### Age distribution")

        fig = px.histogram(
            data,
            x="Age",
            nbins=20,
            title="Patient records by age",
            labels={"Age": "Age (years)"},
            color_discrete_sequence=["#2dd4bf"]
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0"
        )

        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:

        st.markdown("### BMI vs Glucose")

        fig = px.scatter(
            data,
            x="BMI",
            y="Glucose",
            color="Risk_Category",
            hover_data=["Patient_ID", "Age"],
            title="Relationship between BMI and glucose"
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:

        st.markdown("### Average health indicators")

        averages = pd.DataFrame({
            "Indicator": [
                "BMI",
                "Glucose",
                "Systolic BP",
                "Diastolic BP",
                "Heart Rate",
                "Cholesterol"
            ],
            "Average": [
                data["BMI"].mean(),
                data["Glucose"].mean(),
                data["Systolic_BP"].mean(),
                data["Diastolic_BP"].mean(),
                data["Heart_Rate"].mean(),
                data["Cholesterol"].mean()
            ]
        })

        fig = px.bar(
            averages,
            x="Indicator",
            y="Average",
            title="Average value by indicator",
            color="Indicator"
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Recent patient records")

    st.dataframe(
        data.head(10),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Healthcare Report",
        data=create_csv_report(data),
        file_name="smarthealthcare_report.csv",
        mime="text/csv"
    )


# ============================================================
# PAGE 2: PATIENT ANALYSIS
# ============================================================

elif page == "Patient Analysis":

    st.subheader("🔎 Patient Analysis")

    st.write(
        "Enter example health indicators to compare a record "
        "with the dataset and demonstrate the classification model."
    )

    with st.form("patient_form"):

        c1, c2, c3 = st.columns(3)

        with c1:

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=120,
                value=35
            )

            bmi = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=80.0,
                value=24.0,
                step=0.1
            )

            glucose = st.number_input(
                "Glucose",
                min_value=20,
                max_value=500,
                value=100
            )

        with c2:

            systolic_bp = st.number_input(
                "Systolic BP",
                min_value=50,
                max_value=300,
                value=120
            )

            diastolic_bp = st.number_input(
                "Diastolic BP",
                min_value=30,
                max_value=200,
                value=80
            )

        with c3:

            heart_rate = st.number_input(
                "Heart Rate",
                min_value=20,
                max_value=250,
                value=75
            )

            cholesterol = st.number_input(
                "Cholesterol",
                min_value=50,
                max_value=700,
                value=180
            )

        submitted = st.form_submit_button(
            "🔍 Analyze Patient",
            use_container_width=True
        )

    if submitted:

        patient_values = [
            age,
            bmi,
            glucose,
            systolic_bp,
            diastolic_bp,
            heart_rate,
            cholesterol
        ]

        patient_df = pd.DataFrame(
            [patient_values],
            columns=FEATURES
        )

        st.markdown("### Entered patient indicators")

        st.dataframe(
            patient_df,
            use_container_width=True,
            hide_index=True
        )

        # KNN comparison.
        st.markdown("### 👥 Similar patient records")

        similar = find_similar_patients(
            data,
            patient_values,
            number_of_neighbors=5
        )

        if not similar.empty:

            st.dataframe(
                similar,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Similarity score is an illustrative transformation "
                "of standardized Euclidean distance, not a clinical score."
            )

        # Logistic regression.
        st.markdown("### 🧠 Educational model output")

        if model is not None:

            patient_scaled = model_scaler.transform(
                patient_df[FEATURES]
            )

            prediction = int(model.predict(patient_scaled)[0])

            probability = model.predict_proba(
                patient_scaled
            )[0]

            positive_index = list(model.classes_).index(1)

            estimated_score = probability[positive_index] * 100

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                if prediction == 1:
                    st.warning(
                        "The model assigned the higher "
                        "indicator-score category."
                    )
                else:
                    st.success(
                        "The model assigned the lower "
                        "indicator-score category."
                    )

            with result_col2:

                st.metric(
                    "Model-estimated class-1 probability",
                    f"{estimated_score:.1f}%"
                )

            st.caption(
                "This estimate is based on synthetic or user-provided "
                "data and illustrative labels. It is not a validated "
                "medical risk prediction or diagnosis."
            )

        else:

            st.warning(
                "The model requires records containing both "
                "label classes (0 and 1)."
            )


# ============================================================
# PAGE 3: DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.subheader("🗂️ Healthcare Data Explorer")

    st.write(
        "Explore the dataset, filter records, and review "
        "descriptive statistics."
    )

    with st.expander("Filter patient records", expanded=True):

        min_age = int(data["Age"].min())
        max_age = int(data["Age"].max())

        age_range = st.slider(
            "Select age range",
            min_value=min_age,
            max_value=max_age,
            value=(min_age, max_age)
        )

        selected_category = st.multiselect(
            "Select indicator-score category",
            options=data["Risk_Category"].unique().tolist(),
            default=data["Risk_Category"].unique().tolist()
        )

    filtered_data = data[
        data["Age"].between(age_range[0], age_range[1])
        & data["Risk_Category"].isin(selected_category)
    ]

    st.metric("Filtered records", len(filtered_data))

    st.dataframe(
        filtered_data,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Descriptive statistics")

    st.dataframe(
        filtered_data[FEATURES].describe().round(2),
        use_container_width=True
    )

    st.download_button(
        "⬇️ Download Filtered Data",
        data=create_csv_report(filtered_data),
        file_name="filtered_healthcare_data.csv",
        mime="text/csv"
    )


# ============================================================
# PAGE 4: OPTIONAL MONGODB DATABASE
# ============================================================

elif page == "Database":

    st.subheader("🗄️ MongoDB Integration")

    st.write(
        "This optional section can save and retrieve records "
        "from a MongoDB database."
    )

    st.info(
        "Install pymongo and provide your MongoDB connection URI "
        "to enable database operations."
    )

    mongo_uri = st.text_input(
        "MongoDB connection URI",
        type="password",
        placeholder="mongodb://localhost:27017"
    )

    database_name = st.text_input(
        "Database name",
        value="SmartHealthcare"
    )

    collection_name = st.text_input(
        "Collection name",
        value="patient_records"
    )

    save_limit = st.number_input(
        "Maximum number of records to save",
        min_value=1,
        max_value=10000,
        value=min(100, max(1, len(data)))
    )

    if st.button("💾 Save records to MongoDB"):

        if not mongo_uri.strip():

            st.error("Enter a MongoDB connection URI first.")

        else:

            try:

                from pymongo import MongoClient

                client = MongoClient(
                    mongo_uri,
                    serverSelectionTimeoutMS=5000
                )

                # Verify that the server is reachable.
                client.admin.command("ping")

                collection = client[database_name][collection_name]

                records = data.head(int(save_limit)).copy()

                # Avoid duplicate Patient_ID records.
                for record in records.to_dict("records"):

                    patient_id = str(record["Patient_ID"])

                    collection.update_one(
                        {"Patient_ID": patient_id},
                        {"$set": record},
                        upsert=True
                    )

                st.success(
                    f"Saved or updated {len(records)} records."
                )

                client.close()

            except ImportError:

                st.error(
                    "pymongo is not installed. Run: "
                    "pip install pymongo"
                )

            except Exception as error:

                st.error(
                    f"MongoDB operation failed: {error}"
                )

    st.divider()

    if st.button("📥 Retrieve records from MongoDB"):

        if not mongo_uri.strip():

            st.error("Enter a MongoDB connection URI first.")

        else:

            try:

                from pymongo import MongoClient

                client = MongoClient(
                    mongo_uri,
                    serverSelectionTimeoutMS=5000
                )

                client.admin.command("ping")

                collection = client[database_name][collection_name]

                records = list(
                    collection.find(
                        {},
                        {"_id": 0}
                    ).limit(1000)
                )

                if records:

                    retrieved_data = pd.DataFrame(records)

                    st.success(
                        f"Retrieved {len(retrieved_data)} records."
                    )

                    st.dataframe(
                        retrieved_data,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.info("No records found in this collection.")

                client.close()

            except ImportError:

                st.error(
                    "pymongo is not installed. Run: "
                    "pip install pymongo"
                )

            except Exception as error:

                st.error(
                    f"MongoDB operation failed: {error}"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align: center;
        color: #94a3b8;
        padding: 12px;
        font-size: 13px;
    ">
        SmartHealthcare | Big Data Analytics Project<br>
        Educational demonstration using healthcare data
    </div>
    """,
    unsafe_allow_html=True
)