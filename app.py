import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
import joblib
import streamlit.components.v1 as components

HEART_MODEL_FILE = "heart_model.pkl"
DIABETES_MODEL_FILE = "diabetes_model.json"
LIVER_MODEL_FILE = "liver_model.pkl"

st.set_page_config(
    page_title="HealthScan",
    page_icon="🩺",
    layout="centered"
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #191629;
    color: #F2EFFA;
}

.block-container {
    padding-top: 3.5rem;
    padding-bottom: 2rem;
    max-width: 950px;
}

/* Header */

.hero {
    display: flex;
    align-items: center;
    gap: 16px;
    background: linear-gradient(120deg, #40345F, #574579, #705B91);
    border: 1px solid #806DA0;
    border-radius: 16px;
    padding: 16px 22px;
    margin-bottom: 26px;
}

.hero-icon {
    width: 54px;
    height: 54px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.8rem;
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.28);
    border-radius: 14px;
}

.hero h1 {
    color: #FFFFFF;
    font-size: 1.7rem;
    font-weight: 800;
    line-height: 1.2;
    margin: 0;
    padding: 0;
}

.hero p {
    color: #E3DCEF;
    font-size: 0.9rem;
    margin: 3px 0 0;
}

/* Page headings */

.page-title {
    color: #F4EFFD;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 10px 0 7px;
}

.page-sub {
    color: #C2B9D5;
    font-size: 0.92rem;
    margin-bottom: 24px;
    line-height: 1.7;
}

/* About page */

.about-content {
    color: #DCD5EA;
    font-size: 0.95rem;
    line-height: 1.85;
}

.about-content p {
    margin: 0 0 10px;
}

.about-content h3 {
    color: #D6C1F5;
    font-size: 1.2rem;
    margin-top: 25px;
    margin-bottom: 8px;
}

.about-content h2 {
    color: #F3EDFC;
    font-size: 1.5rem;
}

.about-content ul,
.about-content ol {
    padding-left: 22px;
}

.about-content li {
    margin-bottom: 7px;
}

/* Forms */

div[data-testid="stForm"] {
    background: #252039;
    border: 1px solid #51456C;
    border-radius: 15px;
    padding: 23px;
}

div[data-testid="stForm"] label {
    color: #EAE4F5;
    font-weight: 500;
}

div[data-testid="stForm"] input {
    border-radius: 8px;
}

div[data-testid="stForm"] input,
div[data-testid="stForm"] textarea {
    color: #FFFFFF;
}

div[data-testid="stForm"] div[data-baseweb="select"] {
    border-radius: 8px;
}

/* Hide "Press Enter to submit form" text */

div[data-testid="InputInstructions"] {
    display: none;
}

/* Buttons */

div.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    width: 100%;
    background: #8065AE;
    color: #FFFFFF;
    border: 1px solid #9A80C5;
    border-radius: 9px;
    padding: 0.65rem 1rem;
    font-weight: 600;
    transition: 0.2s ease;
}

div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background: #9579C2;
    border-color: #B9A2DD;
    color: #FFFFFF;
}

/* Results */

.result-card {
    padding: 20px;
    border-radius: 13px;
    margin: 22px 0 15px;
}

.result-high {
    background: #422B3B;
    border: 1px solid #9B607E;
}

.result-low {
    background: #263B37;
    border: 1px solid #568A76;
}

.result-mid {
    background: #3F3622;
    border: 1px solid #A38C4A;
}

.result-card h2 {
    color: #FFFFFF;
    margin: 0 0 8px;
    font-size: 1.35rem;
}

.result-card p {
    color: #E3DCE8;
    margin: 0;
    line-height: 1.7;
}

/* General text */

.stMarkdown, .stCaption {
    color: #DCD5EA;
}

div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: #211B32;
    border-right: 1px solid #44385D;
}

section[data-testid="stSidebar"] > div {
    padding: 1.6rem 1rem;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: #F0EAF8;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] {
    margin-top: 5px;
    width: 100%;
}

/* hide the "Menu" label box */
section[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stWidgetLabel"] {
    display: none !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] > div,
section[data-testid="stSidebar"] div[role="radiogroup"] {
    display: flex !important;
    flex-direction: column !important;
    align-items: stretch !important;
    width: 100% !important;
    gap: 8px;
}

/* every option: same width and same height */
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    background: #302744;
    border: 1px solid #514267;
    border-radius: 9px;
    padding: 0 14px !important;
    margin: 0 !important;
    width: 100% !important;
    min-width: 100% !important;
    height: 48px;
    box-sizing: border-box;
    display: flex !important;
    align-items: center;
    cursor: pointer;
}

/* hide the round radio circle (keep the text and hidden input) */
section[data-testid="stSidebar"] div[role="radiogroup"] > label > :not(:has(p)):not(input) {
    display: none !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: #44365D;
    border-color: #9B83C2;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: #5A4680;
    border-color: #9B83C2;
}

section[data-testid="stSidebar"] hr {
    border-color: #493D60;
    margin: 20px 0;
}

/* Footer and disclaimer */

.footer-note {
    text-align: center;
    color: #B3A9C7;
    font-size: 0.82rem;
    line-height: 1.9;
    margin-top: 40px;
    padding: 18px 0;
    border-top: 1px solid #403650;
}

.note-box {
    background: #332B40;
    border: 1px solid #66577D;
    border-radius: 10px;
    padding: 15px 17px;
    margin-top: 22px;
    color: #E6DCEF;
    font-size: 0.86rem;
    line-height: 1.8;
}

/* Mobile */

@media (max-width: 700px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 3rem;
    }

    .hero {
        padding: 14px;
        gap: 12px;
    }

    .hero-icon {
        width: 44px;
        height: 44px;
        font-size: 1.4rem;
    }

    .hero h1 {
        font-size: 1.35rem;
    }

    .hero p {
        font-size: 0.8rem;
    }

    .page-title {
        font-size: 1.5rem;
    }

    div[data-testid="stForm"] {
        padding: 16px;
    }
}

/* Desktop only: bigger font sizes for sidebar, About page and forms */

@media (min-width: 701px) {

    /* Sidebar */
    section[data-testid="stSidebar"] .side-title {
        font-size: 34px !important;
    }

    section[data-testid="stSidebar"] .side-sub {
        font-size: 15px !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] > label p {
        font-size: 1.15rem !important;
    }

    section[data-testid="stSidebar"] .side-box {
        height: auto !important;
        min-height: 96px;
        padding: 14px 16px !important;
    }

    section[data-testid="stSidebar"] .side-box-title {
        font-size: 16px !important;
    }

    section[data-testid="stSidebar"] .side-box-text {
        font-size: 14.5px !important;
    }

    section[data-testid="stSidebar"] .side-dev {
        font-size: 13.5px !important;
    }

    /* About page */
    .about-content {
        font-size: 1.1rem;
    }

    .about-content h3 {
        font-size: 1.4rem;
    }

    .about-content h2 {
        font-size: 1.7rem;
    }

    .note-box {
        font-size: 1rem;
    }

    /* Forms */
    div[data-testid="stForm"] [data-testid="stWidgetLabel"] p {
        font-size: 1.1rem !important;
    }

    div[data-testid="stForm"] input {
        font-size: 1.05rem !important;
    }

    div[data-testid="stForm"] div[data-baseweb="select"] div {
        font-size: 1.05rem !important;
    }

    div[data-testid="stFormSubmitButton"] > button p {
        font-size: 1.1rem !important;
    }

    div[data-baseweb="popover"] li {
        font-size: 1.05rem !important;
    }
}

</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# Stop the Enter key from submitting the form.
# The result is shown only when the Predict button is clicked.
components.html(
    """
    <script>
    const parentWindow = window.parent;
    if (!parentWindow.noEnterSubmit) {
        parentWindow.noEnterSubmit = true;
        parentWindow.document.addEventListener("keydown", function (e) {
            if (e.key === "Enter" &&
                e.target.closest &&
                e.target.closest('[data-testid="stNumberInput"]')) {
                e.preventDefault();
                e.stopPropagation();
            }
        }, true);
    }
    </script>
    """,
    height=0
)


@st.cache_resource
def load_model(file_name):
    return joblib.load(file_name)


def hero():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-icon">🩺</div>
            <div>
                <h1>HealthScan</h1>
                <p>Smart insights for a healthier tomorrow.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def page_heading(title, subtitle):
    st.markdown(
        f'<div class="page-title">{title}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="page-sub">{subtitle}</div>',
        unsafe_allow_html=True
    )


def show_result(model, input_df, disease_name):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    percent = probability * 100

    if 35 <= percent <= 65:
        if prediction == 1:
            leaning = "slightly towards a higher risk"
        else:
            leaning = "slightly towards a lower risk"
        card = (
            '<div class="result-card result-mid">'
            '<h2>Borderline Result</h2>'
            f'<p>Your risk of {disease_name} is not clearly high or low. '
            f'The result leans {leaning}. '
            'Please get a check-up done with a doctor to know for sure.</p>'
            '</div>'
        )
    elif prediction == 1:
        card = (
            '<div class="result-card result-high">'
            '<h2>Higher Risk</h2>'
            f'<p>Your details show a higher risk of {disease_name}. '
            'Please consult a doctor for a proper check-up.</p>'
            '</div>'
        )
    else:
        card = (
            '<div class="result-card result-low">'
            '<h2>Lower Risk</h2>'
            f'<p>Your details show a lower risk of {disease_name}. '
            'Keep a healthy lifestyle, and see a doctor if you notice '
            'any symptoms.</p>'
            '</div>'
        )

    st.markdown(card, unsafe_allow_html=True)

    st.write(f"**Risk score: {percent:.1f}%**")
    st.progress(int(round(percent)))

    st.caption(
        "This is an educational project, not a medical diagnosis."
    )


# About page

def about_page():

    page_heading(
        "About HealthScan",
        "A simple tool to understand your health risk."
    )

    st.markdown(
        """
        <div class="about-content">
        <h3>What is HealthScan?</h3>
        <p>HealthScan gives you a quick idea of your risk for three common
        health problems. Fill in a short form and get your result in a
        few seconds.</p>
        <h3>What can it check?</h3>
        <ul>
        <li><b>Heart Disease:</b> uses your age, gender, chest pain type,
        blood pressure, heart rate, ECG slope and blood sugar to show
        your risk of heart disease.</li>
        <li><b>Diabetes:</b> uses your BMI, blood pressure, cholesterol,
        daily habits like exercise, food and smoking, and your general
        health to show your risk of diabetes.</li>
        <li><b>Liver Disease:</b> uses your age, BMI, alcohol use,
        smoking, family history, physical activity and liver function
        test score to show your risk of liver disease.</li>
        </ul>
        <h3>How to use it</h3>
        <ol>
        <li>Pick a check from the left side of the screen.</li>
        <li>Fill in your details. Values from a recent medical report
        give the best result.</li>
        <li>Press the Predict button.</li>
        </ol>
        <h3>Understanding your result</h3>
        <p>You will see a risk level and a risk score from 0% to 100%.
        A higher score means a higher risk. If the score is in the
        middle (35% to 65%), you will see a "Borderline Result", which
        means the risk is not clearly high or low and a doctor's
        check-up is advised.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="note-box">
            <b>Please note:</b> HealthScan is an educational project.
            Its results are only a guide, not a medical diagnosis, and
            do not replace a doctor's advice. If you have symptoms like
            chest pain, difficulty in breathing or yellow eyes, please
            see a doctor without waiting. HealthScan does not ask for
            your name, phone number or other personal details.
        </div>
        """,
        unsafe_allow_html=True
    )


# Heart Disease page

def heart_page():

    page_heading(
        "Heart Disease Prediction",
        "Enter the required health information below."
    )

    try:
        model = load_model(HEART_MODEL_FILE)

    except FileNotFoundError:
        st.error(
            f"Model file '{HEART_MODEL_FILE}' was not found. "
            "Keep it in the same folder as app.py."
        )
        return

    sex_options = {"Male": "M", "Female": "F"}

    chest_pain_options = {
        "Typical Angina (TA)": "TA",
        "Atypical Angina (ATA)": "ATA",
        "Non-Anginal Pain (NAP)": "NAP",
        "Asymptomatic (ASY)": "ASY",
    }

    yes_no_options = {"No": "N", "Yes": "Y"}

    slope_options = {
        "Up": "Up",
        "Flat": "Flat",
        "Down": "Down"
    }

    sugar_options = {"No": 0, "Yes": 1}

    with st.form("heart_form"):

        age = st.number_input(
            "Age", min_value=1, max_value=100, value=50
        )

        sex = st.selectbox(
            "Gender", list(sex_options.keys())
        )

        chest_pain = st.selectbox(
            "Chest Pain Type", list(chest_pain_options.keys())
        )

        exercise_angina = st.selectbox(
            "Chest pain during exercise?",
            list(yes_no_options.keys())
        )

        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=80, max_value=220, value=120
        )

        max_hr = st.number_input(
            "Maximum Heart Rate",
            min_value=60, max_value=220, value=150
        )

        st_slope = st.selectbox(
            "ST Slope (from ECG)", list(slope_options.keys())
        )

        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl?",
            list(sugar_options.keys())
        )

        submitted = st.form_submit_button("Predict Heart Disease")

    if submitted:

        input_df = pd.DataFrame([{
            "Age": age,
            "Sex": sex_options[sex],
            "ChestPainType": chest_pain_options[chest_pain],
            "RestingBP": resting_bp,
            "ExerciseAngina": yes_no_options[exercise_angina],
            "MaxHR": max_hr,
            "ST_Slope": slope_options[st_slope],
            "FastingBS": sugar_options[fasting_bs],
        }])

        show_result(model, input_df, "heart disease")


# Diabetes page

def diabetes_page():

    page_heading(
        "Diabetes Prediction",
        "Enter the health details below to check the diabetes prediction."
    )

    try:
        model = XGBClassifier()
        model.load_model(DIABETES_MODEL_FILE)

    except FileNotFoundError:
        st.error("Diabetes model file not found.")
        return

    except Exception as e:
        st.error(f"Error loading Diabetes model: {e}")
        return

    features = [
        "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
        "Stroke", "HeartDiseaseorAttack", "PhysActivity",
        "Fruits", "Veggies", "HvyAlcoholConsump",
        "AnyHealthcare", "NoDocbcCost", "GenHlth",
        "MentHlth", "PhysHlth", "DiffWalk", "Sex",
        "Age", "Education", "Income"
    ]

    yes_no = {"No": 0, "Yes": 1}
    sex_options = {"Female": 0, "Male": 1}

    age_options = {
        "Below 18": 0, "18 - 24": 1, "25 - 29": 2, "30 - 34": 3,
        "35 - 39": 4, "40 - 44": 5, "45 - 49": 6,
        "50 - 54": 7, "55 - 59": 8, "60 - 64": 9,
        "65 - 69": 10, "70 - 74": 11, "75 - 79": 12,
        "80 or older": 13
    }

    health_options = {
        "Excellent": 1, "Very good": 2, "Good": 3,
        "Fair": 4, "Poor": 5
    }

    education_options = {
        "Never attended school": 1,
        "Grades 1 - 8": 2,
        "Grades 9 - 11": 3,
        "Grade 12 or GED": 4,
        "College 1 - 3 years": 5,
        "College 4+ years": 6
    }

    income_options = {
        "Less than $10,000": 1,
        "$10,000 - $15,000": 2,
        "$15,000 - $20,000": 3,
        "$20,000 - $25,000": 4,
        "$25,000 - $35,000": 5,
        "$35,000 - $50,000": 6,
        "$50,000 - $75,000": 7,
        "$75,000 or more": 8
    }

    with st.form("diabetes_form"):

        col1, col2 = st.columns(2)

        with col1:
            sex = st.selectbox("Gender", list(sex_options.keys()))

        with col2:
            age = st.selectbox(
                "Age Group", list(age_options.keys()), index=7
            )

        col3, col4 = st.columns(2)

        with col3:
            bmi = st.number_input(
                "BMI", min_value=12.0, max_value=45.0,
                value=25.0, step=0.1
            )

        with col4:
            general_health = st.selectbox(
                "General Health", list(health_options.keys()), index=2
            )

        col5, col6 = st.columns(2)

        with col5:
            high_bp = st.selectbox(
                "High Blood Pressure?", list(yes_no.keys())
            )

        with col6:
            high_chol = st.selectbox(
                "High Cholesterol?", list(yes_no.keys())
            )

        col7, col8 = st.columns(2)

        with col7:
            chol_check = st.selectbox(
                "Cholesterol checked in last 5 years?",
                list(yes_no.keys()), index=1
            )

        with col8:
            stroke = st.selectbox(
                "Ever had a stroke?", list(yes_no.keys())
            )

        col9, col10 = st.columns(2)

        with col9:
            heart = st.selectbox(
                "Heart disease or heart attack?", list(yes_no.keys())
            )

        with col10:
            diff_walk = st.selectbox(
                "Difficulty walking or climbing stairs?",
                list(yes_no.keys())
            )

        col11, col12 = st.columns(2)

        with col11:
            smoker = st.selectbox(
                "Smoked at least 100 cigarettes in life?",
                list(yes_no.keys())
            )

        with col12:
            alcohol = st.selectbox(
                "Heavy alcohol consumption?", list(yes_no.keys())
            )

        col13, col14 = st.columns(2)

        with col13:
            activity = st.selectbox(
                "Physical activity in last 30 days?",
                list(yes_no.keys()), index=1
            )

        with col14:
            fruits = st.selectbox(
                "Eat fruit daily?", list(yes_no.keys()), index=1
            )

        col15, col16 = st.columns(2)

        with col15:
            veggies = st.selectbox(
                "Eat vegetables daily?", list(yes_no.keys()), index=1
            )

        with col16:
            healthcare = st.selectbox(
                "Have health insurance or coverage?",
                list(yes_no.keys()), index=1
            )

        col17, col18 = st.columns(2)

        with col17:
            no_doc_cost = st.selectbox(
                "Skipped doctor visit due to cost?",
                list(yes_no.keys())
            )

        with col18:
            mental_days = st.number_input(
                "Days of poor mental health (last 30 days)",
                min_value=0, max_value=30, value=0
            )

        col19, col20 = st.columns(2)

        with col19:
            physical_days = st.number_input(
                "Days of poor physical health (last 30 days)",
                min_value=0, max_value=30, value=0
            )

        with col20:
            education = st.selectbox(
                "Highest Education Completed", list(education_options.keys()), index=5
            )

        income = st.selectbox(
            "Yearly Household Income (in USD)", list(income_options.keys()), index=5
        )

        submitted = st.form_submit_button("Predict Diabetes")

    if submitted:

        values = {
            "HighBP": yes_no[high_bp],
            "HighChol": yes_no[high_chol],
            "CholCheck": yes_no[chol_check],
            "BMI": bmi,
            "Smoker": yes_no[smoker],
            "Stroke": yes_no[stroke],
            "HeartDiseaseorAttack": yes_no[heart],
            "PhysActivity": yes_no[activity],
            "Fruits": yes_no[fruits],
            "Veggies": yes_no[veggies],
            "HvyAlcoholConsump": yes_no[alcohol],
            "AnyHealthcare": yes_no[healthcare],
            "NoDocbcCost": yes_no[no_doc_cost],
            "GenHlth": health_options[general_health],
            "MentHlth": mental_days,
            "PhysHlth": physical_days,
            "DiffWalk": yes_no[diff_walk],
            "Sex": sex_options[sex],
            "Age": age_options[age],
            "Education": education_options[education],
            "Income": income_options[income],
        }

        input_df = pd.DataFrame([values])[features].astype(float)

        show_result(model, input_df, "diabetes")


# Liver Disease page

def liver_page():

    page_heading(
        "Liver Disease Prediction",
        "Enter the required health information below."
    )

    try:
        saved = load_model(LIVER_MODEL_FILE)

    except FileNotFoundError:
        st.error(
            f"Model file '{LIVER_MODEL_FILE}' was not found. "
            "Keep it in the same folder as app.py."
        )
        return

    model = saved["model"]
    scaler = saved["scaler"]
    features = saved["features"]

    gender_options = {"Male": 0, "Female": 1}
    yes_no_options = {"No": 0, "Yes": 1}
    genetic_options = {"Low": 0, "Medium": 1, "High": 2}

    with st.form("liver_form"):

        age = st.number_input(
            "Age", min_value=20, max_value=80, value=45
        )

        gender = st.selectbox(
            "Gender", list(gender_options.keys())
        )

        bmi = st.number_input(
            "BMI", min_value=15.0, max_value=40.0,
            value=25.0, step=0.1
        )

        genetic = st.selectbox(
            "Genetic Risk (family history)",
            list(genetic_options.keys())
        )

        alcohol = st.number_input(
            "Alcohol Consumption (units per week)",
            min_value=0.0, max_value=20.0,
            value=5.0, step=0.5
        )

        smoking = st.selectbox(
            "Do you smoke?", list(yes_no_options.keys())
        )

        activity = st.number_input(
            "Physical Activity (hours per week)",
            min_value=0.0, max_value=10.0,
            value=5.0, step=0.5
        )

        liver_test = st.number_input(
            "Liver Function Test score",
            min_value=20.0, max_value=100.0,
            value=60.0, step=1.0
        )

        submitted = st.form_submit_button("Predict Liver Disease")

    if submitted:

        input_df = pd.DataFrame([{
            "Age": age,
            "Gender": gender_options[gender],
            "BMI": bmi,
            "AlcoholConsumption": alcohol,
            "Smoking": yes_no_options[smoking],
            "GeneticRisk": genetic_options[genetic],
            "PhysicalActivity": activity,
            "LiverFunctionTest": liver_test,
        }])

        input_df = input_df[features]

        scaled_df = pd.DataFrame(
            scaler.transform(input_df),
            columns=features
        )

        show_result(model, scaled_df, "liver disease")


# Sidebar

st.sidebar.markdown(
    """
    <div style="text-align:center; padding:12px 5px 18px;">
        <h1 class="side-title" style="font-size:28px; margin-bottom:4px; color:#FFFFFF;">
            HealthScan
        </h1>
        <p class="side-sub" style="font-size:12px; color:#C9BDDE;">
            Smart Health Predictions
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Menu",
    [
        "About",
        "Heart Disease",
        "Diabetes",
        "Liver Disease"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div class="side-box" style="
        background:#302744;
        border:1px solid #514267;
        padding:0 14px;
        border-radius:9px;
        margin-top:10px;
        width:100%;
        height:96px;
        box-sizing:border-box;
        display:flex;
        flex-direction:column;
        justify-content:center;
    ">
        <p class="side-box-title" style="font-size:13px; font-weight:700; margin:0 0 4px; color:#D6C1F5;">
            About HealthScan
        </p>
        <p class="side-box-text" style="font-size:12px; color:#D0C7DF; line-height:1.5; margin:0;">
            An educational project for health risk awareness.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <p class="side-dev" style="
        font-size:11px;
        color:#B3A9C7;
        text-align:center;
        margin-top:25px;
        line-height:1.7;
    ">
        Developed by GEHU Students
    </p>
    """,
    unsafe_allow_html=True
)


# Main application

hero()

if page == "About":
    about_page()

elif page == "Heart Disease":
    heart_page()

elif page == "Diabetes":
    diabetes_page()

elif page == "Liver Disease":
    liver_page()


# Footer

st.markdown(
    """
    <div class="footer-note">
        Made by <b>Nikita, Shakshi & Priyanka</b>
        <br>
        Graphic Era Hill University (GEHU)
        <br><br>
        HealthScan
    </div>
    """,
    unsafe_allow_html=True
)
