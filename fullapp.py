import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# Function to load and train the model
def load_and_train_model():
    url = "heart.csv"  # Update path
    df = pd.read_csv(url)

    # Define features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model with hyperparameter tuning
    model = RandomForestClassifier(random_state=42)

    # Set up hyperparameter grid for tuning
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
    }

    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Best model after tuning
    best_model = grid_search.best_estimator_

    # Evaluate the model
    predictions = best_model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    return best_model, acc

# Function to suggest diet based on prediction
def suggest_diet(prediction):
    if prediction == 1:
        return [
            "Focus on a heart-healthy diet: fruits, vegetables, whole grains.",
            "Avoid saturated fats, trans fats, and processed foods.",
            "Include omega-3 fatty acids (e.g., salmon, walnuts).",
            "Reduce sodium intake to control blood pressure.",
            "Maintain a healthy weight and monitor portion sizes."
        ]
    else:
        return [
            "Continue with a balanced diet rich in nutrients.",
            "Stay active and maintain a healthy weight.",
            "Regularly include fruits, vegetables, and lean proteins.",
            "Limit intake of sugary drinks and junk food."
        ]

# Main Streamlit app
st.set_page_config(page_title="Heart Disease Prediction App", layout="centered")
st.title("💓 Heart Disease Prediction App")
st.markdown("This app predicts the likelihood of heart disease and provides dietary recommendations for your health.")

# Load and train the model
with st.spinner("Training model..."):
    model, accuracy = load_and_train_model()
st.success(f"Model trained successfully! Accuracy: {accuracy:.2f}")

st.markdown("---")

# Input form
st.header("📋 Patient Details")
st.markdown("Please fill in the details below:")

age = st.slider("Age", min_value=1, max_value=120, value=30, help="Enter the patient's age.")
sex = st.radio("Sex", options=["Female", "Male"], index=1, help="Select the patient's gender.")
cp = st.selectbox("Chest Pain Type", options=["0: Typical Angina", "1: Atypical Angina", "2: Non-anginal Pain", "3: Asymptomatic"], index=0)
trestbps = st.slider("Resting Blood Pressure (mmHg)", min_value=50, max_value=200, value=120, help="Resting blood pressure in mmHg.")
chol = st.slider("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200, help="Serum cholesterol in mg/dl.")
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", options=["No", "Yes"], index=0, help="Is fasting blood sugar higher than 120 mg/dl?")
restecg = st.selectbox("Resting ECG Results", options=["0: Normal", "1: ST-T Wave Abnormality", "2: Left Ventricular Hypertrophy"], index=0)
thalach = st.slider("Max Heart Rate Achieved", min_value=50, max_value=220, value=150, help="Maximum heart rate achieved.")
exang = st.radio("Exercise-Induced Angina", options=["No", "Yes"], index=0, help="Does exercise cause chest pain?")
oldpeak = st.slider("ST Depression Induced", min_value=0.0, max_value=10.0, value=1.0, step=0.1, help="ST depression induced by exercise relative to rest.")
slope = st.selectbox("Slope of Peak Exercise", options=["0: Upsloping", "1: Flat", "2: Downsloping"], index=1)
ca = st.slider("Number of Major Vessels (0-4)", min_value=0, max_value=4, value=0, help="Number of major vessels colored by fluoroscopy.")
thal = st.selectbox("Thalassemia", options=["Normal", "Fixed Defect", "Reversible Defect"])
thal_mapping = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
thal = thal_mapping[thal]

# Prediction
if st.button("🔍 Predict"):
    # Prepare input data
    input_data = np.array([[age, int(sex == "Male"), int(cp[0]), trestbps, chol, int(fbs == "Yes"), int(restecg[0]),
                            thalach, int(exang == "Yes"), oldpeak, int(slope[0]), ca, thal]])
    prediction = model.predict(input_data)

    # Display result
    result = "Positive for Heart Disease" if prediction[0] == 1 else "Negative for Heart Disease"
    result_color = "red" if prediction[0] == 1 else "green"
    st.markdown(f"### **Prediction:** <span style='color:{result_color}'>{result}</span>", unsafe_allow_html=True)

    # Suggest diet
    with st.expander("🍎 Dietary Recommendations", expanded=True):
        diet_recommendations = suggest_diet(prediction[0])
        for item in diet_recommendations:
            st.write(f"- {item}")

st.markdown("---")
st.markdown("💡 *Disclaimer: This app is for informational purposes only and is not a substitute for professional medical advice.*")
