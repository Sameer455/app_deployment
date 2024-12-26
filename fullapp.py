import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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
        'n_estimators': [50, 100, 150],  # Number of trees in the forest
        'max_depth': [10, 20, None],      # Maximum depth of each tree
        'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
        'min_samples_leaf': [1, 2, 4],    # Minimum samples required to be at a leaf node
        'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider when splitting a node
    }

    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Best model after tuning
    best_model = grid_search.best_estimator_

    # Evaluate the model
    predictions = best_model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    # Return the trained model and accuracy
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
st.title("Heart Disease Prediction with Dietary Recommendations")

# Load and train the model
with st.spinner("Training model..."):
    model, accuracy = load_and_train_model()
st.success(f"Model trained successfully! Accuracy: {accuracy:.2f}")

# Input form
st.header("Patient Details")
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", options=["Female", "Male"], index=1)
cp = st.selectbox("Chest Pain Type (0-3)", options=list(range(4)), index=0)
trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=200, value=120)
chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No", "Yes"], index=0)
restecg = st.selectbox("Resting ECG Results (0-2)", options=list(range(3)), index=0)
thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=220, value=150)
exang = st.selectbox("Exercise-Induced Angina", options=["No", "Yes"], index=0)
oldpeak = st.number_input("ST Depression Induced", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of Peak Exercise (0-2)", options=list(range(3)), index=1)
ca = st.selectbox("Number of Major Vessels (0-4)", options=list(range(5)), index=0)
thal = st.selectbox(
    "Thalassemia",
    options=["Normal", "Fixed Defect", "Reversible Defect"]
)
thal_mapping = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
thal = thal_mapping[thal] 

# Prediction
if st.button("Predict"):
    # Prepare input data
    input_data = np.array([[age, int(sex == "Male"), cp, trestbps, chol, int(fbs == "Yes"), restecg,
                            thalach, int(exang == "Yes"), oldpeak, slope, ca, thal + 1]])
    prediction = model.predict(input_data)

    # Display result
    result = "Positive for Heart Disease" if prediction[0] == 1 else "Negative for Heart Disease"
    st.subheader(f"Prediction: {result}")

    # Suggest diet
    diet_recommendations = suggest_diet(prediction[0])
    st.subheader("Dietary Recommendations")
    for item in diet_recommendations:
        st.write(f"- {item}")
