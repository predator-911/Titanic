import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model
model = joblib.load('model.pkl')

# Function to preprocess the data
def preprocess_data(df):
    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    if 'Embarked' in df.columns:
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Convert categorical variables to numerical
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    if 'Embarked' in df.columns:
        df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    
    # Drop columns that won't be used in the model
    df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True, errors='ignore')
    
    return df

# Function to make predictions
def make_prediction(input_data):
    input_data = preprocess_data(input_data)
    predictions = model.predict(input_data)
    return predictions

# Streamlit app layout
def main():
    st.title("Titanic Survival Prediction")

    # Option to upload a CSV file
    uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type=["csv"])
    
    if uploaded_file:
        # Handle the uploaded file
        input_df = pd.read_csv(uploaded_file)
        st.write("Input Data:")
        st.write(input_df)

        predictions = make_prediction(input_df)
        output_df = input_df.copy()
        output_df['Survived'] = predictions
        st.write("Predictions:")
        st.write(output_df)
        
        # Option to download the prediction results
        csv = output_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download predictions as CSV",
            data=csv,
            file_name='titanic_predictions.csv',
            mime='text/csv',
        )
    else:
        # If no file is uploaded, ask for user input for a single prediction
        st.write("Provide details for a single passenger:")

        Pclass = st.selectbox("Pclass", [1, 2, 3], index=0)
        Sex = st.selectbox("Sex", ["male", "female"])
        Age = st.slider("Age", 0, 100, 30)
        SibSp = st.number_input("SibSp", min_value=0, max_value=10, value=0)
        Parch = st.number_input("Parch", min_value=0, max_value=10, value=0)
        Fare = st.number_input("Fare", min_value=0.0, max_value=1000.0, value=30.0)
        Embarked = st.selectbox("Embarked", ["C", "Q", "S"])

        # Convert input to DataFrame
        input_data = pd.DataFrame({
            'Pclass': [Pclass],
            'Sex': [Sex],
            'Age': [Age],
            'SibSp': [SibSp],
            'Parch': [Parch],
            'Fare': [Fare],
            'Embarked': [Embarked]
        })

        st.write("Input Data:")
        st.write(input_data)

        # Predict
        prediction = make_prediction(input_data)[0]
        st.write(f"Survived: {'Yes' if prediction == 1 else 'No'}")

if __name__ == "__main__":
    main()
