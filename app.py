import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load the pre-trained model
model = joblib.load('titanic_model.pkl')

def preprocess_data(df):
    # Handle missing values for Age and Fare
    imputer = SimpleImputer(strategy='median')
    if 'Age' in df.columns:
        df['Age'] = imputer.fit_transform(df[['Age']])
    if 'Fare' in df.columns:
        df['Fare'] = imputer.fit_transform(df[['Fare']])
    
    # Handle missing values for Embarked, if the column exists
    if 'Embarked' in df.columns:
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Convert categorical variables to numerical
    if 'Sex' in df.columns:
        df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    
    # One-hot encode Embarked, if the column exists
    if 'Embarked' in df.columns:
        df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    
    # Ensure the dataframe has the correct structure
    expected_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0  # Add missing columns with default value 0
    
    # Drop columns that won't be used in the model, if they exist
    df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True, errors='ignore')
    
    return df

def main():
    st.title("Titanic Survival Prediction")

    # Option to upload a CSV file
    st.subheader("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)
        test_df = preprocess_data(test_df)
        
        # Prepare the test data
        X_test = test_df.drop(['PassengerId'], axis=1)
        
        # Predict using the trained model
        predictions = model.predict(X_test)
        
        # Create a DataFrame with the results
        output = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': predictions})
        
        # Display results
        st.write("Prediction Results:")
        st.write(output)
        
        # Allow user to download the results
        st.download_button(
            label="Download Predictions",
            data=output.to_csv(index=False),
            file_name='submission.csv',
            mime='text/csv'
        )

    # Option to input individual features
    st.subheader("Input Individual Passenger Data")

    # Create input fields
    pclass = st.selectbox("Pclass", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    sibsp = st.number_input("SibSp (Number of Siblings/Spouses Aboard)", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parch (Number of Parents/Children Aboard)", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare", min_value=0.0, value=10.0)
    embarked = st.selectbox("Embarked", ["C", "Q", "S"])
    
    # Convert categorical input to match preprocessing
    sex_encoded = LabelEncoder().fit(["male", "female"]).transform([sex])[0]
    embarked_encoded = pd.get_dummies([embarked], columns=["Embarked"], drop_first=True).reindex(columns=["Embarked_Q", "Embarked_S"], fill_value=0).values[0]
    
    if st.button("Predict"):
        # Prepare the input data for prediction
        input_df = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [sex_encoded],
            'Age': [age],
            'SibSp': [sibsp],
            'Parch': [parch],
            'Fare': [fare],
            'Embarked_Q': [embarked_encoded[0]],
            'Embarked_S': [embarked_encoded[1]]
        })
        
        # Predict using the trained model
        prediction = model.predict(input_df)[0]
        
        st.write(f"Survived: {'Yes' if prediction == 1 else 'No'}")

if __name__ == "__main__":
    main()
