
import streamlit as st
import pandas as pd

features = ['EDUCATION', 'AGE','LIMIT_BAL','PAY_JUNE',
       'PAY_MAY', 'PAY_APRIL', 'PAY_MARCH', 'PAY_FEBRUARY', 'PAY_JANUARY','SEX','MARRIAGE','PAY_AMT_JUNE',
       'PAY_AMT_MAY', 'PAY_AMT_APRIL', 'PAY_AMT_MARCH', 'PAY_AMT_FEBRUARY', 'PAY_AMT_JANUARY','BILL_AMT_APRIL', 'BILL_AMT_FEBRUARY', 'BILL_AMT_JANUARY',
       'BILL_AMT_JUNE', 'BILL_AMT_MARCH', 'BILL_AMT_MAY']

# Function to train model
#def train_model():
       


# Function to make predictions
def make_prediction(input_data):
    # Add your data preprocessing steps if needed
    # Example: input_data = preprocess_data(input_data)
    X_test = input_data[features]

    # Make predictions using the loaded model
    #prediction = bbest_model.predict_proba(X_test)[:, 1]
    prediction = [0.10665705, 0.09051057, 0.89876176]
    return prediction

# Streamlit app
def main():
       st.title("Prediction App")
       
       # Read a default CSV file located in the same GitHub repository
       default_file_path = 'train_customers.csv'  # Update with your actual file path
       default_data = pd.read_csv(default_file_path)
       
       # Display the default data
       #st.write("Default Data:")
       #st.write(default_data)

 
       # Upload a CSV file
       st.write("Upload a CSV file for prediction.")
       uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
       
       if uploaded_file is not None:
              # Read the uploaded CSV file into a Pandas DataFrame
              input_data = pd.read_csv(uploaded_file)
              
              # Display the uploaded data
              #st.write("Uploaded Data:")
              #st.write(input_data)
              
              # Make predictions
              prediction = make_prediction(input_data)
              
              # Display the prediction
              st.write("Prediction:", prediction)
              
          
        

# Run the Streamlit app
if __name__ == "__main__":
    main()
