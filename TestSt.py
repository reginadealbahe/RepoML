
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # linear algebra

features = ['EDUCATION', 'AGE','LIMIT_BAL','PAY_JUNE',
       'PAY_MAY', 'PAY_APRIL', 'PAY_MARCH', 'PAY_FEBRUARY', 'PAY_JANUARY','SEX','MARRIAGE','PAY_AMT_JUNE',
       'PAY_AMT_MAY', 'PAY_AMT_APRIL', 'PAY_AMT_MARCH', 'PAY_AMT_FEBRUARY', 'PAY_AMT_JANUARY','BILL_AMT_APRIL', 'BILL_AMT_FEBRUARY', 'BILL_AMT_JANUARY',
       'BILL_AMT_JUNE', 'BILL_AMT_MARCH', 'BILL_AMT_MAY']

# Function to train model
def train_model():
       st.write("Training model...")

       df_series = pd.read_csv('train_series.csv')
       # Pivot the time-dependent columns (PAY, BILL_AMT, PAY_AMT) to build columns for each month
       df_pivoted = df_series.pivot(index='ID', columns='MONTH', values=['PAY', 'BILL_AMT', 'PAY_AMT'])
       df_pivoted.columns = [f'{col}_{month}' for col, month in df_pivoted.columns]

       # Reset the index to get 'ID' as a separate column
       df_pivoted.reset_index(inplace=True)
       # Load the "train_customers.csv" file into a DataFrame
       df_customers = pd.read_csv('train_customers.csv')
       
       # Merge the two DataFrames on the 'ID' column
       df_customers = pd.merge(df_customers, df_pivoted, on='ID', how='inner')

       # Load the "train_target.csv" file into a DataFrame
       df_target = pd.read_csv('train_target.csv')
       df_data_train = pd.merge(df_customers, df_target, on='ID', how='inner')

       y = data_train.DEFAULT_JULY #target variable
       X = data_train[features]

       from sklearn.datasets import fetch_covtype
       from sklearn.model_selection import RandomizedSearchCV
       from sklearn.tree import DecisionTreeClassifier
       from sklearn.ensemble import RandomForestClassifier
       from sklearn.feature_selection import mutual_info_classif, f_classif
       from sklearn.metrics import roc_auc_score, roc_curve, classification_report
       from sklearn.model_selection import train_test_split, GridSearchCV
       
       model = RandomForestClassifier(n_estimators=150) #number of decision trees, large data sets more trees
       
       # Define the hyperparameter grid to search
       param_grid = {
           'n_estimators': [50, 100, 150],  # Adjust the values as needed
           'max_depth': [None, 10, 20, 30],  # Adjust the values as needed, to limit the depth of each tree and avoid overfitting
           'min_samples_split': [2, 5, 10],  # Adjust the values as needed,sets the threshold for deciding when to make a further division in the tree structure
           'min_samples_leaf': [1, 2, 4]  # Adjust the values as needed,threshold for determining when a node should be a terminal leaf rather than splitting it further
       }
       
       # Create GridSearchCV
       grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy',verbose=2)
       #cv=5 data will be split into 5 folds, large datasets small, small datasets big 5 or 10
       #n_jobs=-1 to use all CPU
       #verbose2=detailed output
       
       # Fit the model to the training data
       grid_search.fit(X, y) #x features, y target, default
       
       # Get the best hyperparameters and model
       best_params = grid_search.best_params_ #stores a dictionary containing the best hyperparameter values found by the grid search
       best_model = grid_search.best_estimator_ #stores the best machine learning model with the optimal hyperparameters
       
       st.write("Model training complete!")       


# Function to make predictions
def make_prediction(input_data):
    # Add your data preprocessing steps if needed
    # Example: input_data = preprocess_data(input_data)
    X_test = input_data[features]

    # Make predictions using the loaded model
    prediction = best_model.predict_proba(X_test)[:, 1]
    #prediction = [0.10665705, 0.09051057, 0.89876176]
    return prediction

# Streamlit app
def main():
       st.title("Prediction App")
       train_model()
 
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
