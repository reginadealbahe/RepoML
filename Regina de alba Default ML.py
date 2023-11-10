#!/usr/bin/env python
# coding: utf-8

# In[48]:


#PART A DATA ENGINEERING
import pandas as pd


# In[49]:


df_series = pd.read_csv(r'/Users/renataherrera/Desktop/Python individual /train_series.csv')


# In[50]:


print(df_series)


# In[51]:


# Pivot the time-dependent columns (PAY, BILL_AMT, PAY_AMT) to build columns for each month
df_pivoted = df_series.pivot(index='ID', columns='MONTH', values=['PAY', 'BILL_AMT', 'PAY_AMT'])


# In[52]:


print(df_pivoted)


# In[53]:


df_pivoted.columns = [f'{col}_{month}' for col, month in df_pivoted.columns]


# In[54]:


print(df_pivoted)


# In[55]:


# Reset the index to get 'ID' as a separate column
df_pivoted.reset_index(inplace=True)


# In[56]:


print(df_pivoted)


# In[57]:


# Load the "train_customers.csv" file into a DataFrame
df_customers = pd.read_csv(r'/Users/renataherrera/Desktop/Python individual /train_customers.csv')


# In[58]:


# Merge the two DataFrames on the 'ID' column
df_customers = pd.merge(df_customers, df_pivoted, on='ID', how='inner')


# In[59]:


print (df_customers)


# In[60]:


# Load the "train_target.csv" file into a DataFrame
df_target = pd.read_csv(r'/Users/renataherrera/Desktop/Python individual /train_target.csv')


# In[61]:


df_customersmerge = pd.merge(df_customers, df_target, on='ID', how='inner')


# In[62]:


print(df_customersmerge)


# In[63]:


# Define the file path where you want to save the CSV file
output_file = r'/Users/renataherrera/Desktop/DeAlbaRegina_A_train.csv'


# In[64]:


# Use the to_csv method to export the DataFrame to a CSV file
df_customersmerge.to_csv(output_file,index=False)


# In[65]:


#PART B EXPLORATORY DATA ANALYSIS 
import matplotlib.pyplot as plt
import seaborn as sns


# In[66]:


defaulters_count=df_customersmerge.groupby("EDUCATION")["DEFAULT_JULY"].sum()

defaulters_count


# In[67]:


defaulters_count.plot(kind='bar', color='royalblue')
plt.xlabel('Education Level')
plt.ylabel('Defaulters')
plt.title('Distribution of Defaulters given the Education')
plt.xticks(ticks=[0,1,2,3,4,5], labels=['Graduate School', 'University', 'High School', 'Others', 'Unknown1', 'Unknown2'], rotation=50)
plt.show()


# We can observe in this bar chart that individuals with highschool educational level are more likely to default. This might be because income of people with a high school education may, on average, earn lower incomes compared to those with higher levels of education, also they may have accumulated student loans or other debts related to education. Important to consider that without a college education, individuals may have fewer opportunities to learn about financial management, budgeting, and responsible credit use. Since the difference of defaults regarding the education level is big, the bank should give this a lot of consideration before giving a loan. 

# In[ ]:





# In[ ]:





# In[68]:


import pandas as pd
import matplotlib.pyplot as plt

#DataFrame 
data_df = df_customersmerge

plt.figure(figsize=(12, 6))
limit_bal_default = data_df.loc[data_df['DEFAULT_JULY'] == 1]['LIMIT_BAL']
limit_bal_nondefault = data_df.loc[data_df['DEFAULT_JULY'] == 0]['LIMIT_BAL']

# Get a color 
box_color = 'blue'  

box = plt.boxplot([limit_bal_default, limit_bal_nondefault], notch=True, patch_artist=True,
                   boxprops=dict(facecolor=box_color), whiskerprops=dict(color="green", linewidth=2))

plt.title('Distribution LIMIT_BAL for Defaulters and Non-Defaulters', fontsize=14) 
plt.xticks([1, 2], ['Defaulters', 'Non-Defaulters'])
plt.xlabel('Default Status')
plt.ylabel('LIMIT_BAL')

plt.show()


# In this box plot we can observe that non defaulters have higher credit limit, which is good for the bank in order to have lower risks, but still there are many defaulters with high limit so the bank should lower this in order to avoid losing money. 

# In[69]:


#3 more insights
import pandas as pd
import matplotlib.pyplot as plt

#DataFrame 
data_df = df_customersmerge

# Filter data for defaulters and non-defaulters
defaulters_age = data_df.loc[data_df['DEFAULT_JULY'] == 1]['AGE']
nondefaulters_age = data_df.loc[data_df['DEFAULT_JULY'] == 0]['AGE']

# Create a histogram for defaulters
plt.hist(defaulters_age, bins=10, alpha=0.5, label='Defaulters', color='red')

# Create a histogram for non-defaulters
plt.hist(nondefaulters_age, bins=10, alpha=0.5, label='Non-Defaulters', color='blue')

# Add labels and a legend
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Defaulters by Age')
plt.legend()

# Show the plot
plt.show()




# In this histogram we can observe that people between the ages 20´s-30´s are more likely to default being 26-27 the peak of default. We can infer this happens because people in this age group have a limited credit history, not very high income level, many life changes and also many people of that age are still finishing paying off their education loans. 
# Having this in mind, as a bank we should get a broader range of information for people in this age group, like income, employment history, credit history, and debt-to-income ratio before giving the loans. 
# Also, we can observe that people older than 50 are not very likely to ask for loans,this can be because of the financial stability, retirement and reduced expenses at that age. 

# In[70]:


defaulters_count=df_customersmerge.groupby("SEX")["DEFAULT_JULY"].sum()

defaulters_count


# In[71]:


defaulters_count.plot(kind='bar', color='royalblue')
plt.xlabel('SEX')
plt.ylabel('Defaulters')
plt.title('Distribution of Defaulters given Sex')
plt.xticks(ticks=[0,1], labels=['MALE', 'FEMALE'], rotation=50)
plt.show()


#    In this bar chart we can observe that females are more likely to default than men, this is probably because of the disparities on income between men and women, women are more likely to be in less stable jobs and another important factor to consider is that women tend to have more family responsabilities which doesn´t let them work as much time as men. It's important to emphasize that these factors are complex, and individual circumstances can vary widely. Not all females face a higher risk of default, and the observed trend may be more pronounced in a specific region.

# In[72]:


defaulters_count=df_customersmerge.groupby("MARRIAGE")["DEFAULT_JULY"].sum()

defaulters_count


# In[73]:


defaulters_count.plot(kind='bar', color='royalblue')
plt.xlabel('MARRIAGE')
plt.ylabel('Defaulters')
plt.title('Distribution of Defaulters given Marriage')
plt.xticks(ticks=[1,2,3], labels=['Married', 'Single', 'Other'], rotation=50)
plt.show()


# In this chart we can observe that single people are more likely to default, but the difference is not very big. This might be because single people are less responsible than married people, also married people have double income which makes it easier to pay off bills, also singles may experience significant life events, such as job loss, health issues, or unexpected expenses, without the financial support and sharing of responsibilities that marriage can provide.

# 

# In[74]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df_1 = pd.read_csv('DeAlbaRegina_A_train.csv')
df_2 = pd.read_csv('test_data.csv')
# Concatenate the two DataFrames vertically
data_train = pd.concat([df_1, df_2], ignore_index=True)
data_train.columns


# In[75]:


y = data_train.DEFAULT_JULY #target variable
features = ['EDUCATION', 'AGE','LIMIT_BAL','PAY_JUNE',
       'PAY_MAY', 'PAY_APRIL', 'PAY_MARCH', 'PAY_FEBRUARY', 'PAY_JANUARY','SEX','MARRIAGE','PAY_AMT_JUNE',
       'PAY_AMT_MAY', 'PAY_AMT_APRIL', 'PAY_AMT_MARCH', 'PAY_AMT_FEBRUARY', 'PAY_AMT_JANUARY','BILL_AMT_APRIL', 'BILL_AMT_FEBRUARY', 'BILL_AMT_JANUARY',
       'BILL_AMT_JUNE', 'BILL_AMT_MARCH', 'BILL_AMT_MAY']
X= data_train[features]
X.describe()


# In[76]:


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

# Make predictions using the best model
y_pred = best_model.predict(X)

# Evaluate the model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y, y_pred) #accuracy comparing y target with y predicted target

print("Best Hyperparameters:", best_params)
print("Accuracy on Test Data:", accuracy)


# In[77]:


print("Best AUC:", {roc_auc_score(y, best_model.predict_proba(X)[:,1])}) #for binary classification


# In[78]:


submission_data = pd.read_csv('submission_features.csv')
submission_data.head()


# In[79]:


X_test = submission_data[features]
X_test.describe() #to make the model make predictions in the submissions data set


# In[80]:


# Get probability estimates 
probability = best_model.predict_proba(X_test)[:, 1]
print(probability)


# In[81]:


# Create a DataFrame with 'ID' and 'Prediction' columns
result_df = pd.DataFrame({'ID': submission_data['ID'], 'DEFAULT_JULY': probability})
result_df.head()


# In[82]:


result_df.to_csv('predictions.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:




