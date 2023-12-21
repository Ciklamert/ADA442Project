import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector as selector
import streamlit as st
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.impute import KNNImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, chi2
import joblib
def preprocess_input(data):
    data["job"] = data["job"].replace(
        {"blue-collar": 0, "services": 1, "admin.": 2, "entrepreneur": 3, "self-employed": 4, "technician": 5,
         "management": 6, "student": 7, "retired": 8, "housemaid": 9, "unemployed": 10})
    data["marital"] = data["marital"].replace({"married": 1, "single": 2, "divorced": 3, "unknown": 0})
    data["education"] = data["education"].replace(
        {"basic.9y": 1, "high.school": 2, "university.degree": 3, "professional.course": 4, "basic.6y": 5,
         "basic.4y": 6, "illiterate": 7, "unknown": 0})
    data["default"] = data["default"].replace({"no": 1, "yes": 2, "unknown": 0})
    data["housing"] = data["housing"].replace({"no": 1, "yes": 2, "unknown": 0})
    data["loan"] = data["loan"].replace({"no": 1, "yes": 2, "unknown": 0})
    data["contact"] = data["contact"].replace({"cellular": 1, "telephone": 2, "unknown": 0})
    data["month"] = data["month"].replace(
        {"may": 1, "jun": 2, "nov": 3, "sep": 4, "jul": 5, "aug": 6, "mar": 7, "oct": 8, "apr": 9, "dec": 10})
    data["day_of_week"] = data["day_of_week"].replace({"fri": 1, "wed": 2, "mon": 3, "thu": 4, "tue": 5})
    data["poutcome"] = data["poutcome"].replace({"nonexistent": 1, "failure": 2, "success": 3})
    #input_data["y"] = input_data["y"].replace({"no": 0, "yes": 1})
    # Replace "unknown" values with NaN
    data.replace("unknown", np.nan, inplace=True)

    # Convert columns to numeric
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')
        default_value = 0
        data[column].fillna(default_value, inplace=True)
        data[column] = data[column].astype(int)

    # Impute missing values using KNNImputer
    imputer = KNNImputer(n_neighbors=5)
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Scale numerical features using MinMaxScaler
    scaler = MinMaxScaler()
    numerical_features = ['duration', 'campaign', 'pdays', 'previous', 'emp.var.rate',
                          'cons.conf.idx', 'euribor3m']
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    return data


# In[42]:


def make_prediction(model, input_data):
    # Use the trained model to make predictions on the input data
    prediction = model.predict(input_data)
    return prediction


# In[43]:

if __name__ == '__main__':
    st.title("Bank Marketing Prediction App")
    job = st.selectbox("Job",["blue-collar", "services", "admin.", "entrepreneur", "self-employed", "technician",
         "management", "student", "retired", "housemaid", "unemployed"])
    marital_status = st.selectbox("Marital Status", ["married", "single", "divorced", "unknown"])
    education = st.selectbox("Education",
                             ["basic.9y", "high.school", "university.degree", "professional.course", "basic.6y",
                              "basic.4y", "illiterate", "unknown"])
    default = st.selectbox("Default", ["no", "yes", "unknown"])
    housing = st.selectbox("Housing", ["no", "yes", "unknown"])
    loan = st.selectbox("Loan", ["no", "yes", "unknown"])
    contact = st.selectbox("Contact", ["cellular", "telephone", "unknown"])
    month = st.selectbox("Month", ["may", "jun", "nov", "sep", "jul", "aug", "mar", "oct", "apr", "dec"])
    day_of_week = st.selectbox("Day of week", ["fri","wed","mon","thu","tue"])
    #age = st.slider("Age", min_value=0.0, max_value=88.0, value=25.0)
    duration = st.slider("Duration", min_value=0.0, max_value=3643.0, value=300.0)
    campaign = st.slider("Campaign", min_value=0.0, max_value=35.0, value=10.0)
    pdays = st.slider("Pdays", min_value=0.0, max_value=999.0, value=15.0)
    previous = st.slider("Previous", min_value=0.0, max_value=6.0, value=5.0)
    poutcome = st.selectbox("Poutcome", ["nonexistent", "failure", "success"])
    emp_var_rate = st.slider("Employment Variation Rate", min_value=-3.0, max_value=1.0, value=0.0)
    #cons_price_idx = st.slider("Consumer Price Index", min_value=0.0, max_value=94.0, value=35.0)
    cons_conf_idx = st.slider("Consumer Confidence Index", min_value=-50.0, max_value=0.0, value=-35.0)
    euribor3m = st.slider("Euribor 3 Month Rate", min_value=0.0, max_value=5.0, value=3.0)
    #nr_employed = st.slider("Number of Employees", min_value=0.0, max_value=5228.0, value=5000.0)

    # Create a dataframe with the user input
    input_data = pd.DataFrame({
        #'age': [age],
        'job': [job],
        'marital': [marital_status],
        'education': [education],
        'default': [default],
        'housing': [housing],
        'loan': [loan],
        'contact': [contact],
        'month': [month],
        'day_of_week': [day_of_week],
        'duration': [duration],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'poutcome': [poutcome],
        'emp.var.rate': [emp_var_rate],
        #'cons.price.idx': [cons_price_idx],
        'cons.conf.idx': [cons_conf_idx],
        'euribor3m': [euribor3m],
        #'nr.employed': [nr_employed],
    })

    # Preprocess the input data
    input_data = preprocess_input(input_data)

    # Display the preprocessed input data
    st.subheader("Preprocessed Input Data")
    st.write(input_data)

    # Load the trained model

    model = joblib.load("trained_model.pkl")

    # Make predictions
    prediction = make_prediction(model, input_data)

    # Display the prediction
    st.subheader("Prediction")
    st.write(prediction)
