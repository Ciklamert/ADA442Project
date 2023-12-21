#!/usr/bin/env python
# coding: utf-8

# In[1]:

#Importing dependencies
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import KNNImputer
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# In[2]:

#Column scheme
column_order = ["age", "job", "marital", "education", "default", "housing", "loan", 
                "contact", "month", "day_of_week", "duration", "campaign", "pdays", 
                "previous", "poutcome", "emp.var.rate", "cons.price.idx", 
                "cons.conf.idx", "euribor3m", "nr.employed", "y"]

# Read the CSV file with the specified column order
bank_additional = pd.read_csv("bank-additional.csv", delimiter=';', names=column_order)


# In[3]:


bank_additional.shape


# In[4]:


bank_additional.head()


# In[5]:

#Looking whether the data is balancer or not.
y = list(bank_additional['y'])
instances_per_class = {'yes' : y.count('yes'), 'no' : y.count('no')}

plt.bar([0, 1], instances_per_class.values(), color='green')
plt.xlabel('Class')
plt.ylabel('No. of Instances')
plt.title('Class Imbalance Check')
plt.xticks([0, 1], ['Subscribed', 'Not Subscribed'])
plt.show()

print(f"Class Labels : {list(instances_per_class.keys())}\nNo. of Inst. : {list(instances_per_class.values())}\n\nTotal number of features : {len(bank_additional.columns)-1}")


# In[6]:


bank_additional.dropna(inplace = True) #drop null values


# In[7]:


bank_additional.drop_duplicates(inplace = True) # drop duplicates


# In[8]:
#Mapping categorical variables

bank_additional["marital"] = bank_additional["marital"].replace({"married":1,"single":2,"divorced":3})
bank_additional["default"] = bank_additional["default"].replace({"no":1,"yes":2})
bank_additional["housing"] = bank_additional["housing"].replace({"no":1,"yes":2})
bank_additional["loan"] = bank_additional["loan"].replace({"no":1,"yes":2})
bank_additional["contact"] = bank_additional["contact"].replace({"cellular":1,"telephone":2})
bank_additional["month"] = bank_additional["month"].replace({"may":1,"jun":2,"nov":3,"sep":4,"jul":5,"aug":6,"mar":7,"oct":8,"apr":9,"dec":10})
bank_additional["education"] = bank_additional["education"].replace({"basic.9y":1,"high.school":2,"university.degree":3,"professional.course":4,"basic.6y":5,"basic.4y":6,"illiterate":7})
bank_additional["y"]= bank_additional["y"].replace({"no":1,"yes":2})
bank_additional["day_of_week"] = bank_additional["day_of_week"].replace({"fri":5,"wed":3,"mon":1,"thu":4,"tue":2})
bank_additional["poutcome"] = bank_additional["poutcome"].replace({"nonexistent":1,"failure":2,"success":3})
bank_additional["job"] = bank_additional["job"].replace({"blue-collar":0,"services":1,"admin.":2,"entrepreneur":3,"self-employed":4,"technician":5,"management":6,"student":7,"retired":8,"housemaid":9,"unemployed":10})


# In[9]:

#making unknown null.
bank_additional.replace("unknown", np.nan, inplace=True)


# In[10]:


for column in bank_additional.columns:
    bank_additional[column] = pd.to_numeric(bank_additional[column], errors='coerce')
    default_value = 0
    bank_additional[column].fillna(default_value, inplace=True)
    bank_additional[column] = bank_additional[column].astype(int)


# In[11]:


# Balancing the data
negatives = bank_additional[bank_additional["y"] == 1.0]
positives = bank_additional[bank_additional["y"] == 2.0]
    # balance data
if len(positives) < len(negatives):
    negatives = negatives.sample(n=len(positives))
elif len(negatives) < len(positives):
    positives = positives.sample(n=len(negatives))
bank_additional = pd.concat([negatives, positives])


# In[19]:

#See balanced data
y = list(bank_additional['y'])
instances_per_class = {'yes' : y.count(2.0), 'no' : y.count(1.0)}
plt.bar([0, 1], instances_per_class.values(), color='green')
plt.xlabel('Class')
plt.ylabel('No. of Instances')
plt.title('Class Imbalance Check')
plt.xticks([0, 1], ['Subscribed', 'Not Subscribed'])
plt.show()

print(f"Class Labels : {list(instances_per_class.keys())}\nNo. of Inst. : {list(instances_per_class.values())}\n\nTotal number of features : {len(bank_additional.columns)-1}")

"""
imputer = KNNImputer(n_neighbors=5)
bank_additional = pd.DataFrame(imputer.fit_transform(bank_additional), columns=bank_additional.columns) 
"""
# In[20]:
#Feature selection
bank_additional.drop(['age','cons.price.idx','nr.employed'],axis = 1,inplace = True)
X = bank_additional.drop('y', axis=1)
y = bank_additional['y']

# splitting data into train and test, 0.2 test size, 0.8 train size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline for numeric features
numeric_features = [ 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.conf.idx', 'euribor3m']
numeric_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)), #Imputing missing values with KNN imputer
    ('scaler', MinMaxScaler())]) #min max scaler

categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
categorical_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5))])  #Imputing missing values with KNN imputer
""""
scaler = MinMaxScaler()
numerical_features = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'] 
bank_additional[numerical_features] = scaler.fit_transform(bank_additional[numerical_features])
"""

# Creating a variable for pipeline steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])



# In[14]:


bank_additional.head()



# In[16]:


bank_additional.info()
bank_additional.head()


# Evaluating all three of the models, hyperparameter tuning them
# In[23]:
models = {
    'Logistic Regression': (LogisticRegression(max_iter=500), {
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__penalty': ['l2']}),
    'Random Forest': (RandomForestClassifier(), {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [2, 4, 6],
        'classifier__min_samples_split': [2, 5, 10]}),
    'Gradient Boosting': (GradientBoostingClassifier(), {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.1, 0.05, 0.01],
        'classifier__max_depth': [3, 4, 5]})
}

# Train and evaluate models
results = {}
for model_name, (model, params) in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    grid_search = GridSearchCV(pipeline, param_grid=params, cv=3) # Grid search
    grid_search.fit(X_train, y_train) # model training
    best_model = grid_search.best_estimator_ # best result of grid search
    predictions = best_model.predict(X_test) # making predictions with the best model
    accuracy = accuracy_score(y_test, predictions) # finding the accuracy
    results[model_name] = {'best_model': best_model, 'accuracy': accuracy}
    print(f"{results[model_name]}")

# Determine best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
print(best_model_name)
best_model = results[best_model_name]['best_model']
best_accuracy = results[best_model_name]['accuracy']

# In[27]:

"""
logreg = LogisticRegression(max_iter=500)
logreg.fit(X_train, y_train)
logreg_pred = logreg.predict(X_test)
logreg_accuracy = accuracy_score(y_test, logreg_pred)
logreg_precision = precision_score(y_test, logreg_pred)
print("Logistic regression accuracy: " + str(logreg_accuracy))
print("Logistic regression precision: " + str(logreg_precision))


# In[43]:


rf = RandomForestClassifier() #bunu seçtik
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
print("Random forest accuracy: " +  str(rf_accuracy))
print("Random forest precision: " + str(rf_precision))


# In[35]:


gradient_boosting = GradientBoostingClassifier()
gradient_boosting.fit(X_train,y_train)
gradient_boosting_pred = gradient_boosting.predict(X_test)

gradient_boosting_accuracy = accuracy_score(y_test,gradient_boosting_pred)
gradient_boosting_precision = precision_score(y_test,gradient_boosting_pred)
print("Gradient boosting accuracy: " + str(gradient_boosting_accuracy))
print("Gradient boosting precision: " + str(gradient_boosting_precision))


# In[38]:


# In[44]:


#HYPERPARAMETER TUNING
rf_params =  {
    'n_estimators': [100, 200, 300],
    'max_depth': [2, 4, 6],
    'min_samples_split': [2, 5, 10]
}

rf_grid = GridSearchCV(rf, rf_params, cv=5,scoring="accuracy")


# In[45]:


rf_grid.fit(X_train, y_train)
best_rf_model = rf_grid.best_estimator_
rf_best_pred = best_rf_model.predict(X_test)
rf_best_accuracy = accuracy_score(y_test, rf_best_pred)
rf_best_precision = precision_score(y_test, rf_best_pred)
print("Final evolution accuracy: " + str(rf_best_accuracy))
print("Final evolution precision: " + str(rf_best_precision))

print(X_train.shape)
print(X_test.shape)

"""
import joblib

# Save the model to a file
joblib.dump(best_model, 'trained_model.pkl')

