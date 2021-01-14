from flask import app, Flask, request
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

df = pd.read_csv('adult_data_usa.csv')
X = df.drop(columns='income')
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

categorical_features = ['education', 'workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex']
# for feature in categorical_features:

for feature in categorical_features:
    X_train[feature] = X_train[feature].apply(lambda x: x.strip().rstrip())
    X_test[feature] = X_test[feature].apply(lambda x: x.strip().rstrip())
    print(X_train[feature].unique())

enc = OrdinalEncoder()
enc.fit(X_train[categorical_features])
X_train[categorical_features] = enc.transform(X_train[categorical_features])
X_test[categorical_features] = enc.transform(X_test[categorical_features])


# Train the model
dt = RandomForestClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

# Save model and testFile
X_test.to_csv("testFile.csv")
pickle.dump(dt, open("model_dt.p", "wb"))
pickle.dump(enc, open("label_encoder.p", "wb"))