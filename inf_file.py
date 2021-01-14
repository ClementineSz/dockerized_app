import json

from flask import app, Flask, request
import pandas as pd
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return  'Welcome to Clem\'s API'

@app.route('/predict_one')
def predict_one():
    dt = pickle.load(open("model_dt.p", "rb"))
    enc = pickle.load(open("label_encoder.p", "rb"))
    features = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
       'marital-status', 'occupation', 'relationship', 'race', 'sex',
       'capital-gain', 'capital-loss', 'hours-per-week']
    categorical_features = ['education', 'workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex']

    param = request.args
    X_test_one = []
    for feature in features:
        X_test_one.append(param[feature])
    train_x = pd.DataFrame([X_test_one], columns=features)
    print(train_x[categorical_features])
    print(enc.categories_)
    train_x[categorical_features] = enc.transform(train_x[categorical_features])

    y_pred = dt.predict(train_x)
    return str(y_pred)


@app.route('/predict_multiple')
def predict_multiple():
    features = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week']
    categorical_features = ['education', 'workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex']


    dt = pickle.load(open("model_dt.p", "rb"))
    enc = pickle.load(open("label_encoder.p", "rb"))
    body = request.data
    test_x = []
    df = json.loads(body)
    print(df["data"])
    for x in df["data"]:
        test_x.append(list(x.values()))
    print('avant',test_x)
    test_x = pd.DataFrame(test_x, columns=features)
    print('apres',test_x)
    test_x[categorical_features] = enc.transform(test_x[categorical_features])

    y_pred = dt.predict(test_x)
    return str(y_pred)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
