import pandas as pd
import numpy as np

import pickle
from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)

def get_total_income(df):
    df['TotalIncome'] = np.log(df['ApplicantIncome'].astype('float') + df['CoapplicantIncome'].astype('float'))
    df.drop(columns=['ApplicantIncome', 'CoapplicantIncome'], inplace=True)
    return df
def log_loan_amount(df):
    df['LoanAmount'] = np.log(df['LoanAmount'].astype('float'))
    df.drop(columns=['LoanAmount'], inplace=True)
    return df

class RawFeats:
    def __init__(self, feats):
        self.feats = feats

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        return X[self.feats]

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

model = pickle.load( open( "../notebooks/model.p", "rb" ) )

class Scoring(Resource):
    def post(self):
        json_data = request.get_json()
        df = pd.DataFrame(json_data.values(), index=json_data.keys()).transpose()
        # getting predictions from our model.
        # it is much simpler because we used pipelines during development
        res = model.predict_proba(df)
        # we cannot send numpt array as a result
        return res.tolist() 

# assign endpoint
api.add_resource(Scoring, '/scoring')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5555)
