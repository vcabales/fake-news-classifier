from flask import Flask, request, render_template, jsonify
from sklearn.externals import joblib
import pandas as pd
import numpy as np

app = Flask(__name__, template_folder='./templates')

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        result = request.form
    #query_df = pd.DataFrame(result)
    return jsonify(result)

@app.route('/report')
def report():
	return render_template('report.html')

#TODO: Need to make a pipeline that would take the dataset, put it into a vectorizer, and feed that into MultinnomialNB

    # try:
    #     json_ = result
    #     query_df = pd.DataFrame(json_)
    #     query = pd.get_dummies(query_df)
    #     for col in model_columns:
    #         print ("col")
    #         if col not in query.columns:
    #             query[col] = 0
    # except ValueError:
    #     return jsonify("ERROR")
    #
    # prediction = clf.predict(query)
    # return jsonify({'prediction': list(prediction)})

if __name__ == '__main__':
    clf = joblib.load('model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    app.run(port=3000)
