import pickle
from flask import Flask, request

from flasgger import Swagger

import numpy as np
import pandas as pd

with open('model.pkl' , 'rb') as model_pkl:
    model_lgbm=pickle.load(model_pkl)

api=Flask(__name__)
swagger = Swagger(api)

@api.route('/predict_lgbm',methods=["GET"])
def predict_lgbm():
    """ test documentation API"""
    sepal_length = request.args.get("sepal_length")
    sepal_width = request.args.get("sepal_width")
    peta_length = request.args.get("peta_length")
    petal_width = request.args.get("petal_width")

    input_data = np.array([[sepal_length,sepal_width, peta_length,petal_width]])
    prediction = model_lgbm.predict(input_data)
    return src(prediction)

@api.route('/predict_lgbm_file',methods=["POST"])
def predict_lgbm_file():
    input_data=pd.read_csv(requests.files.get("input_file"))
    prediction = model_lgbm.predict(input_data)
    return src(list(prediction))

if __name__=='__main__':
    api.run(host='0.0.0.0',port=8888)