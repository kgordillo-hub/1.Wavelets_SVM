import time

import flask
import json
import base64
import pandas as pd

from io import StringIO
from flask import Response, request
from Algorithm.SVM_Wavelet import train_model, make_prediction
from threading import Thread

application = flask.Flask(__name__)

trained = False

@application.route("/waveletsSVM/trainModel", methods=["POST"])
def train():
    if request.json is None:
        # Expect application/json request
        response = Response("Empty request", status=415)
    else:
        try:
            request_content = json.loads(request.data)
            message = request_content

            print("Training - JSON content ", message)

            data = message['data']
            csv_decoded = base64.b64decode(data).decode('utf-8')
            df = pd.read_csv(StringIO(csv_decoded, newline=''), delimiter=',')

            closing_prices = df['closing_price'].values.tolist()
            dates = df['dates'].values.tolist()

            print("Closing prices: ", closing_prices)
            print("Dates: ", dates)
            y_pred, y_test = train_model(closing_prices=closing_prices, dates=dates)
            global trained
            trained = True
            service_response = {'Predicted_values': y_pred.tolist(), 'Real_values': y_test.tolist()}
            #response = Response("Trained", 200)
            response = Response(json.dumps(service_response, default=str).encode('UTF-8'), 200)
        except Exception as ex:
            print(ex)
            response = Response("Error processing", 500)

    return response


@application.route("/waveletsSVM/predict", methods=["POST"])
def predict():
    global trained
    if request.json is None:
        # Expect application/json request
        response = Response("Empty request", status=415)
    else:
        try:
            if trained:
                request_content = json.loads(request.data)
                message = request_content
                days_to_predict = message["days_to_predict"]
                print("Predict - JSON content ", message)

                y_pred, _dates = make_prediction(prediction_days=days_to_predict)
                #y_pred = sc_y.inverse_transform(y_pred.reshape(1, -1)).ravel()
                print("Prediction: ", y_pred)
                print("Dates: ", _dates)
                service_response = {'Predicted_values': y_pred.tolist(), 'Dates': _dates.tolist()}
                response = Response(json.dumps(service_response, default=str).encode('UTF-8'), 200)
            else:
                response = Response("Call the training model method first", 405)
        except Exception as ex:
            print(ex)
            response = Response("Error processing", 500)

    return response


if __name__ == "__main__":
    application.run(host="0.0.0.0", threaded=True)
