import flask
import json
from flask import Response, request
from Algorithm.SVM_Wavelet import train_model, make_prediction

application = flask.Flask(__name__)

sc_y = None
svr = None


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

            closing_prices = message["closing_price"]
            dates = message["dates"]

            print("Closing prices: ", closing_prices)
            print("Dates: ", dates)

            global sc_y, svr
            svr, sc_y = train_model(closing_prices=closing_prices, dates=dates)
            response = Response("Trained", 200)
        except Exception as ex:
            print(ex)
            response = Response("Error processing", 500)

    return response


@application.route("/waveletsSVM/predict", methods=["POST"])
def predict():
    global svr, sc_y
    if request.json is None:
        # Expect application/json request
        response = Response("Empty request", status=415)
    else:
        try:
            if svr:
                request_content = json.loads(request.data)
                message = request_content
                days_to_predict = message["days_to_predict"]
                print("Predict - JSON content ", message)

                y_pred, _dates = make_prediction(svr, prediction_days=days_to_predict)
                y_pred = sc_y.inverse_transform(y_pred.reshape(1, -1)).ravel()
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
    application.run(host="0.0.0.0")
