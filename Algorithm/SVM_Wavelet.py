# Standard libraries
import numpy as np
import pandas as pd

from sklearn import svm
from numpy.lib.stride_tricks import sliding_window_view
# Library for Wavelets transformations, the code was taken from:
# https://github.com/pistonly/modwtpy/blob/master/modwt.py
from Algorithm.lib.modwt import modwt, imodwt
from sklearn.preprocessing import MinMaxScaler

X_global = []
Y_global = []
dates_global = []


def train_model(closing_prices, dates, window_length=12):
    print("Training model...")
    global dates_global
    dates_global = dates
    if window_length > len(closing_prices):
        window_length = len(closing_prices)
    # closing_prices = sc_x.fit_transform(closing_prices.reshape(-1, 1))
    level = 4
    # initialize an array to store the coefficients corresponding to scale 2^4
    approx_coeff = get_coeff_from_series(closing_prices)
    # Getting approximation coefficients and last detail coefficients only
    recwt = np.zeros((np.shape(approx_coeff)[0], np.shape(approx_coeff)[1]))
    # store coefficients corresponding to scale the 4 of the wavelet transformation
    recwt[(level - 1):] = approx_coeff[-2]
    recwt[level:] = approx_coeff[-1]
    # De-noised financial series. In this case we only use the aC to reconstruct the time series
    dFs = imodwt(recwt, 'sym4')
    # training model using approx. coefficients
    X, Y = slide_window(dFs, window_length)
    sc_x = MinMaxScaler(feature_range=(0, 1))
    sc_y = MinMaxScaler(feature_range=(0, 1))

    global X_global
    X_global = sc_x.fit_transform(X)
    global Y_global
    Y_global = sc_y.fit_transform(np.reshape(Y, (-1, 1))).ravel()

    svr = train_model_approx(X_global, Y_global)

    return svr, sc_y


def make_prediction(svr, prediction_days=3, past_days=12):
    global Y_global
    global X_global

    if past_days > len(X_global):
        past_days = len(X_global)
    X_ = []
    Y_ = [np.array(Y_global)[-1]]
    X_.append(X_global[-1])
    print("Pred_days: " + str(prediction_days))
    for i in range(prediction_days):
        Y_array = np.array([Y_[-1]])
        X_array = np.array(X_[-1][-past_days + 1:])
        X_Y_concat = np.array([np.concatenate((X_array, Y_array))])
        X_ = np.concatenate((X_, X_Y_concat))
        p_value = svr.predict(X_[-1].reshape(1, -1))
        Y_ = np.concatenate((Y_, p_value))
    return Y_[:prediction_days], add_day_to_dates(prediction_days)


# Implementing slide window
def slide_window(series, window_length=2):
    _X, _Y = [], []
    # Auxiliary variable to store the sliding window combinations. We sum up +1 as we are taking the last values of
    # Aux_window as the output values of our time series
    aux_window = sliding_window_view(series, window_length + 1)
    # Taking first 'window_length' values as the input (X) and the last value (window_length+1) as the output (Y)
    for i in range(len(aux_window)):
        _Y.append(aux_window[i][-1])
        _X.append(aux_window[i][:-1])

    return _X, _Y


# Using modwt with 'sym4' wavelet and 5 levels (4 detail coefficients (dC) and 1 approximation coefficient (aC))
def apply_modwt(_data, type_s='sym4', _level=3):
    _coeff = modwt(_data, type_s, _level)
    return _coeff


def get_coeff_from_series(closing_prices):
    # calling function defined previously
    level = 4
    coeff = apply_modwt(closing_prices, type_s='sym4', _level=level)
    return coeff


def train_model_approx(X, Y):
    # Values needed to fully reconstruct the time series
    svr = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr.fit(X, Y)

    return svr


def add_day_to_dates(prediction_days):
    _dates = np.array([])
    global dates_global
    lastDate = np.array(dates_global)[-1]
    for i in range(prediction_days):
        newDate = pd.to_datetime(lastDate) + pd.DateOffset(days=i + 1)
        _dates = np.append(_dates, newDate)
    return _dates.astype('datetime64[D]')
