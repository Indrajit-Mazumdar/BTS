import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error


def survival_days_to_labels(y_days):
    y_labels = np.copy(y_days)
    y_labels[y_labels < 300] = 0
    y_labels[(y_labels >= 300) & (y_labels <= 450)] = 1
    y_labels[y_labels > 450] = 2
    y_labels = y_labels.astype(np.uint8)
    return y_labels


def mse_XGBoost(y_pred, dtrain):
    y_true = dtrain.get_label()
    MSE = mean_squared_error(y_true, y_pred)
    return 'MSE', MSE


def accuracy_XGBoost(y_pred, dtrain):
    y_true = dtrain.get_label()
    y_true_labels = survival_days_to_labels(y_true)
    y_pred_labels = survival_days_to_labels(y_pred)
    Accuracy = accuracy_score(y_true_labels, y_pred_labels)
    return 'Accuracy', Accuracy
