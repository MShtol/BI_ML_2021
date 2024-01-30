import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    TP = np.sum((y_pred == y_true) & (y_pred == 1))
    TN = np.sum((y_pred == y_true) & (y_pred == 0))
    FP = np.sum((y_pred != y_true) & (y_pred == 1))
    FN = np.sum((y_pred != y_true) & (y_pred == 0))
    if (TP + FP) != 0:
        precision = TP / (TP + FP)
    else:
        precision = 0
    if (TP + FN) != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    if (precision + recall) != 0:
        f1 = 2*precision*recall/(precision+recall)
    else:
        f1 = np.inf
    return precision, recall, f1, accuracy


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    accuracy = sum(y_pred == y_true)/len(y_pred)
    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    mse = np.sum((y_true - y_pred) ** 2) / len(y_true)
    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    mae = np.sum(abs(y_true - y_pred)) / len(y_true)
    return mae
    