import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    confusion_matrix,
    matthews_corrcoef,
    f1_score,
)


def auc(y, z, round=True):
    if round:
        y = y.round()
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, z)


def acc(y, z, round=True):
    if round:
        y = np.round(y)
        z = np.round(z)
    return accuracy_score(y, z)


def tpr(y, z, round=True):
    if round:
        y = np.round(y)
        z = np.round(z)
    return recall_score(y, z)


def tnr(y, z, round=True):
    if round:
        y = np.round(y)
        z = np.round(z)
    c = confusion_matrix(y, z)
    return c[0, 0] / c[0].sum()


def mcc(y, z, round=True):
    if round:
        y = np.round(y)
        z = np.round(z)
    return matthews_corrcoef(y, z)


def f1(y, z, round=True):
    if round:
        y = np.round(y)
        z = np.round(z)
    return f1_score(y, z)


CLA_METRICS = [auc, acc, tpr, tnr, f1, mcc]
