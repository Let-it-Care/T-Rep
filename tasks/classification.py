import numpy as np
import torch
from . import _eval_protocols as eval_protocols
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score

def eval_classification(model, train_data, train_labels, test_data, test_labels, encoding_protocol='full_series', eval_protocol='linear'):

    if encoding_protocol == 'full_series':
        # 'full_series' encodes time series in 1 vector (no temporal dimension). This is the default and simplest setting.
        # It should be sufficient for most applications, but for maximum performance use 'timedim'.
        assert train_labels.ndim == 1 or train_labels.ndim == 2
        train_repr = model.encode(train_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
        test_repr = model.encode(test_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
    elif encoding_protocol == 'timedim':
        # 'timedim' encodes time series at a user-specified temporal granularity, resulting in higher-dimensional representations
        # to classify, but more easily separable. This can boost performance, but it more computationally expensive and requires
        # tuning hyper-parameter k.
        assert train_data.shape[1] == test_data.shape[1]
        T = train_data.shape[1]
        k = 10
        w = (T // k) if T > k else 1
        train_repr = model.encode(train_data, encoding_window=w if train_labels.ndim == 1 else None)
        test_repr = model.encode(test_data, encoding_window=w if train_labels.ndim == 1 else None)

        train_repr = train_repr.reshape(train_repr.shape[0], -1)
        test_repr = test_repr.reshape(test_repr.shape[0], -1)

    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    else:
        assert False, 'unknown evaluation protocol'

    def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])

    if train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)

    clf = fit_clf(train_repr, train_labels)

    acc = clf.score(test_repr, test_labels)
    if eval_protocol == 'linear':
        y_score = clf.predict_proba(test_repr)
    else:
        y_score = clf.decision_function(test_repr)
    test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))
    auprc = average_precision_score(test_labels_onehot, y_score)
    
    return y_score, { 'acc': acc, 'auprc': auprc }
