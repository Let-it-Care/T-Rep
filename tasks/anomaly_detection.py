import torch
import numpy as np
import time
import torch.nn.functional as F
from functools import partial
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
import bottleneck as bn

from . import _eval_protocols as eval_protocols
from utils import upsample_minority_class
from .sepsis_ad_metrics import compute_auc, compute_accuracy_f_measure


# consider delay threshold and missing segments
def get_range_proba(predict, label, delay=7):
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_predict = np.array(predict)
    pos = 0

    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos:min(pos + delay + 1, sp)]:
                new_predict[pos: sp] = 1
            else:
                new_predict[pos: sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)

    if is_anomaly:  # anomaly in the end
        if 1 in predict[pos: min(pos + delay + 1, sp)]:
            new_predict[pos: sp] = 1
        else:
            new_predict[pos: sp] = 0

    return new_predict


# set missing = 0
def reconstruct_label(timestamp, label):
    timestamp = np.asarray(timestamp, np.int64)
    index = np.argsort(timestamp)

    timestamp_sorted = np.asarray(timestamp[index])
    interval = np.min(np.diff(timestamp_sorted))

    label = np.asarray(label, np.int64)
    label = np.asarray(label[index])

    idx = (timestamp_sorted - timestamp_sorted[0]) // interval

    new_label = np.zeros(shape=((timestamp_sorted[-1] - timestamp_sorted[0]) // interval + 1,), dtype=np.int)
    new_label[idx] = label

    return new_label


def eval_ad_result(test_pred_list, test_labels_list, test_timestamps_list, delay):
    labels = []
    pred = []
    for test_pred, test_labels, test_timestamps in zip(test_pred_list, test_labels_list, test_timestamps_list):
        assert test_pred.shape == test_labels.shape == test_timestamps.shape
        test_labels = reconstruct_label(test_timestamps, test_labels)
        test_pred = reconstruct_label(test_timestamps, test_pred)
        test_pred = get_range_proba(test_pred, test_labels, delay)
        labels.append(test_labels)
        pred.append(test_pred)
    labels = np.concatenate(labels)
    pred = np.concatenate(pred)
    return {
        'f1': f1_score(labels, pred),
        'precision': precision_score(labels, pred),
        'recall': recall_score(labels, pred)
    }


def np_shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def eval_anomaly_detection(
        model,
        all_train_data,
        all_train_labels,
        all_train_timestamps,
        all_test_data,
        all_test_labels,
        all_test_timestamps,
        delay,
        verbose=False
    ):
    t = time.time()

    def encode_data_wm(data):
        return model.encode(
                data.reshape(1, -1, 1),
                mask='mask_last',
                causal=True,
                sliding_length=1,
                sliding_padding=200,
                batch_size=256,
            ).squeeze()
    
    def encode_data_wom(data):
        return model.encode(
                data.reshape(1, -1, 1),
                causal=True,
                sliding_length=1,
                sliding_padding=200,
                batch_size=256,
            ).squeeze()
        
    wm = []
    if verbose:
        print("Encoding representations with mask...")
    for k in all_train_data:
        wm.append(encode_data_wm(np.concatenate([all_train_data[k], all_test_data[k]])))
  
    wom = []
    if verbose:
        print("Encoding representations without mask...")
    for k in all_train_data:
        wom.append(encode_data_wom(np.concatenate([all_train_data[k], all_test_data[k]])))


    
    all_train_repr = {}
    all_test_repr = {}
    all_train_repr_wom = {}
    all_test_repr_wom = {}
    for i, k in enumerate(all_train_data):
        full_repr = wm[i]
        full_repr_wom = wom[i]

        all_train_repr[k] = full_repr[:len(all_train_data[k])]
        all_test_repr[k] = full_repr[len(all_train_data[k]):]

        all_train_repr_wom[k] = full_repr_wom[:len(all_train_data[k])]
        all_test_repr_wom[k] = full_repr_wom[len(all_train_data[k]):]

    if verbose:
        print("Done")
        print("Computing anomalies")

        
    res_log = []
    labels_log = []
    timestamps_log = []
    for k in all_train_data:

        train_data = all_train_data[k]
        train_labels = all_train_labels[k]
        train_timestamps = all_train_timestamps[k]

        test_data = all_test_data[k]
        test_labels = all_test_labels[k]
        test_timestamps = all_test_timestamps[k]

        train_err = np.sum(np.abs(all_train_repr_wom[k] - all_train_repr[k]), axis=1)
        test_err = np.sum(np.abs(all_test_repr_wom[k] - all_test_repr[k]), axis=1)

        ma = np_shift(bn.move_mean(np.concatenate([train_err, test_err]), 21), 1)
        train_err_adj = (train_err - ma[:len(train_err)]) / ma[:len(train_err)]
        test_err_adj = (test_err - ma[len(train_err):]) / ma[len(train_err):]
        train_err_adj = train_err_adj[22:]

        thr = np.mean(train_err_adj) + 4 * np.std(train_err_adj)
        test_res = (test_err_adj > thr) * 1

        for i in range(len(test_res)):
            if i >= delay and test_res[i-delay:i].sum() >= 1:
                test_res[i] = 0

        res_log.append(test_res)
        labels_log.append(test_labels)
        timestamps_log.append(test_timestamps)
    
    t = time.time() - t
    
    if verbose:
        print(f"Done! Execution time: {t}")

    eval_res = eval_ad_result(res_log, labels_log, timestamps_log, delay)
    eval_res['infer_time'] = t
    return res_log, eval_res


def eval_anomaly_detection_coldstart(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay):
    t = time.time()
    
    all_data = {}
    all_repr = {}
    all_repr_wom = {}
    for k in all_train_data:
        all_data[k] = np.concatenate([all_train_data[k], all_test_data[k]])
        all_repr[k] = model.encode(
            all_data[k].reshape(1, -1, 1),
            mask='mask_last',
            causal=True,
            sliding_length=1,
            sliding_padding=200,
            batch_size=256
        ).squeeze()
        all_repr_wom[k] = model.encode(
            all_data[k].reshape(1, -1, 1),
            causal=True,
            sliding_length=1,
            sliding_padding=200,
            batch_size=256
        ).squeeze()
        
    res_log = []
    labels_log = []
    timestamps_log = []
    for k in all_data:
        data = all_data[k]
        labels = np.concatenate([all_train_labels[k], all_test_labels[k]])
        timestamps = np.concatenate([all_train_timestamps[k], all_test_timestamps[k]])
        
        err = np.abs(all_repr_wom[k] - all_repr[k]).sum(axis=1)
        ma = np_shift(bn.move_mean(err, 21), 1)
        err_adj = (err - ma) / ma
        
        MIN_WINDOW = len(data) // 10
        thr = bn.move_mean(err_adj, len(err_adj), MIN_WINDOW) + 4 * bn.move_std(err_adj, len(err_adj), MIN_WINDOW)
        res = (err_adj > thr) * 1
        
        for i in range(len(res)):
            if i >= delay and res[i-delay:i].sum() >= 1:
                res[i] = 0

        res_log.append(res[MIN_WINDOW:])
        labels_log.append(labels[MIN_WINDOW:])
        timestamps_log.append(timestamps[MIN_WINDOW:])
    t = time.time() - t
    
    eval_res = eval_ad_result(res_log, labels_log, timestamps_log, delay)
    eval_res['infer_time'] = t
    return res_log, eval_res


def eval_anomaly_detection_sepsis(
        model, 
        train_data, 
        train_labels, 
        test_data, 
        test_labels, 
        window_size,
        eval_protocol='linear',
        raw_data=False
    ):

    # Encode test data with trained model
    assert train_labels.ndim == 1
    if raw_data:
        train_repr = np.reshape(train_data, (train_data.shape[0], train_data.shape[1] * train_data.shape[2]))
        test_repr = np.reshape(test_data, (test_data.shape[0], test_data.shape[1] * test_data.shape[2]))
    else:
        train_repr = model.encode(train_data, sliding_length=6, sliding_padding=15, batch_size=64)
        test_repr = model.encode(test_data, sliding_length=6, sliding_padding=15, batch_size=64)
        train_repr = F.max_pool1d(
            torch.from_numpy(train_repr).transpose(1, 2),
            kernel_size=train_repr.shape[1],
        ).transpose(1, 2).squeeze()
        test_repr = F.max_pool1d(
            torch.from_numpy(test_repr).transpose(1, 2),
            kernel_size=test_repr.shape[1],
        ).transpose(1, 2).squeeze()


    # Fit classifier for anomaly detection
    train_repr, train_labels = upsample_minority_class(train_repr, train_labels, upsample_ratio=0.5)

    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    else:
        assert False, 'unknown evaluation protocol'

    clf = fit_clf(train_repr, train_labels)

    N_TEST_PATIENTS = 10084
    flat_test_pred = clf.predict(test_repr).astype(int)
    test_preds = np.zeros((N_TEST_PATIENTS, 45), dtype=np.int16)
    test_preds[:, -40:] = flat_test_pred.reshape(-1, 45 - window_size + 1)

    test_aucs = np.array([compute_auc(test_labels[i], test_preds[i]) for i in range(N_TEST_PATIENTS)])
    print(f"Mean AUC: {test_aucs.mean()}")

    test_accs = []
    test_f_measures = []
    for i in range(N_TEST_PATIENTS):
        acc, f_m, _, _ = compute_accuracy_f_measure(test_labels[i], test_preds[i])
        test_accs.append(acc)
        test_f_measures.append(f_m)
    test_accs = np.array(test_accs)
    test_f_measures = np.array(test_f_measures)

    print(f"Mean Accuracy: {test_accs.mean()}")
    print(f"Mean F measure: {test_f_measures.mean()}")
    
    return test_preds, {
        'acc': test_accs.mean(),
        'auprc': test_aucs.mean(),
        'f_measure': test_f_measures.mean(),
    }