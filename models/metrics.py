import itertools
import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_curve,
)


class StreamSegMetrics_classification:
    def __init__(self, n_classes):
        self.metric_dict = {"kappa_score": -1, "acc": -1, "f1": -1}
        self.pred_list = []
        self.target_list = []
        self.n_classes = n_classes

    def update(self, pred, target):
        self.pred_list.append(pred.transpose())
        self.target_list.append(target.transpose())

    def get_results(self):
        pred_np_flatten = np.array(
            [item for item in itertools.chain.from_iterable(self.pred_list)]
        )
        target_np_flatten = np.array(
            [item for item in itertools.chain.from_iterable(self.target_list)]
        )

        self.metric_dict["kappa_score"] = cohen_kappa_score(
            pred_np_flatten, pred_np_flatten
        )
        self.metric_dict["acc"] = accuracy_score(target_np_flatten, pred_np_flatten)
        self.metric_dict["f1"] = f1_score(
            target_np_flatten, pred_np_flatten, average="macro"
        )

        return self.metric_dict

    def get_pred_flatten(self):
        return np.array(
            [item for item in itertools.chain.from_iterable(self.pred_list)]
        )

    def reset(self):
        self.pred_list = []
        self.target_list = []


def metrics_np(np_res, np_gnd, b_auc=False):
    f1m = []
    accm = []
    aucm = []
    specificitym = []
    precisionm = []
    sensitivitym = []
    ioum = []
    mccm = []

    epsilon = 2.22045e-16
    for i in range(np_res.shape[0]):
        label = np_gnd[i, ...]
        pred = np_res[i, ...]
        label = label.flatten()
        pred = pred.flatten()
        # assert label.max() == 1 and (pred).max() <= 1
        # assert label.min() == 0 and (pred).min() >= 0

        y_pred = np.zeros_like(pred)
        y_pred[pred > 0.5] = 1

        try:
            tn, fp, fn, tp = confusion_matrix(
                y_true=label, y_pred=pred
            ).ravel()  # for binary
        except ValueError:
            tn, fp, fn, tp = 0, 0, 0, 0
        accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
        specificity = tn / (tn + fp + epsilon)  #
        sensitivity = tp / (tp + fn + epsilon)  # Recall
        precision = tp / (tp + fp + epsilon)
        f1_score = (2 * sensitivity * precision) / (sensitivity + precision + epsilon)
        iou = tp + (tp + fp + fn + epsilon)

        tp_tmp, tn_tmp, fp_tmp, fn_tmp = (
            tp / 1000,
            tn / 1000,
            fp / 1000,
            fn / 1000,
        )  # to prevent overflowing
        mcc = (tp_tmp * tn_tmp - fp_tmp * fn_tmp) / math.sqrt(
            (tp_tmp + fp_tmp)
            * (tp_tmp + fn_tmp)
            * (tn_tmp + fp_tmp)
            * (tn_tmp + fn_tmp)
            + epsilon
        )  # Matthews correlation coefficient

        f1m.append(f1_score)
        accm.append(accuracy)
        specificitym.append(specificity)
        precisionm.append(precision)
        sensitivitym.append(sensitivity)
        ioum.append(iou)
        mccm.append(mcc)
        if b_auc:
            fpr, tpr, thresholds = roc_curve(sorted(y_pred), sorted(label))
            AUC = auc(fpr, tpr)
            aucm.append(AUC)

    output = dict()
    output["f1"] = np.array(f1m).mean()
    output["acc"] = np.array(accm).mean()
    output["spe"] = np.array(specificitym).mean()
    output["sen"] = np.array(sensitivitym).mean()
    output["iou"] = np.array(ioum).mean()
    output["pre"] = np.array(precisionm).mean()
    output["mcc"] = np.array(mccm).mean()

    if b_auc:
        output["auc"] = np.array(aucm).mean()

    return output
