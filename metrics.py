import numpy as np
from sklearn.metrics import roc_curve, auc

CRITERIA = ('loss', 'micro', 'macro')


def compute_roc(labels, preds):
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    return auc(fpr, tpr)


def compute_term_auc(labels, preds):
    n_prots = labels.shape[0]
    aucs = []
    for i in range(labels.shape[1]):
        pos_n = labels[:, i].sum()
        if 0 < pos_n < n_prots:
            fpr, tpr, _ = roc_curve(labels[:, i], preds[:, i])
            aucs.append(auc(fpr, tpr))
    return float(np.mean(aucs)) if aucs else 0.0
