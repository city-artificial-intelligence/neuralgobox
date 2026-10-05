#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import os

from collections import Counter
from aminoacids import MAXLEN, to_ngrams
import logging

from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from utils import get_goplus_defs, MOLECULAR_FUNCTION, CELLULAR_COMPONENT, BIOLOGICAL_PROCESS

logging.basicConfig(level=logging.INFO)

eval_terms = {
    'mf': set(['GO:0001227', 'GO:0001228', 'GO:0003735', 'GO:0004867', 'GO:0005096']),
    'bp': set(['GO:0000381', 'GO:0032729', 'GO:0032755', 'GO:0032760', 'GO:0046330', 'GO:0051897', 'GO:0120162']),
    'cc': set(['GO:0005762', 'GO:0022625', 'GO:0042788', 'GO:1904813'])}

@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Prediction model')
@ck.option(
    '--ont', '-ont', default='mf',
    help='Prediction model')
@ck.option(
    '--model', '-m', default='deepgozero_blast',
    help='Prediction model')
@ck.option(
    '--combine', '-c', is_flag=True,
    help='Prediction model')
@ck.option(
    '--diamond-only', is_flag=True,
    help='Evaluate the DiamondScore predictions alone')
def main(data_root, ont, model, combine, diamond_only):
    # Load interpro data
    test_data_file = f'{data_root}/{ont}/predictions_{model}.pkl'
    terms_file = f'{data_root}/{ont}/terms.pkl'
    df = pd.read_pickle(test_data_file)

    terms_df = pd.read_pickle(terms_file)
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    preds = np.empty((len(df), len(terms)), dtype=np.float32)
    labels = np.zeros((len(df), len(terms)), dtype=np.float32)

    alpha = 0.5
    annots = Counter()
    train_df = pd.read_pickle(f'{data_root}/{ont}/train_data.pkl')
    for i, row in enumerate(train_df.itertuples()):
        annots.update(row.prop_annotations)
    valid_df = pd.read_pickle(f'{data_root}/{ont}/valid_data.pkl')
    for i, row in enumerate(valid_df.itertuples()):
        annots.update(row.prop_annotations)
        
    for i, row in enumerate(df.itertuples()):
        if combine:
            preds[i, :] = row.blast_preds * alpha + row.preds * (1 - alpha)
        elif diamond_only:
            preds[i, :] = row.blast_preds
        else:
            preds[i, :] = row.preds
        annots.update(row.prop_annotations)
        for go_id in row.prop_annotations:
            if go_id in terms_dict:
                labels[i, terms_dict[go_id]] = 1

    if combine:
        model += '_diam'
    if diamond_only:
        model += '_diamondscore'

    total_n = 0
    total_sum = 0
    aucs = []
    anns = []
    gos = []
    posns = []
    n_prots = len(df)
    for go_id, i in terms_dict.items():
        pos_n = np.sum(labels[:, i])
        if pos_n > 0 and pos_n < n_prots:
            total_n += 1
            roc_auc, fpr, tpr = compute_roc(labels[:, i], preds[:, i])
            if go_id in eval_terms[ont]:
                roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
                roc_df.to_pickle(f'{data_root}/{ont}/zero_{go_id}_{model}_auc_train.pkl')
                print(go_id, roc_auc)
            total_sum += roc_auc
            aucs.append(roc_auc)
            anns.append(annots[go_id])
            gos.append(go_id)
            posns.append(int(pos_n))
    out_df = pd.DataFrame({'terms': gos, 'aucs': aucs,
                           'annots': anns, 'test_pos': posns})
    out_df.to_pickle(f'{data_root}/{ont}/{model}_auc_annots.pkl')
    print(f'Average AUC for {ont} {total_sum / total_n:.3f}  (n_terms={total_n})')

        
def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc, fpr, tpr

if __name__ == '__main__':
    main()
