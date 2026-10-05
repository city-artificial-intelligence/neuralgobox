import math

import click as ck
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils import FUNC_DICT, Ontology, NAMESPACES

THRESHOLDS = 101

def load_inputs(data_root, ont, model, combine, diamond_only=False, alpha=0.5):
    go_rels = Ontology(f'{data_root}/go.obo', with_rels=True)
    terms = pd.read_pickle(f'{data_root}/{ont}/terms.pkl')['gos'].values.flatten()

    train_df = pd.concat([pd.read_pickle(f'{data_root}/{ont}/train_data.pkl'),
                          pd.read_pickle(f'{data_root}/{ont}/valid_data.pkl')])
    test_df = pd.read_pickle(f'{data_root}/{ont}/predictions_{model}.pkl')

    annotations = [set(x) for x in train_df['prop_annotations'].values]
    test_annotations = [set(x) for x in test_df['prop_annotations'].values]
    go_rels.calculate_ic(annotations + test_annotations)

    go_set = go_rels.get_namespace_terms(NAMESPACES[ont])
    go_set.remove(FUNC_DICT[ont])

    labels = [set(y for y in x if y in go_set)
              for x in test_df['prop_annotations'].values]

    if combine:
        eval_preds = np.stack([r.blast_preds * alpha + r.preds * (1 - alpha)
                               for r in test_df.itertuples()])
    elif diamond_only:
        eval_preds = np.stack([r.blast_preds for r in test_df.itertuples()])
    else:
        eval_preds = np.stack([r.preds for r in test_df.itertuples()])

    return go_rels, terms, go_set, labels, eval_preds

def build_matrices(go_rels, terms, go_set, labels, eval_preds):
    voc = [t for t in terms if t in go_set]
    label_terms = set().union(*labels) if labels else set()
    universe = sorted(set(voc) | (label_terms & go_set))
    col = {g: k for k, g in enumerate(universe)}
    n, k = len(labels), len(universe)

    scores = np.full((n, k), -np.inf, dtype=eval_preds.dtype)
    for j, go_id in enumerate(terms):
        if go_id in go_set:

            c = col[go_id]
            np.maximum(scores[:, c], eval_preds[:, j], out=scores[:, c])

    lab = np.zeros((n, k), dtype=bool)
    for i, s in enumerate(labels):
        if s:
            lab[i, [col[g] for g in s]] = True

    ic = np.array([go_rels.get_ic(g) for g in universe], dtype=np.float64)
    nic = np.array([go_rels.get_norm_ic(g) for g in universe], dtype=np.float64)
    return scores, lab, ic, nic

def sweep(scores, lab, ic, nic):
    keep = lab.any(axis=1)
    total = int(keep.sum())
    labn = lab.sum(axis=1).astype(np.float64)
    lab_nic = lab @ nic
    lab_ic = lab @ ic

    out = {k: np.zeros(THRESHOLDS) for k in
           ('f', 'p', 'r', 's', 'ru', 'mi', 'avg_ic', 'wf')}

    for t in range(THRESHOLDS):
        pred = scores >= (t / 100.0)
        tp = pred & lab

        tpn = tp.sum(axis=1).astype(np.float64)
        predn = pred.sum(axis=1).astype(np.float64)
        fpn = predn - tpn
        fnn = labn - tpn

        tpic = tp @ nic
        fpic = (pred @ nic) - tpic
        fnic = lab_nic - tpic

        tp_ic = tp @ ic
        fp_ic = (pred @ ic) - tp_ic
        fn_ic = lab_ic - tp_ic

        sel_p = keep & (predn > 0)
        p_total = int(sel_p.sum())

        denom_r, denom_wr = tpn + fnn, tpic + fnic
        if not np.all(denom_r[keep] > 0) or not np.all(denom_wr[keep] > 0):
            raise ValueError(f'zero recall denominator at t={t}; the original '
                             'would have raised ZeroDivisionError here')

        r = float((tpn[keep] / denom_r[keep]).sum()) / total
        wr = float((tpic[keep] / denom_wr[keep]).sum()) / total
        ru = float(fn_ic[keep].sum()) / total
        mi_raw = float(fp_ic[keep].sum())

        avg_ic = (float(tp_ic[keep].sum()) + mi_raw) / total
        mi = mi_raw / total

        p = wp = 0.0
        if p_total > 0:
            p = float((tpn[sel_p] / (tpn + fpn)[sel_p]).sum()) / p_total
            wp = float((tpic[sel_p] / (tpic + fpic)[sel_p]).sum()) / p_total

        f = wf = 0.0
        if p + r > 0:
            f = 2 * p * r / (p + r)
            wf = 2 * wp * wr / (wp + wr)

        for key, val in (('f', f), ('p', p), ('r', r), ('ru', ru), ('mi', mi),
                         ('avg_ic', avg_ic), ('wf', wf),
                         ('s', math.sqrt(ru * ru + mi * mi))):
            out[key][t] = val

    return out

def summarise(res):
    thr = np.arange(THRESHOLDS) / 100.0
    i_f, i_w = int(np.argmax(res['f'])), int(np.argmax(res['wf']))
    order = np.argsort(res['r'])
    return dict(
        fmax=res['f'][i_f], tmax=thr[i_f], avgic=res['avg_ic'][i_f],
        wfmax=res['wf'][i_w], wtmax=thr[i_w], smin=res['s'].min(),
        aupr=float(np.trapz(res['p'][order], res['r'][order])),
        precisions=res['p'][order], recalls=res['r'][order],
    )

@ck.command()
@ck.option('--data-root', '-dr', default='data')
@ck.option('--ont', '-ont', default='mf')
@ck.option('--model', '-m', default='deepgozero_blast')
@ck.option('--combine', '-c', is_flag=True)
@ck.option('--diamond-only', is_flag=True)
def main(data_root, ont, model, combine, diamond_only):
    go_rels, terms, go_set, labels, eval_preds = load_inputs(
        data_root, ont, model, combine, diamond_only)
    scores, lab, ic, nic = build_matrices(go_rels, terms, go_set, labels, eval_preds)

    print('Computing Fmax')
    res = summarise(sweep(scores, lab, ic, nic))

    if combine:
        model += '_diam'
    if diamond_only:
        model += '_diamondscore'
    print(model, ont)
    print(f"Fmax: {res['fmax']:0.3f}, Smin: {res['smin']:0.3f}, threshold: {res['tmax']}")
    print(f"WFmax: {res['wfmax']:0.3f}, threshold: {res['wtmax']}")
    print(f"AUPR: {res['aupr']:0.3f}")
    print(f"AVGIC: {res['avgic']:0.3f}")

    plt.figure()
    plt.plot(res['recalls'], res['precisions'], color='darkorange', lw=2,
             label=f"AUPR curve (area = {res['aupr']:0.2f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Area Under the Precision-Recall curve')
    plt.legend(loc='lower right')
    plt.savefig(f'{data_root}/{ont}/aupr_{model}.pdf')
    pd.DataFrame({'precisions': res['precisions'], 'recalls': res['recalls']}
                 ).to_pickle(f'{data_root}/{ont}/pr_{model}.pkl')

if __name__ == '__main__':
    main()
