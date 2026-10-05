import json

import click as ck
import numpy as np
import pandas as pd
import torch as th

from metrics import compute_roc
from neuralgobox.data import get_data
from utils import get_goplus_defs

BATCH_SIZE = 1000


@ck.command()
@ck.option('--data-root', '-dr', default='data', help='DeepGOZero data folder')
@ck.option('--ont', '-ont', default='mf', type=ck.Choice(['mf', 'bp', 'cc']),
           help='GO sub-ontology')
@ck.option('--arm', type=ck.Choice(['neuralgobox', 'deepgozero']), required=True,
           help='Model the checkpoint belongs to')
@ck.option('--model-name', '-m', required=True,
           help='Checkpoint basename under data/{ont}/, e.g. neuralgobox_zs_sel-macro')
@ck.option('--out-file', '-o', default=None, help='Optional TSV of per-class AUCs')
@ck.option('--device', '-d', default='cuda:0', help='Device')
def main(data_root, ont, arm, model_name, out_file, device):
    df = pd.concat([pd.read_pickle(f'{data_root}/{ont}/{s}_data.pkl')
                    for s in ('train', 'valid', 'test')])
    terms = pd.read_pickle(f'{data_root}/{ont}/terms_zero_10.pkl')['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    iprs = pd.read_pickle(f'{data_root}/{ont}/interpros.pkl')['interpros'].values
    iprs_dict = {v: k for k, v in enumerate(iprs)}

    if arm == 'neuralgobox':
        from neuralgobox.data import load_normal_forms
        from neuralgobox.models import NeuralGOBoxModel
        buckets = load_normal_forms(f'{data_root}/go.norm', terms_dict)
        relations, zero_classes = buckets['relations'], buckets['zclasses']
        net = NeuralGOBoxModel(len(iprs_dict), len(terms_dict), len(zero_classes),
                               len(relations), device)
    else:
        from deepgozero import DGELModel, load_normal_forms
        *_, relations, zero_classes = load_normal_forms(f'{data_root}/go.norm', terms_dict)
        net = DGELModel(len(iprs_dict), len(terms_dict), len(zero_classes),
                        len(relations), device)
    net = net.to(device)
    net.load_state_dict(th.load(f'{data_root}/{ont}/{model_name}.th', map_location=device))
    net.eval()

    with open(f'{data_root}/eval_terms.json') as f:
        zero_terms = [t for t in json.load(f)[ont] if t in zero_classes]
    data, labels = get_data(df, iprs_dict, {t: i for i, t in enumerate(zero_terms)})
    labels = labels.numpy()
    go_ids = th.LongTensor([zero_classes[t] for t in zero_terms]).to(device)

    scores = []
    with th.no_grad():
        for i in range(0, len(data), BATCH_SIZE):
            out = net.predict_zero(data[i:i + BATCH_SIZE].to(device), go_ids)
            if arm == 'neuralgobox':
                out = th.sigmoid(out)
            scores.append(out.cpu().numpy())
    scores = np.concatenate(scores)

    definitions = get_goplus_defs(f'{data_root}/definitions_go.txt')
    rows = []
    for i, term in enumerate(zero_terms):
        n_pos = int(labels[:, i].sum())
        if 0 < n_pos < len(labels):
            rows.append((term, compute_roc(labels[:, i], scores[:, i]), n_pos,
                         term in definitions))
    res = pd.DataFrame(rows, columns=['term', 'auc', 'n_pos', 'defined'])
    print(f'all classes:     n={len(res)} AUC={res.auc.mean():.6f}')
    print(f'defined classes: n={res.defined.sum()} AUC={res[res.defined].auc.mean():.6f}')
    if out_file:
        res.to_csv(out_file, sep='\t', index=False)


if __name__ == '__main__':
    main()
