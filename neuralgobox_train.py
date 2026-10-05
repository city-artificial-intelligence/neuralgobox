import math
from functools import partial
from multiprocessing import Pool

import click as ck
import numpy as np
import torch as th
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR

from metrics import CRITERIA, compute_roc, compute_term_auc
from neuralgobox.data import (buckets_to_tensors, load_data, load_normal_forms,
                              propagate_annots)
from neuralgobox.models import NeuralGOBoxModel
from torch_utils import FastTensorDataLoader
from utils import Ontology

BATCH_SIZE = 37
LR = 5e-4
MILESTONES = [2, 4]
EL_WEIGHT = {'mf': 0.016, 'bp': 0.002, 'cc': 0.016}


@ck.command()
@ck.option('--data-root', '-dr', default='data', help='DeepGOZero data folder')
@ck.option('--ont', '-ont', default='mf', type=ck.Choice(['mf', 'bp', 'cc']),
           help='GO sub-ontology')
@ck.option('--zero-shot', is_flag=True,
           help='Train on the zero-shot vocabulary (terms_zero_10.pkl) instead '
                'of the full one (terms.pkl)')
@ck.option('--model-name', '-m', default='neuralgobox',
           help='Basename for the checkpoints and predictions')
@ck.option('--epochs', '-ep', default=10, help='Training epochs')
@ck.option('--el-weight', default=None, type=float,
           help='Axiom loss weight lambda; defaults to the selected value for '
                'the sub-ontology. 0 trains MLPBox')
@ck.option('--seed', '-s', default=0, type=int, help='Random seed')
@ck.option('--load', '-ld', is_flag=True,
           help='Skip training and evaluate the saved checkpoints')
@ck.option('--device', '-d', default='cuda:0', help='Device')
def main(data_root, ont, zero_shot, model_name, epochs, el_weight, seed, load,
         device):
    th.manual_seed(seed)
    np.random.seed(seed)
    if el_weight is None:
        el_weight = EL_WEIGHT[ont]

    terms_file = f'{data_root}/{ont}/' + ('terms_zero_10.pkl' if zero_shot
                                          else 'terms.pkl')
    ckpt_file = lambda c: f'{data_root}/{ont}/{model_name}_sel-{c}.th'
    preds_file = lambda c: f'{data_root}/{ont}/predictions_{model_name}_sel-{c}.pkl'
    print(f'config: {model_name} ont={ont} seed={seed} epochs={epochs} '
          f'el_weight={el_weight} terms={terms_file}')

    go = Ontology(f'{data_root}/go.obo', with_rels=True)
    iprs_dict, terms_dict, train_data, valid_data, test_data, test_df = load_data(
        data_root, ont, terms_file)
    n_terms = len(terms_dict)
    _, train_labels = train_data
    valid_labels = valid_data[1].numpy()
    test_labels = test_data[1].numpy()

    buckets = load_normal_forms(f'{data_root}/go.norm', terms_dict)
    n_rels = len(buckets['relations'])
    n_zeros = len(buckets['zclasses'])
    normal_forms = buckets_to_tensors(buckets, device)
    print(f'classes: {n_terms} trained + {n_zeros} axiom-only, relations: {n_rels}')

    train_loader = FastTensorDataLoader(*train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = FastTensorDataLoader(*valid_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = FastTensorDataLoader(*test_data, batch_size=BATCH_SIZE, shuffle=False)

    net = NeuralGOBoxModel(len(iprs_dict), n_terms, n_zeros, n_rels, device).to(device)
    optimizer = th.optim.Adam(net.parameters(), lr=LR)
    scheduler = MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.1)

    def run_eval(loader, n_labels):
        steps = int(math.ceil(n_labels / BATCH_SIZE))
        total, chunks = 0.0, []
        with th.no_grad():
            for bf, bl in loader:
                logits = net(bf.to(device))
                total += F.binary_cross_entropy_with_logits(
                    logits, bl.to(device)).detach().item()
                chunks.append(th.sigmoid(logits).detach().cpu().numpy())
        return total / steps, np.concatenate(chunks)

    if not load:
        best = {c: float('-inf') for c in CRITERIA}
        th.manual_seed(seed + 1)
        for epoch in range(epochs):
            net.train()
            train_loss = train_elloss = 0.0
            train_steps = int(math.ceil(len(train_labels) / BATCH_SIZE))
            with ck.progressbar(length=train_steps, show_pos=True) as bar:
                for batch_features, batch_labels in train_loader:
                    bar.update(1)
                    logits = net(batch_features.to(device))
                    loss = F.binary_cross_entropy_with_logits(
                        logits, batch_labels.to(device))
                    if el_weight:
                        el_loss = net.el_loss(normal_forms)
                        total_loss = loss + el_weight * el_loss
                        train_elloss += el_loss.detach().item()
                    else:
                        total_loss = loss
                    train_loss += loss.detach().item()
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
            train_loss /= train_steps
            train_elloss /= train_steps

            net.eval()
            valid_loss, preds = run_eval(valid_loader, len(valid_labels))
            scores = {'loss': -valid_loss,
                      'micro': compute_roc(valid_labels, preds),
                      'macro': compute_term_auc(valid_labels, preds)}
            print(f'Epoch {epoch}: Loss - {train_loss}, EL Loss: {train_elloss}, '
                  f'Valid loss - {valid_loss}, AUC - {scores["micro"]}, '
                  f'termAUC - {scores["macro"]}, '
                  f'lr - {optimizer.param_groups[0]["lr"]}')
            for c in CRITERIA:
                if scores[c] > best[c]:
                    best[c] = scores[c]
                    print(f'Saving model (sel-{c})')
                    th.save(net.state_dict(), ckpt_file(c))
            scheduler.step()

    for c in CRITERIA:
        print(f'Loading the best model: {ckpt_file(c)}')
        net.load_state_dict(th.load(ckpt_file(c)))
        net.eval()
        test_loss, preds = run_eval(test_loader, len(test_labels))
        print(f'Test Loss - {test_loss}, AUC - {compute_roc(test_labels, preds)}')
        with Pool(16) as p:
            preds = p.map(partial(propagate_annots, go=go, terms_dict=terms_dict),
                          list(preds))
        test_df['preds'] = preds
        test_df.to_pickle(preds_file(c))
        print(f'Saved {preds_file(c)}')


if __name__ == '__main__':
    main()
