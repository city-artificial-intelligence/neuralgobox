import click as ck
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

ONTS = (('mf', 'MFO'), ('bp', 'BPO'), ('cc', 'CCO'))
MODELS = (('mlp', 'MLP'), ('deepgozero', 'DeepGOZero'), ('mlpbox', 'MLPBox'),
          ('neuralgobox', 'NeuralGOBox'))
BINS = ((1, 5), (6, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 70), (71, 100),
        (101, None))
LABELS = [f'{lo}–{hi}' if hi else '>100' for lo, hi in BINS]


@ck.command()
@ck.option('--data-root', '-dr', default='data', help='DeepGOZero data folder')
@ck.option('--out-file', '-o', default='figure2.pdf', help='Output figure')
def main(data_root, out_file):
    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    for ax, (ont, ont_name) in zip(axes, ONTS):
        print(ont_name)
        for model, name in MODELS:
            per_seed = []
            for seed in (0, 1, 2):
                df = pd.read_pickle(f'{data_root}/{ont}/{model}_{ont}_s{seed}_sel-macro_blast_auc_annots.pkl')
                per_seed.append([df.aucs[(df.annots >= lo) & ((df.annots <= hi) if hi else True)].mean()
                                 for lo, hi in BINS])
            means = np.mean(per_seed, axis=0)
            print(f'  {name:<12}' + ' '.join(f'{m:.3f}' for m in means))
            ax.plot(LABELS, means, marker='o', label=name)
        counts = [int(((df.annots >= lo) & ((df.annots <= hi) if hi else True)).sum()) for lo, hi in BINS]
        print(f'  {"classes":<12}' + ' '.join(f'{c:>5}' for c in counts))
        ax.set_title(ont_name)
        ax.set_ylabel('class-centric AUC')
    axes[-1].set_xlabel('annotations per class')
    axes[-1].legend(ncol=4, loc='lower right')
    fig.tight_layout()
    fig.savefig(out_file)


if __name__ == '__main__':
    main()
