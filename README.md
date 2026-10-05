# NeuralGOBox

Code, training logs and result tables for

> Naman Singh, Ernesto Jiménez-Ruiz, Michael Akintunde and Tillman Weyde.
> *NeuralGOBox: Predicting Sparsely Annotated Protein Functions with Gene Ontology Box Embeddings.*
> SWAT4HCLS 2027.

NeuralGOBox replaces DeepGOZero's ELEm embedding of the Gene Ontology with an
ELK-Em box embedding, and scores proteins against each class's box. This
repository is built on [DeepGOZero](https://github.com/bio-ontology-research-group/deepgozero)
and runs on its data.

## Installation

Python 3.10 and a CUDA GPU.

```bash
pip install -r requirements.txt
```

[DIAMOND](https://github.com/bbuchfink/diamond) is needed only for `run_diamond.sh`;
the data already includes its output.

## Data

Download DeepGOZero's `data.tar.gz` from https://deepgo.cbrc.kaust.edu.sa/data/deepgozero/
and extract it into `data/`. Checkpoints and predictions are written to `data/<ont>/`,
where `<ont>` is `mf`, `bp` or `cc`.

## Training and evaluation

Each training run keeps three checkpoints, one per selection criterion
(validation cross-entropy, micro AUC, macro AUC), named `<model>_sel-<criterion>`,
and writes test predictions for each.

| Model | Command |
|---|---|
| NeuralGOBox | `python neuralgobox_train.py -ont <ont> -m neuralgobox_<ont>_s<seed> -s <seed> -ep 10` |
| MLPBox | `python neuralgobox_train.py -ont <ont> -m mlpbox_<ont>_s<seed> -s <seed> -ep 24 --el-weight 0` |
| DeepGOZero | `python deepgozero.py -ont <ont> -m deepgozero_<ont>_s<seed> -s <seed> -ep 128` |
| MLP | `python deepgopro.py -ont <ont> -m mlp_<ont>_s<seed> -s <seed> -ep 60` |

Evaluate each checkpoint `<m>` (e.g. `neuralgobox_mf_s0_sel-macro`):

```bash
python predict_diamond.py  -ont <ont> -m <m>                     # adds DiamondScore predictions
python evaluate_fast.py    -ont <ont> -m <m>_blast               # Fmax, Smin, AUPR
python evaluate_fast.py    -ont <ont> -m <m>_blast -c            # ... averaged with DiamondScore
python evaluate_terms.py   -ont <ont> -m <m>_blast               # class-centric AUC
python evaluate_terms.py   -ont <ont> -m <m>_blast -c            # ... averaged with DiamondScore
```

`--diamond-only` in place of `-c` evaluates DiamondScore alone.

### Zero-shot

```bash
python zero_data.py          # writes data/eval_terms.json, the held-out classes
python neuralgobox_train.py -ont <ont> -m neuralgobox_zs_<ont>_s<seed> -s <seed> -ep 10 --zero-shot
python deepgozero.py        -ont <ont> -m deepgozero_zs_<ont>_s<seed> -s <seed> -ep 128 --zero-shot
python evaluate_zero_shot.py -ont <ont> --arm neuralgobox -m neuralgobox_zs_<ont>_s<seed>_sel-macro
python evaluate_zero_shot.py -ont <ont> --arm deepgozero  -m deepgozero_zs_<ont>_s<seed>_sel-macro
```

## Results

`logs/` holds the logs of every run reported in the paper. `make_tables.py`
recomputes the paper's tables from them and checks each value against the
paper; `--markdown` prints the tables below. Values are means ± standard
deviations over seeds 0, 1 and 2; + D averages a model's scores with
DiamondScore's at equal weight, and Epochs gives the selected epoch for each
seed. The macro AUC rows are the paper's Table 3; the AUC column across
selection criteria is its Table 4.

**Supervised results, MFO.**

| Method | Selection | Fmax | Smin | AUPR | AUC | Epochs |
|---|---|---|---|---|---|---|
| DiamondScore | | 0.623 | 10.145 | 0.380 | 0.749 | |
| MLP | cross-entropy | 0.656 ± 0.001 | 9.832 ± 0.003 | 0.655 ± 0.003 | 0.880 ± 0.002 | 4/4/4 |
|  | cross-entropy + D | 0.669 ± 0.001 | 9.568 ± 0.010 | 0.647 ± 0.003 | 0.884 ± 0.002 |  |
|  | micro AUC | 0.657 ± 0.001 | 9.801 ± 0.019 | 0.654 ± 0.003 | 0.884 ± 0.002 | 8/8/8 |
|  | micro AUC + D | 0.670 ± 0.000 | 9.553 ± 0.007 | 0.644 ± 0.003 | 0.888 ± 0.002 |  |
|  | macro AUC | 0.656 ± 0.002 | 9.760 ± 0.045 | 0.630 ± 0.002 | 0.900 ± 0.001 | 39/44/36 |
|  | macro AUC + D | 0.668 ± 0.002 | 9.581 ± 0.006 | 0.624 ± 0.004 | 0.902 ± 0.000 |  |
| DeepGOZero | cross-entropy | 0.651 ± 0.001 | 9.917 ± 0.056 | 0.660 ± 0.003 | 0.905 ± 0.001 | 97/92/87 |
|  | cross-entropy + D | 0.667 ± 0.002 | 9.626 ± 0.023 | 0.676 ± 0.003 | 0.908 ± 0.001 |  |
|  | micro AUC | 0.655 ± 0.003 | 9.810 ± 0.012 | 0.662 ± 0.003 | 0.905 ± 0.001 | 104/106/113 |
|  | micro AUC + D | 0.666 ± 0.002 | 9.587 ± 0.002 | 0.676 ± 0.005 | 0.908 ± 0.001 |  |
|  | macro AUC | 0.655 ± 0.002 | 9.795 ± 0.030 | 0.662 ± 0.002 | 0.905 ± 0.002 | 110/113/105 |
|  | macro AUC + D | 0.665 ± 0.003 | 9.599 ± 0.003 | 0.676 ± 0.003 | 0.908 ± 0.002 |  |
| MLPBox | cross-entropy | 0.651 ± 0.008 | 10.000 ± 0.320 | 0.634 ± 0.008 | 0.886 ± 0.005 | 1/3/3 |
|  | cross-entropy + D | 0.666 ± 0.005 | 9.723 ± 0.179 | 0.638 ± 0.011 | 0.890 ± 0.004 |  |
|  | micro AUC | 0.655 ± 0.004 | 9.829 ± 0.055 | 0.638 ± 0.003 | 0.889 ± 0.001 | 3/3/3 |
|  | micro AUC + D | 0.668 ± 0.002 | 9.618 ± 0.019 | 0.631 ± 0.003 | 0.892 ± 0.001 |  |
|  | macro AUC | 0.654 ± 0.002 | 9.821 ± 0.030 | 0.616 ± 0.005 | 0.891 ± 0.002 | 9/10/13 |
|  | macro AUC + D | 0.668 ± 0.002 | 9.613 ± 0.004 | 0.617 ± 0.003 | 0.894 ± 0.002 |  |
| NeuralGOBox | cross-entropy | 0.653 ± 0.003 | 9.844 ± 0.046 | 0.645 ± 0.005 | 0.924 ± 0.002 | 3/3/3 |
|  | cross-entropy + D | 0.668 ± 0.003 | 9.604 ± 0.009 | 0.637 ± 0.003 | 0.926 ± 0.002 |  |
|  | micro AUC | 0.653 ± 0.003 | 9.844 ± 0.046 | 0.645 ± 0.005 | 0.924 ± 0.002 | 3/3/3 |
|  | micro AUC + D | 0.668 ± 0.003 | 9.604 ± 0.009 | 0.637 ± 0.003 | 0.926 ± 0.002 |  |
|  | macro AUC | 0.653 ± 0.003 | 9.844 ± 0.046 | 0.645 ± 0.005 | 0.924 ± 0.002 | 3/3/3 |
|  | macro AUC + D | 0.668 ± 0.003 | 9.604 ± 0.009 | 0.637 ± 0.003 | 0.926 ± 0.002 |  |

**Supervised results, BPO.**

| Method | Selection | Fmax | Smin | AUPR | AUC | Epochs |
|---|---|---|---|---|---|---|
| DiamondScore | | 0.444 | 45.040 | 0.313 | 0.610 | |
| MLP | cross-entropy | 0.456 ± 0.001 | 44.181 ± 0.069 | 0.430 ± 0.002 | 0.784 ± 0.003 | 2/2/2 |
|  | cross-entropy + D | 0.486 ± 0.002 | 43.880 ± 0.020 | 0.449 ± 0.001 | 0.789 ± 0.003 |  |
|  | micro AUC | 0.462 ± 0.003 | 44.002 ± 0.077 | 0.436 ± 0.002 | 0.798 ± 0.002 | 6/7/5 |
|  | micro AUC + D | 0.487 ± 0.001 | 43.849 ± 0.031 | 0.449 ± 0.001 | 0.801 ± 0.002 |  |
|  | macro AUC | 0.463 ± 0.001 | 44.185 ± 0.032 | 0.434 ± 0.001 | 0.814 ± 0.002 | 33/36/36 |
|  | macro AUC + D | 0.485 ± 0.001 | 43.945 ± 0.019 | 0.444 ± 0.001 | 0.816 ± 0.002 |  |
| DeepGOZero | cross-entropy | 0.451 ± 0.000 | 44.667 ± 0.016 | 0.420 ± 0.001 | 0.800 ± 0.004 | 20/21/21 |
|  | cross-entropy + D | 0.484 ± 0.001 | 44.020 ± 0.036 | 0.446 ± 0.001 | 0.804 ± 0.004 |  |
|  | micro AUC | 0.452 ± 0.001 | 44.612 ± 0.065 | 0.422 ± 0.001 | 0.804 ± 0.002 | 20/24/31 |
|  | micro AUC + D | 0.483 ± 0.001 | 44.020 ± 0.014 | 0.446 ± 0.001 | 0.808 ± 0.002 |  |
|  | macro AUC | 0.456 ± 0.001 | 44.628 ± 0.019 | 0.423 ± 0.001 | 0.814 ± 0.002 | 65/63/70 |
|  | macro AUC + D | 0.484 ± 0.001 | 44.132 ± 0.035 | 0.440 ± 0.001 | 0.816 ± 0.002 |  |
| MLPBox | cross-entropy | 0.447 ± 0.002 | 45.122 ± 0.261 | 0.411 ± 0.003 | 0.785 ± 0.004 | 1/1/1 |
|  | cross-entropy + D | 0.482 ± 0.000 | 44.268 ± 0.089 | 0.443 ± 0.001 | 0.789 ± 0.004 |  |
|  | micro AUC | 0.454 ± 0.006 | 44.894 ± 0.469 | 0.423 ± 0.007 | 0.796 ± 0.006 | 3/2/2 |
|  | micro AUC + D | 0.483 ± 0.001 | 44.266 ± 0.251 | 0.441 ± 0.002 | 0.798 ± 0.006 |  |
|  | macro AUC | 0.461 ± 0.001 | 44.480 ± 0.104 | 0.430 ± 0.002 | 0.804 ± 0.002 | 4/5/3 |
|  | macro AUC + D | 0.485 ± 0.001 | 44.040 ± 0.029 | 0.439 ± 0.003 | 0.805 ± 0.002 |  |
| NeuralGOBox | cross-entropy | 0.447 ± 0.001 | 44.920 ± 0.166 | 0.412 ± 0.005 | 0.808 ± 0.002 | 1/1/1 |
|  | cross-entropy + D | 0.482 ± 0.001 | 44.123 ± 0.047 | 0.444 ± 0.001 | 0.813 ± 0.002 |  |
|  | micro AUC | 0.454 ± 0.006 | 44.575 ± 0.429 | 0.423 ± 0.009 | 0.819 ± 0.006 | 3/2/2 |
|  | micro AUC + D | 0.481 ± 0.003 | 44.045 ± 0.132 | 0.442 ± 0.001 | 0.821 ± 0.005 |  |
|  | macro AUC | 0.460 ± 0.001 | 44.257 ± 0.070 | 0.430 ± 0.002 | 0.829 ± 0.001 | 6/4/5 |
|  | macro AUC + D | 0.485 ± 0.002 | 43.943 ± 0.014 | 0.439 ± 0.001 | 0.831 ± 0.002 |  |

**Supervised results, CCO.**

| Method | Selection | Fmax | Smin | AUPR | AUC | Epochs |
|---|---|---|---|---|---|---|
| DiamondScore | | 0.581 | 11.092 | 0.352 | 0.652 | |
| MLP | cross-entropy | 0.665 ± 0.001 | 10.555 ± 0.072 | 0.669 ± 0.003 | 0.843 ± 0.002 | 2/2/2 |
|  | cross-entropy + D | 0.666 ± 0.001 | 10.542 ± 0.026 | 0.656 ± 0.001 | 0.848 ± 0.002 |  |
|  | micro AUC | 0.665 ± 0.001 | 10.529 ± 0.006 | 0.654 ± 0.002 | 0.861 ± 0.001 | 10/12/14 |
|  | micro AUC + D | 0.666 ± 0.001 | 10.547 ± 0.014 | 0.642 ± 0.000 | 0.864 ± 0.001 |  |
|  | macro AUC | 0.662 ± 0.001 | 10.632 ± 0.035 | 0.643 ± 0.001 | 0.865 ± 0.001 | 28/36/33 |
|  | macro AUC + D | 0.665 ± 0.000 | 10.587 ± 0.008 | 0.634 ± 0.002 | 0.868 ± 0.001 |  |
| DeepGOZero | cross-entropy | 0.663 ± 0.002 | 10.625 ± 0.051 | 0.655 ± 0.013 | 0.849 ± 0.005 | 17/17/16 |
|  | cross-entropy + D | 0.665 ± 0.001 | 10.560 ± 0.022 | 0.666 ± 0.032 | 0.854 ± 0.004 |  |
|  | micro AUC | 0.662 ± 0.002 | 10.583 ± 0.016 | 0.665 ± 0.003 | 0.857 ± 0.005 | 19/21/17 |
|  | micro AUC + D | 0.666 ± 0.001 | 10.560 ± 0.016 | 0.684 ± 0.029 | 0.861 ± 0.004 |  |
|  | macro AUC | 0.660 ± 0.002 | 10.647 ± 0.049 | 0.647 ± 0.009 | 0.860 ± 0.000 | 41/42/21 |
|  | macro AUC + D | 0.665 ± 0.002 | 10.579 ± 0.024 | 0.656 ± 0.037 | 0.864 ± 0.000 |  |
| MLPBox | cross-entropy | 0.659 ± 0.002 | 10.767 ± 0.035 | 0.667 ± 0.015 | 0.843 ± 0.004 | 2/1/1 |
|  | cross-entropy + D | 0.664 ± 0.001 | 10.622 ± 0.046 | 0.657 ± 0.010 | 0.848 ± 0.004 |  |
|  | micro AUC | 0.662 ± 0.001 | 10.681 ± 0.047 | 0.650 ± 0.004 | 0.857 ± 0.003 | 3/3/3 |
|  | micro AUC + D | 0.666 ± 0.001 | 10.589 ± 0.017 | 0.638 ± 0.001 | 0.861 ± 0.002 |  |
|  | macro AUC | 0.658 ± 0.001 | 10.726 ± 0.023 | 0.633 ± 0.003 | 0.859 ± 0.002 | 11/10/8 |
|  | macro AUC + D | 0.665 ± 0.001 | 10.631 ± 0.014 | 0.624 ± 0.001 | 0.862 ± 0.002 |  |
| NeuralGOBox | cross-entropy | 0.660 ± 0.004 | 10.724 ± 0.105 | 0.672 ± 0.015 | 0.866 ± 0.004 | 2/1/1 |
|  | cross-entropy + D | 0.665 ± 0.001 | 10.577 ± 0.064 | 0.659 ± 0.009 | 0.870 ± 0.004 |  |
|  | micro AUC | 0.663 ± 0.001 | 10.613 ± 0.038 | 0.696 ± 0.038 | 0.879 ± 0.001 | 3/3/3 |
|  | micro AUC + D | 0.666 ± 0.001 | 10.563 ± 0.011 | 0.640 ± 0.001 | 0.882 ± 0.001 |  |
|  | macro AUC | 0.661 ± 0.000 | 10.657 ± 0.034 | 0.641 ± 0.005 | 0.880 ± 0.001 | 8/5/7 |
|  | macro AUC + D | 0.666 ± 0.001 | 10.590 ± 0.004 | 0.630 ± 0.002 | 0.882 ± 0.001 |  |

**Table 5, zero-shot class-centric AUC under each selection criterion, on all held-out classes and on those GO defines by an equivalence.**

| Method | Selection | All MFO | All BPO | All CCO | Defined MFO | Defined BPO | Defined CCO |
|---|---|---|---|---|---|---|---|
| DeepGOZero | cross-entropy | 0.814 ± 0.014 | 0.735 ± 0.001 | 0.818 ± 0.001 | 0.877 ± 0.006 | 0.788 ± 0.003 | 0.917 ± 0.005 |
|  | micro AUC | 0.812 ± 0.015 | 0.736 ± 0.001 | 0.822 ± 0.001 | 0.876 ± 0.007 | 0.789 ± 0.002 | 0.923 ± 0.004 |
|  | macro AUC | 0.812 ± 0.015 | 0.734 ± 0.002 | 0.823 ± 0.000 | 0.876 ± 0.006 | 0.788 ± 0.003 | 0.928 ± 0.001 |
| NeuralGOBox | cross-entropy | 0.926 ± 0.013 | 0.829 ± 0.001 | 0.853 ± 0.011 | 0.901 ± 0.009 | 0.802 ± 0.001 | 0.928 ± 0.005 |
|  | micro AUC | 0.910 ± 0.002 | 0.836 ± 0.001 | 0.847 ± 0.006 | 0.910 ± 0.001 | 0.812 ± 0.002 | 0.934 ± 0.001 |
|  | macro AUC | 0.910 ± 0.002 | 0.835 ± 0.000 | 0.848 ± 0.007 | 0.909 ± 0.003 | 0.812 ± 0.000 | 0.937 ± 0.002 |

Run on an NVIDIA A100 GPU with the versions in `requirements.txt`, the commands
above reproduce the reported runs exactly: training logs, checkpoints and test
predictions. On other GPUs, results can differ slightly through floating-point
arithmetic.

## Repository layout

```
neuralgobox/            NeuralGOBox model: box geometry, axiom losses, scorer, data loading
neuralgobox_train.py    trains NeuralGOBox and MLPBox
evaluate_zero_shot.py   zero-shot evaluation of NeuralGOBox and DeepGOZero
evaluate_fast.py        Fmax, Smin, AUPR; vectorised equivalent of DeepGOZero's evaluate.py
metrics.py              micro and macro AUC for checkpoint selection
make_tables.py          recomputes the paper's tables from logs/
test_neuralgobox.py     tests of the NeuralGOBox model
logs/                   logs of every reported run

From DeepGOZero, modified:
deepgozero.py           DeepGOZero
deepgopro.py            MLP baseline
evaluate_terms.py       class-centric AUC
zero_data.py            held-out classes for zero-shot evaluation

From DeepGOZero, unchanged:
predict_diamond.py      DiamondScore
run_diamond.sh          DIAMOND hits (with diamond_data.py)
utils.py, torch_utils.py, aminoacids.py
```

Our files also adapt parts of DeepGOZero's code.

## License

BSD 3-Clause; see `LICENSE`.
