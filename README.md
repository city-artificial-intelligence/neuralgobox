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
| MLP | cross-entropy | 0.656<sub>±0.001</sub> | 9.832<sub>±0.003</sub> | 0.655<sub>±0.003</sub> | 0.880<sub>±0.002</sub> | 4/4/4 |
| + D | cross-entropy | 0.669<sub>±0.001</sub> | 9.568<sub>±0.010</sub> | 0.647<sub>±0.003</sub> | 0.884<sub>±0.002</sub> |  |
|  | micro AUC | 0.657<sub>±0.001</sub> | 9.801<sub>±0.019</sub> | 0.654<sub>±0.003</sub> | 0.884<sub>±0.002</sub> | 8/8/8 |
| + D | micro AUC | 0.670<sub>±0.000</sub> | 9.553<sub>±0.007</sub> | 0.644<sub>±0.003</sub> | 0.888<sub>±0.002</sub> |  |
|  | macro AUC | 0.656<sub>±0.002</sub> | 9.760<sub>±0.045</sub> | 0.630<sub>±0.002</sub> | 0.900<sub>±0.001</sub> | 39/44/36 |
| + D | macro AUC | 0.668<sub>±0.002</sub> | 9.581<sub>±0.006</sub> | 0.624<sub>±0.004</sub> | 0.902<sub>±0.000</sub> |  |
| DeepGOZero | cross-entropy | 0.651<sub>±0.001</sub> | 9.917<sub>±0.056</sub> | 0.660<sub>±0.003</sub> | 0.905<sub>±0.001</sub> | 97/92/87 |
| + D | cross-entropy | 0.667<sub>±0.002</sub> | 9.626<sub>±0.023</sub> | 0.676<sub>±0.003</sub> | 0.908<sub>±0.001</sub> |  |
|  | micro AUC | 0.655<sub>±0.003</sub> | 9.810<sub>±0.012</sub> | 0.662<sub>±0.003</sub> | 0.905<sub>±0.001</sub> | 104/106/113 |
| + D | micro AUC | 0.666<sub>±0.002</sub> | 9.587<sub>±0.002</sub> | 0.676<sub>±0.005</sub> | 0.908<sub>±0.001</sub> |  |
|  | macro AUC | 0.655<sub>±0.002</sub> | 9.795<sub>±0.030</sub> | 0.662<sub>±0.002</sub> | 0.905<sub>±0.002</sub> | 110/113/105 |
| + D | macro AUC | 0.665<sub>±0.003</sub> | 9.599<sub>±0.003</sub> | 0.676<sub>±0.003</sub> | 0.908<sub>±0.002</sub> |  |
| MLPBox | cross-entropy | 0.651<sub>±0.008</sub> | 10.000<sub>±0.320</sub> | 0.634<sub>±0.008</sub> | 0.886<sub>±0.005</sub> | 1/3/3 |
| + D | cross-entropy | 0.666<sub>±0.005</sub> | 9.723<sub>±0.179</sub> | 0.638<sub>±0.011</sub> | 0.890<sub>±0.004</sub> |  |
|  | micro AUC | 0.655<sub>±0.004</sub> | 9.829<sub>±0.055</sub> | 0.638<sub>±0.003</sub> | 0.889<sub>±0.001</sub> | 3/3/3 |
| + D | micro AUC | 0.668<sub>±0.002</sub> | 9.618<sub>±0.019</sub> | 0.631<sub>±0.003</sub> | 0.892<sub>±0.001</sub> |  |
|  | macro AUC | 0.654<sub>±0.002</sub> | 9.821<sub>±0.030</sub> | 0.616<sub>±0.005</sub> | 0.891<sub>±0.002</sub> | 9/10/13 |
| + D | macro AUC | 0.668<sub>±0.002</sub> | 9.613<sub>±0.004</sub> | 0.617<sub>±0.003</sub> | 0.894<sub>±0.002</sub> |  |
| NeuralGOBox | cross-entropy | 0.653<sub>±0.003</sub> | 9.844<sub>±0.046</sub> | 0.645<sub>±0.005</sub> | 0.924<sub>±0.002</sub> | 3/3/3 |
| + D | cross-entropy | 0.668<sub>±0.003</sub> | 9.604<sub>±0.009</sub> | 0.637<sub>±0.003</sub> | 0.926<sub>±0.002</sub> |  |
|  | micro AUC | 0.653<sub>±0.003</sub> | 9.844<sub>±0.046</sub> | 0.645<sub>±0.005</sub> | 0.924<sub>±0.002</sub> | 3/3/3 |
| + D | micro AUC | 0.668<sub>±0.003</sub> | 9.604<sub>±0.009</sub> | 0.637<sub>±0.003</sub> | 0.926<sub>±0.002</sub> |  |
|  | macro AUC | 0.653<sub>±0.003</sub> | 9.844<sub>±0.046</sub> | 0.645<sub>±0.005</sub> | 0.924<sub>±0.002</sub> | 3/3/3 |
| + D | macro AUC | 0.668<sub>±0.003</sub> | 9.604<sub>±0.009</sub> | 0.637<sub>±0.003</sub> | 0.926<sub>±0.002</sub> |  |

**Supervised results, BPO.**

| Method | Selection | Fmax | Smin | AUPR | AUC | Epochs |
|---|---|---|---|---|---|---|
| DiamondScore | | 0.444 | 45.040 | 0.313 | 0.610 | |
| MLP | cross-entropy | 0.456<sub>±0.001</sub> | 44.181<sub>±0.069</sub> | 0.430<sub>±0.002</sub> | 0.784<sub>±0.003</sub> | 2/2/2 |
| + D | cross-entropy | 0.486<sub>±0.002</sub> | 43.880<sub>±0.020</sub> | 0.449<sub>±0.001</sub> | 0.789<sub>±0.003</sub> |  |
|  | micro AUC | 0.462<sub>±0.003</sub> | 44.002<sub>±0.077</sub> | 0.436<sub>±0.002</sub> | 0.798<sub>±0.002</sub> | 6/7/5 |
| + D | micro AUC | 0.487<sub>±0.001</sub> | 43.849<sub>±0.031</sub> | 0.449<sub>±0.001</sub> | 0.801<sub>±0.002</sub> |  |
|  | macro AUC | 0.463<sub>±0.001</sub> | 44.185<sub>±0.032</sub> | 0.434<sub>±0.001</sub> | 0.814<sub>±0.002</sub> | 33/36/36 |
| + D | macro AUC | 0.485<sub>±0.001</sub> | 43.945<sub>±0.019</sub> | 0.444<sub>±0.001</sub> | 0.816<sub>±0.002</sub> |  |
| DeepGOZero | cross-entropy | 0.451<sub>±0.000</sub> | 44.667<sub>±0.016</sub> | 0.420<sub>±0.001</sub> | 0.800<sub>±0.004</sub> | 20/21/21 |
| + D | cross-entropy | 0.484<sub>±0.001</sub> | 44.020<sub>±0.036</sub> | 0.446<sub>±0.001</sub> | 0.804<sub>±0.004</sub> |  |
|  | micro AUC | 0.452<sub>±0.001</sub> | 44.612<sub>±0.065</sub> | 0.422<sub>±0.001</sub> | 0.804<sub>±0.002</sub> | 20/24/31 |
| + D | micro AUC | 0.483<sub>±0.001</sub> | 44.020<sub>±0.014</sub> | 0.446<sub>±0.001</sub> | 0.808<sub>±0.002</sub> |  |
|  | macro AUC | 0.456<sub>±0.001</sub> | 44.628<sub>±0.019</sub> | 0.423<sub>±0.001</sub> | 0.814<sub>±0.002</sub> | 65/63/70 |
| + D | macro AUC | 0.484<sub>±0.001</sub> | 44.132<sub>±0.035</sub> | 0.440<sub>±0.001</sub> | 0.816<sub>±0.002</sub> |  |
| MLPBox | cross-entropy | 0.447<sub>±0.002</sub> | 45.122<sub>±0.261</sub> | 0.411<sub>±0.003</sub> | 0.785<sub>±0.004</sub> | 1/1/1 |
| + D | cross-entropy | 0.482<sub>±0.000</sub> | 44.268<sub>±0.089</sub> | 0.443<sub>±0.001</sub> | 0.789<sub>±0.004</sub> |  |
|  | micro AUC | 0.454<sub>±0.006</sub> | 44.894<sub>±0.469</sub> | 0.423<sub>±0.007</sub> | 0.796<sub>±0.006</sub> | 3/2/2 |
| + D | micro AUC | 0.483<sub>±0.001</sub> | 44.266<sub>±0.251</sub> | 0.441<sub>±0.002</sub> | 0.798<sub>±0.006</sub> |  |
|  | macro AUC | 0.461<sub>±0.001</sub> | 44.480<sub>±0.104</sub> | 0.430<sub>±0.002</sub> | 0.804<sub>±0.002</sub> | 4/5/3 |
| + D | macro AUC | 0.485<sub>±0.001</sub> | 44.040<sub>±0.029</sub> | 0.439<sub>±0.003</sub> | 0.805<sub>±0.002</sub> |  |
| NeuralGOBox | cross-entropy | 0.447<sub>±0.001</sub> | 44.920<sub>±0.166</sub> | 0.412<sub>±0.005</sub> | 0.808<sub>±0.002</sub> | 1/1/1 |
| + D | cross-entropy | 0.482<sub>±0.001</sub> | 44.123<sub>±0.047</sub> | 0.444<sub>±0.001</sub> | 0.813<sub>±0.002</sub> |  |
|  | micro AUC | 0.454<sub>±0.006</sub> | 44.575<sub>±0.429</sub> | 0.423<sub>±0.009</sub> | 0.819<sub>±0.006</sub> | 3/2/2 |
| + D | micro AUC | 0.481<sub>±0.003</sub> | 44.045<sub>±0.132</sub> | 0.442<sub>±0.001</sub> | 0.821<sub>±0.005</sub> |  |
|  | macro AUC | 0.460<sub>±0.001</sub> | 44.257<sub>±0.070</sub> | 0.430<sub>±0.002</sub> | 0.829<sub>±0.001</sub> | 6/4/5 |
| + D | macro AUC | 0.485<sub>±0.002</sub> | 43.943<sub>±0.014</sub> | 0.439<sub>±0.001</sub> | 0.831<sub>±0.002</sub> |  |

**Supervised results, CCO.**

| Method | Selection | Fmax | Smin | AUPR | AUC | Epochs |
|---|---|---|---|---|---|---|
| DiamondScore | | 0.581 | 11.092 | 0.352 | 0.652 | |
| MLP | cross-entropy | 0.665<sub>±0.001</sub> | 10.555<sub>±0.072</sub> | 0.669<sub>±0.003</sub> | 0.843<sub>±0.002</sub> | 2/2/2 |
| + D | cross-entropy | 0.666<sub>±0.001</sub> | 10.542<sub>±0.026</sub> | 0.656<sub>±0.001</sub> | 0.848<sub>±0.002</sub> |  |
|  | micro AUC | 0.665<sub>±0.001</sub> | 10.529<sub>±0.006</sub> | 0.654<sub>±0.002</sub> | 0.861<sub>±0.001</sub> | 10/12/14 |
| + D | micro AUC | 0.666<sub>±0.001</sub> | 10.547<sub>±0.014</sub> | 0.642<sub>±0.000</sub> | 0.864<sub>±0.001</sub> |  |
|  | macro AUC | 0.662<sub>±0.001</sub> | 10.632<sub>±0.035</sub> | 0.643<sub>±0.001</sub> | 0.865<sub>±0.001</sub> | 28/36/33 |
| + D | macro AUC | 0.665<sub>±0.000</sub> | 10.587<sub>±0.008</sub> | 0.634<sub>±0.002</sub> | 0.868<sub>±0.001</sub> |  |
| DeepGOZero | cross-entropy | 0.663<sub>±0.002</sub> | 10.625<sub>±0.051</sub> | 0.655<sub>±0.013</sub> | 0.849<sub>±0.005</sub> | 17/17/16 |
| + D | cross-entropy | 0.665<sub>±0.001</sub> | 10.560<sub>±0.022</sub> | 0.666<sub>±0.032</sub> | 0.854<sub>±0.004</sub> |  |
|  | micro AUC | 0.662<sub>±0.002</sub> | 10.583<sub>±0.016</sub> | 0.665<sub>±0.003</sub> | 0.857<sub>±0.005</sub> | 19/21/17 |
| + D | micro AUC | 0.666<sub>±0.001</sub> | 10.560<sub>±0.016</sub> | 0.684<sub>±0.029</sub> | 0.861<sub>±0.004</sub> |  |
|  | macro AUC | 0.660<sub>±0.002</sub> | 10.647<sub>±0.049</sub> | 0.647<sub>±0.009</sub> | 0.860<sub>±0.000</sub> | 41/42/21 |
| + D | macro AUC | 0.665<sub>±0.002</sub> | 10.579<sub>±0.024</sub> | 0.656<sub>±0.037</sub> | 0.864<sub>±0.000</sub> |  |
| MLPBox | cross-entropy | 0.659<sub>±0.002</sub> | 10.767<sub>±0.035</sub> | 0.667<sub>±0.015</sub> | 0.843<sub>±0.004</sub> | 2/1/1 |
| + D | cross-entropy | 0.664<sub>±0.001</sub> | 10.622<sub>±0.046</sub> | 0.657<sub>±0.010</sub> | 0.848<sub>±0.004</sub> |  |
|  | micro AUC | 0.662<sub>±0.001</sub> | 10.681<sub>±0.047</sub> | 0.650<sub>±0.004</sub> | 0.857<sub>±0.003</sub> | 3/3/3 |
| + D | micro AUC | 0.666<sub>±0.001</sub> | 10.589<sub>±0.017</sub> | 0.638<sub>±0.001</sub> | 0.861<sub>±0.002</sub> |  |
|  | macro AUC | 0.658<sub>±0.001</sub> | 10.726<sub>±0.023</sub> | 0.633<sub>±0.003</sub> | 0.859<sub>±0.002</sub> | 11/10/8 |
| + D | macro AUC | 0.665<sub>±0.001</sub> | 10.631<sub>±0.014</sub> | 0.624<sub>±0.001</sub> | 0.862<sub>±0.002</sub> |  |
| NeuralGOBox | cross-entropy | 0.660<sub>±0.004</sub> | 10.724<sub>±0.105</sub> | 0.672<sub>±0.015</sub> | 0.866<sub>±0.004</sub> | 2/1/1 |
| + D | cross-entropy | 0.665<sub>±0.001</sub> | 10.577<sub>±0.064</sub> | 0.659<sub>±0.009</sub> | 0.870<sub>±0.004</sub> |  |
|  | micro AUC | 0.663<sub>±0.001</sub> | 10.613<sub>±0.038</sub> | 0.696<sub>±0.038</sub> | 0.879<sub>±0.001</sub> | 3/3/3 |
| + D | micro AUC | 0.666<sub>±0.001</sub> | 10.563<sub>±0.011</sub> | 0.640<sub>±0.001</sub> | 0.882<sub>±0.001</sub> |  |
|  | macro AUC | 0.661<sub>±0.000</sub> | 10.657<sub>±0.034</sub> | 0.641<sub>±0.005</sub> | 0.880<sub>±0.001</sub> | 8/5/7 |
| + D | macro AUC | 0.666<sub>±0.001</sub> | 10.590<sub>±0.004</sub> | 0.630<sub>±0.002</sub> | 0.882<sub>±0.001</sub> |  |

**Table 5, zero-shot class-centric AUC under each selection criterion, on all held-out classes and on those GO defines by an equivalence.**

| Method | Selection | All MFO | All BPO | All CCO | Defined MFO | Defined BPO | Defined CCO |
|---|---|---|---|---|---|---|---|
| DeepGOZero | cross-entropy | 0.814<sub>±0.014</sub> | 0.735<sub>±0.001</sub> | 0.818<sub>±0.001</sub> | 0.877<sub>±0.006</sub> | 0.788<sub>±0.003</sub> | 0.917<sub>±0.005</sub> |
|  | micro AUC | 0.812<sub>±0.015</sub> | 0.736<sub>±0.001</sub> | 0.822<sub>±0.001</sub> | 0.876<sub>±0.007</sub> | 0.789<sub>±0.002</sub> | 0.923<sub>±0.004</sub> |
|  | macro AUC | 0.812<sub>±0.015</sub> | 0.734<sub>±0.002</sub> | 0.823<sub>±0.000</sub> | 0.876<sub>±0.006</sub> | 0.788<sub>±0.003</sub> | 0.928<sub>±0.001</sub> |
| NeuralGOBox | cross-entropy | 0.926<sub>±0.013</sub> | 0.829<sub>±0.001</sub> | 0.853<sub>±0.011</sub> | 0.901<sub>±0.009</sub> | 0.802<sub>±0.001</sub> | 0.928<sub>±0.005</sub> |
|  | micro AUC | 0.910<sub>±0.002</sub> | 0.836<sub>±0.001</sub> | 0.847<sub>±0.006</sub> | 0.910<sub>±0.001</sub> | 0.812<sub>±0.002</sub> | 0.934<sub>±0.001</sub> |
|  | macro AUC | 0.910<sub>±0.002</sub> | 0.835<sub>±0.000</sub> | 0.848<sub>±0.007</sub> | 0.909<sub>±0.003</sub> | 0.812<sub>±0.000</sub> | 0.937<sub>±0.002</sub> |

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
