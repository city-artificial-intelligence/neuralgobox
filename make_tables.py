import io
import re
import sys
from datetime import datetime

import numpy as np

ARGS = [a for a in sys.argv[1:] if not a.startswith('--')]
LOGS = ARGS[0] if ARGS else 'logs'
MARKDOWN = '--markdown' in sys.argv
if MARKDOWN:
    sys.stdout = io.StringIO()
ONTS = ('mf', 'bp', 'cc')
SEEDS = (0, 1, 2)
CRITERIA = ('loss', 'micro', 'macro')
ARMS = {'MLP': 'mlp', 'DeepGOZero': 'deepgozero', 'MLPBox': 'mlpbox',
        'NeuralGOBox': 'neuralgobox'}
ZS_LOGS = {arm: tuple(f'{name}_zs_s{s}' for s in SEEDS)
           for arm, name in (('DeepGOZero', 'deepgozero'), ('NeuralGOBox', 'neuralgobox'))}

PAPER_TABLE3 = {
    'MLP':         {'Fmax': (.656, .463, .662), 'Smin': (9.760, 44.185, 10.632), 'AUPR': (.630, .434, .643), 'AUC': (.900, .814, .865)},
    'MLP+D':       {'Fmax': (.668, .485, .665), 'Smin': (9.581, 43.945, 10.587), 'AUPR': (.624, .444, .634), 'AUC': (.902, .816, .868)},
    'DeepGOZero':  {'Fmax': (.655, .456, .660), 'Smin': (9.795, 44.628, 10.647), 'AUPR': (.662, .423, .647), 'AUC': (.905, .814, .860)},
    'DeepGOZero+D': {'Fmax': (.665, .484, .665), 'Smin': (9.599, 44.132, 10.579), 'AUPR': (.676, .440, .656), 'AUC': (.908, .816, .864)},
    'MLPBox':      {'Fmax': (.654, .461, .658), 'Smin': (9.821, 44.480, 10.726), 'AUPR': (.616, .430, .633), 'AUC': (.891, .804, .859)},
    'MLPBox+D':    {'Fmax': (.668, .485, .665), 'Smin': (9.613, 44.040, 10.631), 'AUPR': (.617, .439, .624), 'AUC': (.894, .805, .862)},
    'NeuralGOBox': {'Fmax': (.653, .460, .661), 'Smin': (9.844, 44.257, 10.657), 'AUPR': (.645, .430, .641), 'AUC': (.924, .829, .880)},
    'NeuralGOBox+D': {'Fmax': (.668, .485, .666), 'Smin': (9.604, 43.943, 10.590), 'AUPR': (.637, .439, .630), 'AUC': (.926, .831, .882)},
}
PAPER_TABLE4 = {
    'MLP':         {'loss': (.880, .784, .843), 'micro': (.884, .798, .861), 'macro': (.900, .814, .865)},
    'DeepGOZero':  {'loss': (.905, .800, .849), 'micro': (.905, .804, .857), 'macro': (.905, .814, .860)},
    'MLPBox':      {'loss': (.886, .785, .843), 'micro': (.889, .796, .857), 'macro': (.891, .804, .859)},
    'NeuralGOBox': {'loss': (.924, .808, .866), 'micro': (.924, .819, .879), 'macro': (.924, .829, .880)},
}
PAPER_TABLE5 = {'DeepGOZero': (.812, .734, .823), 'NeuralGOBox': (.910, .835, .848)}
PAPER_TABLE5_DEFINED = {'DeepGOZero': (.876, .788, .928), 'NeuralGOBox': (.909, .812, .937)}
PAPER_ZS_CLASSES = {'all': (4791, 11092, 1492), 'defined': (95, 4598, 151)}
PAPER_DGZ_ZS_LOSS = (.814, .735, .818)
PAPER_TRAIN_MINUTES = {'NeuralGOBox': (34, 69, 44), 'DeepGOZero': (107, 250, 129)}

FLOAT = r'([-0-9.]+)'
results = []


def check(name, got, expected, fmt='{:.3f}'):
    ok = fmt.format(got) == fmt.format(expected)
    results.append(ok)
    print(f'  {"OK " if ok else "XX "} {name:<40} logs {fmt.format(got):>8}   paper {fmt.format(expected):>8}')


def log_file(prefix, ont, seed):
    return open(f'{LOGS}/supervised/{ont}/{prefix}_{ont}_s{seed}.log').read()


def parse_eval(text, prefix, ont, seed, crit):
    block = text.split(f'=== evaluation of {prefix}_{ont}_s{seed}_sel-{crit} ')[1]
    block = block.split('=== evaluation of')[0]
    fmax = re.findall(r'Fmax: ' + FLOAT + ', Smin: ' + FLOAT, block)
    aupr = re.findall(r'\nAUPR: ' + FLOAT, block)
    auc = re.findall(r'Average AUC for \w+ ' + FLOAT, block)
    return {'Fmax': float(fmax[0][0]), 'Smin': float(fmax[0][1]), 'AUPR': float(aupr[0]),
            'AUC': float(auc[0]),
            'Fmax+D': float(fmax[1][0]), 'Smin+D': float(fmax[1][1]), 'AUPR+D': float(aupr[1]),
            'AUC+D': float(auc[1])}


def selected_epochs(text):
    sel, epoch = {}, None
    for line in text.split('\n'):
        m = re.match(r'Epoch (\d+):', line)
        if m:
            epoch = int(m.group(1))
        m = re.match(r'Saving model \(sel-(\w+)\)', line)
        if m:
            sel[m.group(1)] = epoch
    return sel


def train_minutes(text):
    start = re.search(r'=== start=(\S+)', text).group(1)
    end = re.search(r'=== evaluation of \S+ start=(\S+)', text).group(1)
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 60


sup = {}
for arm, prefix in ARMS.items():
    for ont in ONTS:
        for seed in SEEDS:
            text = log_file(prefix, ont, seed)
            sup[arm, ont, seed] = {
                'eval': {c: parse_eval(text, prefix, ont, seed, c) for c in CRITERIA},
                'epochs': selected_epochs(text), 'minutes': train_minutes(text)}


def mean(arm, ont, crit, metric):
    return float(np.mean([sup[arm, ont, s]['eval'][crit][metric] for s in SEEDS]))


print('Table 3 (macro AUC selection, three-seed means)')
for row, metrics in PAPER_TABLE3.items():
    arm, plus = row.removesuffix('+D'), '+D' if row.endswith('+D') else ''
    for metric, expected in metrics.items():
        for ont, e in zip(ONTS, expected):
            check(f'{row} {metric} {ont}', mean(arm, ont, 'macro', metric + plus), e)

print('\nTable 4 (class-centric AUC per selection criterion)')
for arm, by_crit in PAPER_TABLE4.items():
    for crit, expected in by_crit.items():
        for ont, e in zip(ONTS, expected):
            check(f'{arm} {crit} {ont}', mean(arm, ont, crit, 'AUC'), e)

print('\nSeed variation of class-centric AUC (Table 3)')
sds = [np.std([sup[a, o, s]['eval']['macro'][m] for s in SEEDS], ddof=1)
       for a in ARMS for o in ONTS for m in ('AUC', 'AUC+D')]
check('largest standard deviation over seeds', max(sds), 0.002)

print('\nMLP selected epochs (1-based)')
for crit, (lo, hi) in (('loss', (2, 4)), ('macro', (28, 44))):
    ep = [sup['MLP', o, s]['epochs'][crit] + 1 for o in ONTS for s in SEEDS]
    check(f'{crit}: earliest', min(ep), lo, '{:d}')
    check(f'{crit}: latest', max(ep), hi, '{:d}')

print('\nTraining time, minutes (mean over seeds)')
for arm, expected in PAPER_TRAIN_MINUTES.items():
    for ont, e in zip(ONTS, expected):
        check(f'{arm} {ont}', float(np.mean([sup[arm, ont, s]['minutes'] for s in SEEDS])), e, '{:.0f}')


def parse_zero_shot(name):
    text = open(f'{LOGS}/zeroshot/eval/{name}.log').read()
    out = {}
    for m in re.finditer(r'-ont (\w+) .*?_sel-(\w+)\nall classes: +n=(\d+) AUC=' + FLOAT +
                         r'\ndefined classes: +n=(\d+) AUC=' + FLOAT, text):
        ont, crit, n_all, auc_all, n_def, auc_def = m.groups()
        out[ont, crit] = {'all': float(auc_all), 'n_all': int(n_all),
                          'defined': float(auc_def), 'n_defined': int(n_def)}
    return out


zs = {arm: [parse_zero_shot(n) for n in names] for arm, names in ZS_LOGS.items()}


def zs_mean(arm, ont, crit, subset):
    return float(np.mean([z[ont, crit][subset] for z in zs[arm]]))


print('\nTable 5 (zero-shot, macro selection, three-seed means)')
for subset, paper in (('all', PAPER_TABLE5), ('defined', PAPER_TABLE5_DEFINED)):
    for arm, expected in paper.items():
        for ont, e in zip(ONTS, expected):
            check(f'{arm} {subset} {ont}', zs_mean(arm, ont, 'macro', subset), e)

print('\nZero-shot class counts')
for subset, expected in PAPER_ZS_CLASSES.items():
    for ont, e in zip(ONTS, expected):
        counts = {z[ont, c][f'n_{subset}'] for zz in zs.values() for z in zz for c in CRITERIA}
        ok = counts == {e}
        results.append(ok)
        print(f'  {"OK " if ok else "XX "} {subset + " " + ont:<40} logs {sorted(counts)}   paper {e}')

print('\nDeepGOZero zero-shot under cross-entropy selection (reproduction of DeepGOZero Table 9)')
for ont, e in zip(ONTS, PAPER_DGZ_ZS_LOSS):
    check(f'mean {ont}', zs_mean('DeepGOZero', ont, 'loss', 'all'), e)
mf = [z['mf', 'loss']['all'] for z in zs['DeepGOZero']]
check('MFO lowest seed', min(mf), 0.803)
check('MFO highest seed', max(mf), 0.830)
for ont in ('bp', 'cc'):
    v = [z[ont, 'loss']['all'] for z in zs['DeepGOZero']]
    ok = round(max(v) - min(v), 3) <= 0.002
    results.append(ok)
    print(f'  {"OK " if ok else "XX "} {ont + " range over seeds":<40} logs {max(v) - min(v):>8.3f}   paper   <= 0.002')

print('\nZero-shot: NeuralGOBox ahead of DeepGOZero in every cell and criterion')
for subset in ('all', 'defined'):
    for crit in CRITERIA:
        for ont in ONTS:
            b = [z[ont, crit][subset] for z in zs['NeuralGOBox']]
            d = [z[ont, crit][subset] for z in zs['DeepGOZero']]
            gap = np.mean(b) - np.mean(d)
            sd = max(np.std(b, ddof=1), np.std(d, ddof=1))
            ok = gap > sd
            results.append(ok)
            print(f'  {"OK " if ok else "XX "} {subset:<7} {crit:<6} {ont}: gap {gap:+.3f}, largest seed SD {sd:.3f}')

print(f'\n{sum(results)} of {len(results)} checks match')


DIAMONDSCORE = {'Fmax': (.623, .444, .581), 'Smin': (10.145, 45.040, 11.092),
                'AUPR': (.380, .313, .352), 'AUC': (.749, .610, .652)}
CRIT_NAME = {'loss': 'validation cross-entropy', 'micro': 'validation micro AUC',
             'macro': 'validation macro AUC'}


def ms(values):
    return f'{np.mean(values):.3f} ± {np.std(values, ddof=1):.3f}'


def supervised_table(ont):
    i = ONTS.index(ont)
    rows = ['| Method | Selection | Fmax | Smin | AUPR | AUC | Epochs |', '|' + '---|' * 7,
            '| DiamondScore | | ' + ' | '.join(f'{DIAMONDSCORE[m][i]:.3f}'
                                                for m in ('Fmax', 'Smin', 'AUPR', 'AUC')) + ' | |']
    for arm in ARMS:
        for j, crit in enumerate(CRITERIA):
            sel = CRIT_NAME[crit].removeprefix('validation ')
            epochs = '/'.join(str(sup[arm, ont, s]['epochs'][crit] + 1) for s in SEEDS)
            for plus in ('', '+D'):
                name = arm if j == 0 and not plus else ''
                cells = [ms([sup[arm, ont, s]['eval'][crit][m + plus] for s in SEEDS])
                         for m in ('Fmax', 'Smin', 'AUPR', 'AUC')]
                label = sel + (' + D' if plus else '')
                rows.append(f'| {name} | {label} | ' + ' | '.join(cells) + f' | {"" if plus else epochs} |')
    return '\n'.join(rows)


def markdown():
    out = []
    for ont, name in zip(ONTS, ('MFO', 'BPO', 'CCO')):
        out += [f'**Supervised results, {name}.**', '', supervised_table(ont), '']
    out += ['**Table 5, zero-shot class-centric AUC under each selection criterion, '
            'on all held-out classes and on those GO defines by an equivalence.**', '',
            '| Method | Selection | All MFO | All BPO | All CCO | Defined MFO | Defined BPO | Defined CCO |',
            '|' + '---|' * 8]
    for arm in zs:
        for i, c in enumerate(CRITERIA):
            name = arm if i == 0 else ''
            cells = [ms([z[o, c][subset] for z in zs[arm]])
                     for subset in ('all', 'defined') for o in ONTS]
            out.append(f'| {name} | {CRIT_NAME[c].removeprefix("validation ")} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)


if MARKDOWN:
    sys.stdout = sys.__stdout__
    print(markdown())
