import pandas as pd
import torch as th


def get_data(df, iprs_dict, terms_dict):
    data = th.zeros((len(df), len(iprs_dict)), dtype=th.float32)
    labels = th.zeros((len(df), len(terms_dict)), dtype=th.float32)
    for i, row in enumerate(df.itertuples()):
        for ipr in row.interpros:
            if ipr in iprs_dict:
                data[i, iprs_dict[ipr]] = 1
        for go_id in row.prop_annotations:
            if go_id in terms_dict:
                labels[i, terms_dict[go_id]] = 1
    return data, labels


def load_data(data_root, ont, terms_file):
    terms = pd.read_pickle(terms_file)['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    print('Terms', len(terms))

    iprs = pd.read_pickle(f'{data_root}/{ont}/interpros.pkl')['interpros'].values
    iprs_dict = {v: k for k, v in enumerate(iprs)}

    train_df = pd.read_pickle(f'{data_root}/{ont}/train_data.pkl')
    valid_df = pd.read_pickle(f'{data_root}/{ont}/valid_data.pkl')
    test_df = pd.read_pickle(f'{data_root}/{ont}/test_data.pkl')

    train_data = get_data(train_df, iprs_dict, terms_dict)
    valid_data = get_data(valid_df, iprs_dict, terms_dict)
    test_data = get_data(test_df, iprs_dict, terms_dict)

    return iprs_dict, terms_dict, train_data, valid_data, test_data, test_df


def propagate_annots(scores, go, terms_dict):
    prop_annots = {}
    for go_id, j in terms_dict.items():
        score = scores[j]
        for sup_go in go.get_anchestors(go_id):
            if sup_go in prop_annots:
                prop_annots[sup_go] = max(prop_annots[sup_go], score)
            else:
                prop_annots[sup_go] = score
    for go_id, score in prop_annots.items():
        if go_id in terms_dict:
            scores[terms_dict[go_id]] = score
    return scores


def load_normal_forms(go_file, terms_dict):
    buckets = {k: [] for k in ('gci0', 'gci1', 'gci1_bot', 'gci2', 'gci3')}
    relations = {}
    zclasses = {}

    def cls_index(go_id):
        if go_id in terms_dict:
            return terms_dict[go_id]
        if go_id not in zclasses:
            zclasses[go_id] = len(terms_dict) + len(zclasses)
        return zclasses[go_id]

    def rel_index(rel_id):
        if rel_id not in relations:
            relations[rel_id] = len(relations)
        return relations[rel_id]

    with open(go_file) as f:
        for line in f:
            s = line.strip().replace('_', ':')
            if 'SubClassOf' not in s:
                continue
            left, right = (x.strip() for x in s.split(' SubClassOf '))

            if ' and ' in left:
                a, b = (x.strip() for x in left.split(' and '))
                if right in ('Nothing', 'owl:Nothing'):
                    buckets['gci1_bot'].append((cls_index(a), cls_index(b)))
                else:
                    buckets['gci1'].append((cls_index(a), cls_index(b), cls_index(right)))
            elif ' some ' in left:
                rel, c = (x.strip() for x in left.split(' some '))
                buckets['gci3'].append((rel_index(rel), cls_index(c), cls_index(right)))
            elif ' some ' in right:
                rel, d = (x.strip() for x in right.split(' some '))
                buckets['gci2'].append((cls_index(left), rel_index(rel), cls_index(d)))
            else:
                buckets['gci0'].append((cls_index(left), cls_index(right)))

    buckets['relations'] = relations
    buckets['zclasses'] = zclasses
    return buckets


def buckets_to_tensors(buckets, device):
    def to_tensor(bucket, ncols):
        if not bucket:
            return th.empty(0, ncols, dtype=th.long).to(device)
        return th.LongTensor(bucket).to(device)

    return (to_tensor(buckets['gci0'], 2),
            to_tensor(buckets['gci1'], 3),
            to_tensor(buckets['gci1_bot'], 2),
            to_tensor(buckets['gci3'], 3),
            to_tensor(buckets['gci2'], 3))
