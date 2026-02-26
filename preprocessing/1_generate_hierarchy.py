import json
from IPython import embed
import matplotlib.pyplot as plt
import nltk
from collections import defaultdict
from nltk.corpus import wordnet as wn

nltk.download('wordnet')

# This is a file of the WNID and human-readable labels for the 1K ImageNet classes
SYNSET_PATH = '/data/hierarchy_metadata/synsets.txt' 
JSON_PATH = '/data/hierarchy_metadata/pruned_hierarchy.json'

def offset_to_synset(wnid):
    pos = wnid[0]
    offset = int(wnid[1:])
    return wn.synset_from_pos_and_offset(pos, offset)

# oddities = ['crane.n.05',
#     'cardigan.n.02',
#     'cardigan.n.01',
#     'crane.n.04',
#     'maillot.n.02',
#     'maillot.n.01']

# oddities_rename = {
#         'crane.n.05': 'crane_bird',
#         'crane.n.04': 'crane_machine',
#         'cardigan.n.02': 'cardigan_sweater',
#         'cardigan.n.01': 'cardigan_herald',
#         'maillot.n.02': 'maillot_swimsuit',
#         'maillot.n.01': 'maillot_jersey'
#         }

def get_syn_label(syn):
    return syn.name()

# STEP 1: Load synsets with names and indices
data = {}
with open(SYNSET_PATH, 'r') as f:
    for idx, line in enumerate(f):
        if line.strip():
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                wnid, label = parts
                syn = offset_to_synset(wnid)

                leaf_label = get_syn_label(syn)
                hyp_paths = syn.hypernym_paths()
                for p in hyp_paths:
                    for h in p:
                        label = get_syn_label(h)

                        if label not in data:
                            data[label] = {'idxs': set(),
                                           'synonyms': set()}

                        data[label]['idxs'].add(idx)
                        if label == leaf_label:
                            data[label]['leaf'] = True
                        else:
                            data[label]['leaf'] = False

count = 0
for k,v in data.items():
    if v['leaf']:
        count+=1

print('There are {} original leaf nodes.'.format(count))
print('After processing, the hierarchy has {} nodes.'.format(len(data)))

# Recast sets as sorted lists
for k,v in data.items():
    data[k]['idxs'] = sorted(list(v['idxs']))

# Remove nodes with one parent and one child
all_synonyms = set()
for k,v in data.items():
    for _k, _v in data.items():
        if v['idxs'] == _v['idxs']:
            if k != _k:
                v['synonyms'].add(_k)
                all_synonyms.add(_k)

for synonym in list(all_synonyms):
    is_leaf = data[synonym]['leaf']
    if not is_leaf:
        del data[synonym]

# Verify no duplicates
count = 0
for k,v in data.items():
    for _k, _v in data.items():
        if _v['idxs'] == v['idxs']:
            count+=1

    # If v['synonym'] is empty, set to None
    if len(v['synonyms']) == 0:
        v['synonyms'] = None
    else:
        v['synonyms'] = sorted(list(v['synonyms']))

assert(len(data) == count)
print('Pruning nodes with one parent and one child reduced the hierarchy to {} nodes.'.format(len(data))) 

# Save out as a json
with open(JSON_PATH,'w') as f:
    json.dump(data, f, indent=2)

# Test loading back
with open(JSON_PATH, 'r') as f:
    test_data = json.load(f)

assert(test_data==data)


