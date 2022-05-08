import json, os
import random
import argparse

import torch
import torch.nn.functional as F
from tqdm import trange

from data import load_data_instances, DataIterator
from model import MultiInferBert
import utils
import copy
import random

sentence_packs = json.load(open("D:/Heinz/Deep Learning/Project/GTS-main/data/lap14/train.json"))
negative_list = []
positive_list = []
print(len(sentence_packs))

ntrip = 0
for i in range(0, len(sentence_packs)):
     ntrip = ntrip + len(sentence_packs[i]["triples"])

print(ntrip)

for i in range(0, len(sentence_packs)):
    for triplet in sentence_packs[i]['triples']:
        if triplet['sentiment'] == 'negative':
            negative_list.append(sentence_packs[i])
            break

for i in range(0, len(sentence_packs)):
    for triplet in sentence_packs[i]['triples']:
        if triplet['sentiment'] != 'negative':
            positive_list.append(sentence_packs[i])
            break

ready_to_use_list = [x for x in negative_list if x in positive_list]
negative_list = [x for x in negative_list if x not in ready_to_use_list]
positive_list = [x for x in positive_list if x not in ready_to_use_list]


length = len(negative_list)

print(len(negative_list), len(positive_list), len(ready_to_use_list))


new_sentence_packs = []

original_positive = copy.deepcopy(positive_list)
original_negative = copy.deepcopy(negative_list)

for i in range(0, length):
    original_left_sentence = copy.deepcopy(positive_list[i])

    positive_list[i]['id'] = positive_list[i]['id'] + '-0'

    positive_list[i]['sentence'] = positive_list[i]['sentence'] + ' ' + negative_list[i]['sentence']

    sentence_right = negative_list[i]['sentence'].split(" ")
    for j in range(len(sentence_right)):
        sentence_right[j] = sentence_right[j] + "\\O"
    sentence_right = " ".join(sentence_right)

    for triplet in positive_list[i]['triples']:
        triplet['target_tags'] = triplet['target_tags'] + ' ' + sentence_right
        triplet['opinion_tags'] = triplet['opinion_tags'] + ' ' + sentence_right

    sentence_left = original_left_sentence['sentence'].split(" ")
    for k in range(len(sentence_left)):
        sentence_left[k] = sentence_left[k] + "\\O"
    sentence_left = " ".join(sentence_left)

    for triplet in negative_list[i]['triples']:
        triplet['target_tags'] = sentence_left + ' ' + triplet['target_tags']
        triplet['opinion_tags'] = sentence_left + ' ' + triplet['opinion_tags']

    for triplet in negative_list[i]['triples']:
        positive_list[i]['triples'].append(triplet)

    new_sentence_packs.append(positive_list[i])

sliced_positive = original_positive[360:]

for i in range(0, len(sliced_positive)):
    original_left_sentence = copy.deepcopy(sliced_positive[i])

    sliced_positive[i]['id'] = sliced_positive[i]['id'] + '-0'

    sliced_positive[i]['sentence'] = sliced_positive[i]['sentence'] + ' ' + original_negative[i]['sentence']

    sentence_right = original_negative[i]['sentence'].split(" ")
    for j in range(len(sentence_right)):
        sentence_right[j] = sentence_right[j] + "\\O"
    sentence_right = " ".join(sentence_right)

    for triplet in sliced_positive[i]['triples']:
        triplet['target_tags'] = triplet['target_tags'] + ' ' + sentence_right
        triplet['opinion_tags'] = triplet['opinion_tags'] + ' ' + sentence_right

    sentence_left = original_left_sentence['sentence'].split(" ")
    for k in range(len(sentence_left)):
        sentence_left[k] = sentence_left[k] + "\\O"
    sentence_left = " ".join(sentence_left)

    for triplet in original_negative[i]['triples']:
        triplet['target_tags'] = sentence_left + ' ' + triplet['target_tags']
        triplet['opinion_tags'] = sentence_left + ' ' + triplet['opinion_tags']

    for triplet in original_negative[i]['triples']:
        sliced_positive[i]['triples'].append(triplet)

    new_sentence_packs.append(sliced_positive[i])

print(len(new_sentence_packs))

new_sentence_packs.extend(ready_to_use_list)

print(new_sentence_packs[-1])

old_sentence_packs = json.load(open("D:/Heinz/Deep Learning/Project/GTS-main/data/lap14/train.json"))

old_sentence_packs.extend(new_sentence_packs)

# full_sentence_packs = set(old_sentence_packs)

full_sentence_packs = list(old_sentence_packs)

print(len(full_sentence_packs))

random.shuffle(full_sentence_packs)

# with open('train_lap14_full_concatenated.json', 'a') as f:
#       json.dump(full_sentence_packs, f, indent=1)