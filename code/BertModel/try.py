# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 17:32:41 2022

@author: Yifan Zhou
@email: yifanz6@andrew.cmu.edu
@github: https://github.com/Kudalf
"""
import json, os
import random
import argparse

import torch
import torch.nn.functional as F
from tqdm import trange

from data import load_data_instances, DataIterator
from model import MultiInferBert
import utils

parser = argparse.ArgumentParser()

parser.add_argument('--prefix', type=str, default="../../data/",
                    help='dataset and embedding path prefix')
parser.add_argument('--model_dir', type=str, default="savemodel/",
                    help='model path prefix')
parser.add_argument('--task', type=str, default="pair", choices=["pair", "triplet"],
                    help='option: pair, triplet')
parser.add_argument('--mode', type=str, default="train", choices=["train", "test"],
                    help='option: train, test')
parser.add_argument('--dataset', type=str, default="res14", choices=["res14", "lap14", "res15", "res16"],
                    help='dataset')
parser.add_argument('--max_sequence_len', type=int, default=100,
                    help='max length of a sentence')
parser.add_argument('--device', type=str, default="cpu",
                    help='gpu or cpu')

parser.add_argument('--bert_model_path', type=str,
                    default="pretrained/bert-base-uncased",
                    help='pretrained bert model path')
parser.add_argument('--bert_tokenizer_path', type=str,
                    default="bert-base-uncased",
                    help='pretrained bert tokenizer path')
parser.add_argument('--bert_feature_dim', type=int, default=768,
                    help='dimension of pretrained bert feature')

parser.add_argument('--nhops', type=int, default=1,
                    help='inference times')
parser.add_argument('--batch_size', type=int, default=32,
                    help='bathc size')
parser.add_argument('--epochs', type=int, default=100,
                    help='training epoch number')
parser.add_argument('--class_num', type=int, default=4,
                    help='label number')

args = parser.parse_args()
args.class_num = 6
args.task = "triplet"
# Load data
sentence_packs = json.load(open("F:/Learning/Deep Learning/Project/GTS-main/data/res14/test.json"))
instances = load_data_instances(sentence_packs, args)
instance = instances[1]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load model
model_path = "F:/Learning/Deep Learning/Project/GTS-main/code/BertModel/savemodel/berttriplet.pt"
model = torch.load(model_path, map_location=torch.device('cpu'), )
model.eval()

# Make a dataiterator
testset = DataIterator(instances, args)

# get one batch
sentence_ids, tokens, lengths, masks, sens_lens, token_ranges, aspect_tags, tags = testset.get_batch(1)
pred = model(tokens, masks)
preds = torch.argmax(pred, dim=3)

all_ids = []
all_preds = []
all_labels = []
all_lengths = []
all_sens_lengths = []
all_token_ranges = []

all_preds.append(preds)
all_labels.append(tags)
all_lengths.append(lengths)
all_sens_lengths.extend(sens_lens)
all_token_ranges.extend(token_ranges)
all_ids.extend(sentence_ids)

all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
all_labels = torch.cat(all_labels, dim=0).cpu().tolist()
all_lengths = torch.cat(all_lengths, dim=0).cpu().tolist()


metric = utils.Metric(args, all_preds, all_labels, all_lengths, all_sens_lengths, all_token_ranges, ignore_index=-1)
precision, recall, f1 = metric.score_uniontags()

aspect_results = metric.score_aspect()
opinion_results = metric.score_opinion()

num = 30
golden_set = set()
predicted_set = set()
for i in range(0, num):
    predicted_aspect_spans = metric.get_spans(metric.predictions[i], metric.sen_lengths[i], metric.tokens_ranges[i], 1)
    predicted_opinion_spans = metric.get_spans(metric.predictions[i], metric.sen_lengths[i], metric.tokens_ranges[i], 2)
    if metric.args.task == 'pair':
        predicted_tuples = metric.find_pair(metric.predictions[i], predicted_aspect_spans, predicted_opinion_spans, metric.tokens_ranges[i])
    elif metric.args.task == 'triplet':
        predicted_tuples = metric.find_triplet(metric.predictions[i], predicted_aspect_spans, predicted_opinion_spans, metric.tokens_ranges[i])
    
    for pair in predicted_tuples:
        predicted_set.add(str(i) + '-' + '-'.join(map(str, pair)))
    golden_aspect_spans = metric.get_spans(metric.goldens[i], metric.sen_lengths[i], metric.tokens_ranges[i], 1)
    golden_opinion_spans = metric.get_spans(metric.goldens[i], metric.sen_lengths[i], metric.tokens_ranges[i], 2)
    if metric.args.task == 'pair':
        golden_tuples = metric.find_pair(metric.goldens[i], golden_aspect_spans, golden_opinion_spans, metric.tokens_ranges[i])
    elif metric.args.task == 'triplet':
        golden_tuples = metric.find_triplet(metric.goldens[i], golden_aspect_spans, golden_opinion_spans, metric.tokens_ranges[i])
    for pair in golden_tuples:
        golden_set.add(str(i) + '-' + '-'.join(map(str, pair)))

for i in golden_set:
    print(i)
for i in range(32, 65):
    print(instances[i].sentence)


review = "The hot dogs are top notch but average coffee."
review_dict = {'id': '112221', 'sentence': review, 'triples': []}
pseudo_sentence_pack = [review_dict]
instances = load_data_instances(pseudo_sentence_pack, args)

# Make a dataiterator
testset = DataIterator(instances, args)


# get one batch
sentence_ids, tokens, lengths, masks, sens_lens, token_ranges, aspect_tags, tags = testset.get_batch(0)
pred = model(tokens, masks)
preds = torch.argmax(pred, dim=3)

all_ids = []
all_preds = []
all_labels = []
all_lengths = []
all_sens_lengths = []
all_token_ranges = []

all_preds.append(preds)
all_labels.append(tags)
all_lengths.append(lengths)
all_sens_lengths.extend(sens_lens)
all_token_ranges.extend(token_ranges)
all_ids.extend(sentence_ids)

all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
all_labels = torch.cat(all_labels, dim=0).cpu().tolist()
all_lengths = torch.cat(all_lengths, dim=0).cpu().tolist()


metric = utils.Metric(args, all_preds, all_labels, all_lengths, all_sens_lengths, all_token_ranges, ignore_index=-1)
precision, recall, f1 = metric.score_uniontags()

aspect_results = metric.score_aspect()
opinion_results = metric.score_opinion()

predicted_aspect_spans = metric.get_spans(metric.predictions[0], metric.sen_lengths[0], metric.tokens_ranges[0], 1)
predicted_opinion_spans = metric.get_spans(metric.predictions[0], metric.sen_lengths[0], metric.tokens_ranges[0], 2)
if metric.args.task == 'pair':
    predicted_tuples = metric.find_pair(metric.predictions[0], predicted_aspect_spans, predicted_opinion_spans, metric.tokens_ranges[0])
elif metric.args.task == 'triplet':
    predicted_tuples = metric.find_triplet(metric.predictions[0], predicted_aspect_spans, predicted_opinion_spans, metric.tokens_ranges[0])
