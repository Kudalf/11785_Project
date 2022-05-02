# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 16:48:33 2022

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
import numpy as np
from transformers import BertTokenizer, RobertaTokenizer
from transformers import DistilBertTokenizer, DistilBertModel




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
#args.bert_tokenizer_path = "roberta-base"
#args.bert_model_path = "roberta-base"
bert_tokenizer_path = "distilbert-base-uncased"
bertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
robertaTokenizer = RobertaTokenizer.from_pretrained("roberta-base")



# Load data
sentence_packs = json.load(open("F:/Learning/Deep Learning/Project/GTS-main/data/res14/test.json"))
instances = load_data_instances(sentence_packs, args)
instance = instances[1]

bert_token_range = []
token_start = 1
for i, w, in enumerate(instance.tokens):
    token_end = token_start + len(bertTokenizer.encode(w, add_special_tokens=False))
    bert_token_range.append([token_start, token_end-1])
    token_start = token_end

roberta_token_range = []
token_start = 1
for i, w, in enumerate(instance.tokens):
    token_end = token_start + len(robertaTokenizer.encode(w, add_special_tokens=False))
    roberta_token_range.append([token_start, token_end-1])
    token_start = token_end
print(bertTokenizer(instance.sentence))
print(bert_token_range)
print(robertaTokenizer(instance.sentence))
print(roberta_token_range)

print(instance.sentence)

print(bertTokenizer("fastest"))
print(robertaTokenizer("fastest"))
print(robertaTokenizer("fast"))

print(robertaTokenizer("fastest"))
print(robertaTokenizer(instance.sentence))

robertaTokens = robertaTokenizer.encode(instance.sentence)
