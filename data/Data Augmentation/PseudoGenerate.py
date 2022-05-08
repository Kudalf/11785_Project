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
parser.add_argument('--epochs', type=int, default=101,
                    help='training epoch number')
parser.add_argument('--class_num', type=int, default=4,
                    help='label number')

args = parser.parse_args()
args.class_num = 6
args.task = "triplet"

# Load model
model_path = "D:/Heinz/Deep Learning/Project/GTS-main/code/BertModel/savemodel/berttriplet.pt"
model = torch.load(model_path, map_location=torch.device('cpu'), )
model.eval()

review_list = []
# Parsing into Amazon
pseudo_sentence_packs = json.load(
    open("D:/Heinz/Deep Learning/Project/Data Cleaned/Electronics_clean_grouped_0501.json"))

print(len(pseudo_sentence_packs['clean_text_cat']))

# Generate the same format for the original dataset
for i in range(0, len(pseudo_sentence_packs['clean_text_cat'])):
    key = str(i)
    sentence_list = pseudo_sentence_packs['clean_text_cat'][key].split('.')
    for j in range(0, len(sentence_list)):
        # Select sentence that have length between 5 and 100
        if 5 < len(sentence_list[j].split(" ")) < 100:
            sentence = sentence_list[j]
            sentence_dict = {"id": "", "sentence": "", "triples": []}
            sentence_dict["id"] = key
            sentence_dict["sentence"] = sentence.strip()
            new_sentence = copy.deepcopy(sentence_dict)
            review_list.append(new_sentence)

print(len(review_list))

pseudo_sentence_pack = []

for k in range(50001, len(review_list)):
    pseudo_sentence = [review_list[k]]
    ## TODO: parts below are from the original model
    instances = load_data_instances(pseudo_sentence, args)
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
        predicted_tuples = metric.find_pair(metric.predictions[0], predicted_aspect_spans, predicted_opinion_spans,
                                            metric.tokens_ranges[0])
    elif metric.args.task == 'triplet':
        predicted_tuples = metric.find_triplet(metric.predictions[0], predicted_aspect_spans, predicted_opinion_spans,
                                               metric.tokens_ranges[0])
    ## TODO: parts above are from the original model

    # Use the predicted tuples to generate pseudo labels

    i = 0
    for triplet in predicted_tuples:
        sentence = {"uid": "", "target_tags": "",
                    "opinion_tags": "", "sentiment": ""}
        # uid
        sentence["uid"] = pseudo_sentence[0]['id'] + "-" + str(i)
        i += 1

        raw_labeled_sentence = pseudo_sentence[0]['sentence'].split(" ")
        for i in range(len(raw_labeled_sentence)):
            raw_labeled_sentence[i] = raw_labeled_sentence[i] + "\\O"

        # target_tags
        copy_sentence1 = copy.deepcopy(raw_labeled_sentence)
        copy_sentence1[triplet[0]] = copy_sentence1[triplet[0]][:-1] + "B"
        if triplet[0] != triplet[1]:
            for i in range(triplet[0] + 1, triplet[1] + 1):
                copy_sentence1[i] = copy_sentence1[i][:-1] + "I"
        copy_sentence1 = " ".join(copy_sentence1)
        sentence["target_tags"] = copy_sentence1

        # opinion_tags
        raw_labeled_sentence[triplet[2]] = raw_labeled_sentence[triplet[2]][:-1] + "B"
        if triplet[2] != triplet[3]:
            for i in range(triplet[2] + 1, triplet[3] + 1):
                raw_labeled_sentence[i] = raw_labeled_sentence[i][:-1] + "I"
        opinion_labeled_sentence = " ".join(raw_labeled_sentence)
        sentence["opinion_tags"] = opinion_labeled_sentence

        # sentiment tags
        if triplet[4] == 3:
            sentence["sentiment"] = ("negative")
        if triplet[4] == 4:
            sentence["sentiment"] = ("neutral")
        else:
            sentence["sentiment"] = ("positive")

        review_list[k]['triples'].append(sentence.copy())

    if k % 100 == 0:
        print(k)

    # if k == 1000:
    #     with open('train_amazon_1000.json', 'a') as f:
    #         json.dump(review_list, f, indent=1)

    # if k == 5000:
    #     with open('train_amazon_5000.json', 'a') as f:
    #         json.dump(review_list, f, indent=1)
    #
    if k == 10000:
         with open('train_amazon_10000_2.json', 'a') as f:
             json.dump(review_list, f, indent=1)

    if k == 50000:
        with open('train_amazon_50000.json', 'a') as f:
            json.dump(review_list, f, indent=1)

    if k == 75000:
        with open('train_amazon_75000.json', 'a') as f:
            json.dump(review_list, f, indent=1)

    if k == 90000:
        with open('train_amazon_90000.json', 'a') as f:
            json.dump(review_list, f, indent=1)

    if k == 100000:
        with open('train_amazon_100000.json', 'a') as f:
            json.dump(review_list, f, indent=1)

    if k == len(review_list) - 1:
        with open('train_amazon_150000.json', 'a') as f:
            json.dump(review_list, f, indent=1)

print(review_list[6])
