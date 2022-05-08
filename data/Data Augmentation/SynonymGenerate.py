# pip install numpy requests nlpaug

import json, os
import nlpaug.augmenter.word as naw
import random
#import nltk

#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')

sentences_pack = json.load(open("D:/Heinz/Deep Learning/Project/GTS-main/code/BertModel/train_lap14_full_concatenated.json"))
aug = naw.SynonymAug(aug_src='wordnet', aug_max=5)
augmented_list = []

for item in sentences_pack:
    augmented = aug.augment(item["sentence"])
    if len(augmented.split(" ")) != len(item["sentence"].split(" ")):
        pass
    else:
        item["sentence"] = augmented
        augmented_list.append(item)

print(len(augmented_list))

# with open('train_lap14_synonym.json', 'a') as f:
       # json.dump(augmented_list, f, indent=1)

sentences_pack = json.load(open("D:/Heinz/Deep Learning/Project/GTS-main/code/BertModel/train_lap14_full_concatenated.json"))
sentences_pack_2 = json.load(open("D:/Heinz/Deep Learning/Project/GTS-main/code/BertModel/train_lap14_synonym.json"))
full_dataset = sentences_pack + sentences_pack_2
print(len(full_dataset))
random.shuffle(full_dataset)
with open('train_lap14_full_synonym.json', 'a') as f:
    json.dump(full_dataset, f, indent=1)