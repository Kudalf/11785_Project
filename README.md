# Improved GTS model and application on Amazon Review Data
Information Extraction from text is extremely useful for decision-making based on large datasets such as product reviews. Aspect-oriented Fine-grained Opinion Extraction (AFOE) aims to automatically extract opinion pairs (aspect term, opinion term) or opinion triplets (aspect term, opinion term, sentiment) from review text. While pipeline approaches may suffer from error propagation and inconvenience in real-world scenarios, the combination of a pre-trained encoder with a Grid Tagging decoder can turn this work into a unified and generalized task. The GTS model is first proposed by this paper: [Grid Tagging Scheme for Aspect-oriented Fine-grained Opinion Extraction](https://arxiv.org/pdf/2010.04640.pdf). Zhen Wu, Chengcan Ying, Fei Zhao, Zhifang Fan, Xinyu Dai, Rui Xia. In Findings of EMNLP, 2020. To further reduce the GTS model’s error on triplet extraction, we have performed error analysis and experimented with different encoders and data augmentation techniques, which improved the F1 score by 6.

![image](https://user-images.githubusercontent.com/40879931/167280162-efbd7ace-70bc-400f-9011-077fb31b1d4a.png)



## Data
### Lap 14 
Target-opinion-sentiment triplet datasets are completely from alignments of this paper [TOWE](https://www.aclweb.org/anthology/N19-1259/). It contains English sentences extracted from laptop custumer reviews. Human annotators tagged the aspect terms (SB1) and their polarities (SB2); 899 sentences and 1264 triplets were
used for training and 332 for testing (evaluation).

### Amazon Data
[Amazon review dataset](https://nijianmo.github.io/amazon/index.html) is an unlabeled dataset which contains product reviews and metadata from Amazon, including 233.1 million reviews spanning May 1996 - Oct 2018. This dataset includes reviews (ratings, text, helpfulness votes), product metadata (descriptions, category information, price, brand, and image features), and links (also viewed/also bought graphs). We've selected the subcategory "Electronics" for our data augmentation and further application, including 5000 sentences.

### Data Augmentation Files
There are three py files in the "data/Data Augmentation" file. Each is for one of our three data augmentation methods.

For 'Self-training', we used the amazon dataset and generated pseudo label for it then output it in json format.

For 'Synonym Replacement', we used the nlpaug package to randomly replace some words in the sentence with their synonyms.

For 'Sentence Concatenation', we concatenated the sentences with opposite sentiment (the labels are also concatenated).

Self-training method is designed for the format of amazon dataset. Synonym Replacement and Sentence Concatenation' are designed for the lap14 dataset.

## Requirements
See requirement.txt or Pipfile for details
* pytorch==1.7.1
* transformers==3.4.0
* python=3.6

## Usage
- ### Training
The training process is included in the file code/BertModel/Train Project Model.ipynb.
```
python main.py --task triplet --mode train --dataset lap14_concat_syn --bert_model_path bert-base-uncased --bert_tokenizer_path bert-base-uncased --batch_size 8
```
Arguments
- dataset
	- lap14: laptop reviews
 	- res14: restaurant reviews
 	- res15: restaurant reviews
 	- res16: restaurant reviews
 	- neg_pos_concat: lap14 data with positve and negative sentences concatenated
 	- lap14_concat_syn: lap14 data with positve and negative sentences concatenated and synonym replacement
 	- amazon_lap14_full_synonym: lap14 data with positve and negative sentences concatenated and synonym replacement + pseudo labels generated from Amazon data
- bert_model_path and bert_tokenizer_path: Encoders and its correspoding tokenizer
 	- roberta-base: [RoBERTa](https://huggingface.co/roberta-base)
 	- bert-base-uncased: [BERT](https://huggingface.co/bert-base-uncased)
 	- vinai/bertweet-base: [BERTweet](https://huggingface.co/vinai/bertweet-base)
The best model will be saved in the folder "savemodel/".

- ### Error Analysis
The error analysis code is in the file code/BertModel/Error_Analysis.py.

# Results

<table>
	<tr>
	    <th colspan="1">Models</th>
        <th colspan="1">Dataset</th>
	    <th colspan="1">Precision</th>
	    <th colspan="1">Recall</th>
      <th colspan="1">F1</th> 
	</tr >
	<tr >
	    <td>GTS-BERT (baseline)</td>
	    <td>lap14</td>
	    <td>57.52</td>
      <td>51.92</td>
      <td>54.58</td>
	</tr>
    <tr >
	    <td>GTS-BERTweet</td>
	    <td>lap14</td>
	    <td>57.66</td>
      <td>57.98</td>
      <td>57.82</td>
	</tr>
  <tr >
	    <td>GTS-BERT + Augmented Data </td>
	    <td>lap14</td>
	    <td>62.12</td>
      <td>53.95</td>
      <td>57.75</td>
	</tr>
   <tr >
	    <td>GTS-RoBERTa </td>
	    <td>lap14</td>
	    <td> 59.51</td>
      <td>62.57</td>
      <td>61.00</td>
	</tr>
     <tr >
	    <td><b>GTS-RoBERTa + Augmented Data</b></td>
	    <td>lap14</td>
	    <td> 61.06</td>
      <td> 59.27</td>
      <td><b>61.15</b></td>
	</tr>
</table>

    
```

## Reference
[1]. Zhen Wu, Chengcan Ying, Fei Zhao, Zhifang Fan, Xinyu Dai, Rui Xia. [Grid Tagging Scheme for Aspect-oriented Fine-grained Opinion Extraction](https://arxiv.org/pdf/2010.04640.pdf). In Findings of EMNLP, 2020.

[2]. Zhifang Fan, Zhen Wu, Xin-Yu Dai, Shujian Huang, Jiajun Chen. [Target-oriented Opinion Words Extraction with Target-fused Neural Sequence Labeling](https://www.aclweb.org/anthology/N19-1259.pdf). In Proceedings of NAACL, 2019.
