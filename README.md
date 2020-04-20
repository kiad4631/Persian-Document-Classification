![](https://github.com/Davari393/Persian-Document-Classification/blob/master/0_X7PVc7QwrpFnyo4p.png
)
# Persian Document Classification
The classification subject of Persian articles is sparse, limited, and non-automatic.This repo proposes methods based on deep learning for extracting features of Persian texts, methods for classifying texts and finally identifying the subject matter of the collection.

## Table of contents
* Demo
* Dataset
* Methods
  * Fasttext
  * Gensim
  * PyText



## Dataset
The [Hamshahri](http://dataheart.ir/article/3487/%D9%85%D8%AC%D9%85%D9%88%D8%B9%D9%87-%D8%AF%D8%A7%D8%AF%D9%87--%DA%A9%D8%A7%D9%85%D9%84-%D9%87%D9%85%D8%B4%D9%87%D8%B1%DB%8C-%D9%86%D8%B3%D8%AE%D9%87-1-%D8%B4%D8%A7%D9%85%D9%84-166-%D9%87%D8%B2%D8%A7%D8%B1-%D8%B3%D9%86%D8%AF-%D8%AF%D8%B1-%D9%81%D8%B1%D9%85%D8%AA-%D8%A7%DA%A9%D8%B3%D9%84-%D9%88-csv) dataset (a sub-body containing 166,000 documents with a specific theme), one of the most prestigious Persian language resources in the field of natural language, has been applied for feature extraction and text classification. 



## Methods 
1. Fasttext
2. Gensim
3. Pytext



## Fasttext

[FastText](https://github.com/facebookresearch/fastText/) is a library for efficient learning of word representations and sentence classification.

At first, we use [pre-trained](https://fasttext.cc/docs/en/crawl-vectors.html) word vectors for ***Persian*** language, trained on Common Crawl and Wikipedia. This model was trained using CBOW with position-weights, in dimension 300, with character n-grams of length 5, a window of size 5 and 10 negatives.

For more details and download pre-trained (bin/text) file of other languages directly you can go here: https://fasttext.cc/docs/en/crawl-vectors.html


Also, this lines of python code (e.g. Persian/Farsi) works for you to download the bin file:



```python

import fasttext.util

fasttext.util.download_model('fa', if_exists='ignore')  # English

ft = fasttext.load_model('cc.fa.300.bin')

```



However, you need just a pre-trained word embedding text file to do this step. The pre-trained Persian embedding text file is ready [here](https://drive.google.com/open?id=1Zm7Hk4Il3WCcPRBYRhynWhqi1i_1h8w9).

By placing the downloaded embedding text file in the fasttext folder and running [_fasttext_classifier.py_](https://github.com/Davari393/Persian-Document-Classification/tree/master/fasttext) you are given an MLP text classification model (with the report of accuracy and error on the train and test data) with fasttext pre-trained Persian word embeddings.



## Gensim

This is an open-source python library for natural language processing. The [Gensim](https://github.com/RaRe-Technologies/gensim) library enables us to extend embedding by training our Word2vec model (Another word representation model like FastTesxt), using CBOW algorithms or skip-gram algorithm.



To train this model, first of all, the data should be convert to .txt file and some cleaning steps be done on it [here](https://github.com/Davari393/Persian-Document-Classification/tree/master/clean_data). Then the final text file is given to the model and is trained. You can download pre-trained embeddings of the 166,000 documents hamshahri from [here](https://drive.google.com/open?id=1vmdgHgNje5r18VpZ2xf2cbdu5l_bfOXd) or train it and classifier with [_train_gensim&classifier.py_](https://github.com/Davari393/Persian-Document-Classification/tree/master/gensim) code.

## Pytext
PyText is a deep-learning based NLP modeling framework built on PyTorch.
For applying this framwork the first step is cloning the pytext repository: https://github.com/facebookresearch/pytext

As you can see in __"Train your first text classifier"__ section of this repo, there is a _docnn.json_ file that is required for training.

This json file is like this:
```
{
  "version": 8,
  "task": {
    "DocumentClassificationTask": {
      "data": {
        "source": {
          "TSVDataSource": {
            "field_names": ["label", "slots", "text"],
            "train_filename": "tests/data/train_data_tiny.tsv",
            "test_filename": "tests/data/test_data_tiny.tsv",
            "eval_filename": "tests/data/test_data_tiny.tsv"
          }
        }
      },
      "model": {
        "DocModel": {
          "representation": {
            "DocNNRepresentation": {}
          }
        }
      }
    }
  },
  "export_torchscript_path": "/tmp/new_docnn.pt1",
  "export_caffe2_path": "/tmp/model.caffe2.predictor"
}
```
But the config of this project is it:
```
{
  "task": {
    
    "DocClassificationTask": {
      "trainer": {
        "epochs": 10,
        "random_seed": 0,
        "early_stop_after": 0,
        "max_clip_norm": null,
        "report_train_metrics": true
      },
      "data_handler": {
        "columns_to_read": ["text", "doc_label"],
        "eval_batch_size": 1,
        "max_seq_len": -1,
        "shuffle": true,
        "sort_within_batch": true,
        "test_batch_size": 1,
        "train_batch_size": 1,
        "train_path": "/content/train.tsv",
        "eval_path": "/content/eval.tsv",
        "test_path": "/content/test.tsv"
      }
    }
  }
}
```
All you have to do is preparing your dataset(train, test, eval) in __.tsv__ format.
For this purpose, some commands are defined below:
```
./app.sh file.csv file.tsv.txt
mv file.tsv.txt file.tsv
```

Also, the app.sh file has placed in this repo.

TSV format: it means the text and its label have to be seprated with _tab_ and except for this tab, there should be no tab in your texts.
If you want, you can change the config file as well. Then train the data, such as how described that in the pytext repo and enjoy it.

## Citation
Please cite our paper in your publications if it helps your research:
```
@inproceedings{li2019exposing,
  title={Persian Document Classification Using Deep Learning Methods},
  author= Nafiseh Davari, Mahya Mahdian Tabrizi, Alireza Akhavanpour, and Negin Daneshpour},
  booktitle={28th Iranian Conference on Electrical Engineering (ICEE2020)},
  year={2019}
}
```
