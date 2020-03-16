![](https://github.com/Davari393/Persian-Document-Classification/blob/master/0_X7PVc7QwrpFnyo4p.png
)
# Persian-Document-Classification
The classification subject of Persian articles is sparse, limited, and non-automatic.This repo proposes methods based on deep learning for extracting features of Persian texts, methods for classifying texts and finally identifying the subject matter of the collection. The [Hamshahri](http://dataheart.ir/article/3487/%D9%85%D8%AC%D9%85%D9%88%D8%B9%D9%87-%D8%AF%D8%A7%D8%AF%D9%87--%DA%A9%D8%A7%D9%85%D9%84-%D9%87%D9%85%D8%B4%D9%87%D8%B1%DB%8C-%D9%86%D8%B3%D8%AE%D9%87-1-%D8%B4%D8%A7%D9%85%D9%84-166-%D9%87%D8%B2%D8%A7%D8%B1-%D8%B3%D9%86%D8%AF-%D8%AF%D8%B1-%D9%81%D8%B1%D9%85%D8%AA-%D8%A7%DA%A9%D8%B3%D9%84-%D9%88-csv) dataset (a sub-body containing 166,000 documents with a specific theme), one of the most prestigious Persian language resources in the field of natural language, is applied for feature extraction and text classification. These methods are:



1. Fasttext
2. Gensim
3. Pytext



## Fasttext

[FastText](https://github.com/facebookresearch/fastText/) is a library for efficient learning of word representations and sentence classification.

At first, we use [pre-trained](https://fasttext.cc/docs/en/crawl-vectors.html) word vectors for ***Persian*** language, trained on Common Crawl and Wikipedia. This model was trained using CBOW with position-weights, in dimension 300, with character n-grams of length 5, a window of size 5 and 10 negatives.

For more details and download pre-trained (bin/text) file of every language directly you can go here: https://fasttext.cc/docs/en/crawl-vectors.html


Also, this lines of python code (e.g. Persian/Farsi) works for you to download the bin file:



```python

import fasttext.util

fasttext.util.download_model('fa', if_exists='ignore')  # English

ft = fasttext.load_model('cc.fa.300.bin')

```



However, you need just a pre-trained word embedding text file to do this step. The pre-trained Persian embedding text file is ready [here](https://drive.google.com/open?id=1Zm7Hk4Il3WCcPRBYRhynWhqi1i_1h8w9).

By placing the downloaded embedding text file in the fasttext folder and running [_train_fasttext.py_](https://github.com/Davari393/Persian-Document-Classification/tree/master/fasttext) you are given an MLP text classification model (with the report of accuracy and error on the train and test data) with fasttext pre-trained Persian word embeddings.





## Gensim

This is an open-source python library for natural language processing. The [Gensim](https://github.com/RaRe-Technologies/gensim) library enables us to extend embedding by training our Word2vec model (Another word representation model like FastTesxt), using CBOW algorithms or skip-gram algorithm.



To train this model, first of all, the data should be convert to .txt file and then some [cleaning steps](https://github.com/Davari393/Persian-Document-Classification/tree/master/clean_data) be done on it. Then the final text file is given to the model and is trained. You can download pre-trained embeddings of the 166,000 documents hamshahri from [here](https://drive.google.com/open?id=1vmdgHgNje5r18VpZ2xf2cbdu5l_bfOXd) or train it and classifier with [_train_gensim.py_](https://github.com/Davari393/Persian-Document-Classification/tree/master/gensim) code.

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
And you should change it to this:
```
{
  "version": 8,
  "task": {
    "DocumentClassificationTask": {
      "data": {
        "source": {
          "TSVDataSource": {
            "field_names": ["text" , "label"],
            "train_filename": "/content/train.tsv",
            "test_filename": "/content/test.tsv",
            "eval_filename": "/content/eval.tsv"
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
All you have to do is preparing your dataset(train, test, eval) in __.tsv__ format.
For this purpose, some commands are defined below:
```
./app.sh file.csv file.tsv.txt
mv file.tsv.txt file.tsv
```

Also, the app.sh file has placed in this repo.

TSV format: it means the text and its label have to be seprated with _tab_ and except for this tab, there should be no tab in your texts.
If you want, you can change the config file as well. Then train the data, such as how described that in the pytext repo and enjoy it.

