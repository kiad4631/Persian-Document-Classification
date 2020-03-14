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



However, you need just a pre-trained word embedding text file to do this step.

By placing the downloaded embedding text file in the fasttext folder and running [_train_fasttext.py_](https://github.com/Davari393/Persian-Document-Classification/tree/master/fasttext) you are given an MLP text classification model (with the report of accuracy and error on the train and test data) with fasttext pre-trained Persian word embeddings.





## Gensim

This is an open-source python library for natural language processing. The [Gensim](https://github.com/RaRe-Technologies/gensim) library enables us to extend embedding by training our Word2vec model (Another word representation model like FastTesxt), using CBOW algorithms or skip-gram algorithm.



To train this model, first of all, the data should be convert to .txt file and then some [cleaning steps](https://github.com/Davari393/Persian-Document-Classification/tree/master/clean_data) be done on it. Then the final text file is given to the model and is trained. ([_train_gensim.py_](https://github.com/Davari393/Persian-Document-Classification/tree/master/gensim))






