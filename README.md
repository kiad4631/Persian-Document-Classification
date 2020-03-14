# Persian-Document-Classification
Currently, the classification subject of Persian articles is sparse, limited, and non-automatic.This repo proposes methods based on deep learning for extracting features of Persian texts, methods for classifying texts and finally identifying the subject matter of the collection. The Hamshahri dataset (a sub-body containing 166,000 documents with a specific theme), one of the most prestigious Persian language resources in the field of natural language, is applied for feature extraction and text classification.

1. [Fasttext](test_dir)
2. [Gensim](test_dir)
3. [Pytext](test_dir)

## Fasttext
[FastText](https://github.com/facebookresearch/fastText/) is a library for efficient learning of word representations and sentence classification.
At first we use [pre-trained](https://fasttext.cc/docs/en/crawl-vectors.html) word vectors for ***Persian*** language, trained on Common Crawl and Wikipedia. This model was trained using CBOW with position-weights, in dimension 300, with character n-grams of length 5, a window of size 5 and 10 negatives.
For more detailes and download pre-trained (bin/text) file of every language directly you can go here: https://fasttext.cc/docs/en/crawl-vectors.html

Also, this lines of python code (e.g. Persian/Farsi) works for you to download bin file:

```python
import fasttext.util
fasttext.util.download_model('fa', if_exists='ignore')  # English
ft = fasttext.load_model('cc.fa.300.bin')
```

However, you need just pre-trained text file to do this step.
By running _train.py_ you be given a MLP text classification model with pre-trained persian word embeddings.









