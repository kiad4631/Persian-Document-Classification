# Persian-Document-Classification
Currently, the classification subject of Persian articles is sparse, limited, and non-automatic.This repo proposes methods based on deep learning for extracting features of Persian texts, methods for classifying texts and finally identifying the subject matter of the collection. The Hamshahri dataset (a sub-body containing 166,000 documents with a specific theme), one of the most prestigious Persian language resources in the field of natural language, is applied for feature extraction and text classification.

1. [Fasttext](test_dir)
2. [Gensim](test_dir)
3. [Pytext](test_dir)

## Fasttext
[FastText](https://github.com/facebookresearch/fastText/) is a library for efficient learning of word representations and sentence classification.
At first we use pre-trained word vectors for _Persian_ language, trained on Common Crawl and Wikipedia. This model was trained using CBOW with position-weights, in dimension 300, with character n-grams of length 5, a window of size 5 and 10 negatives.




