# Neural Machine Translation

Pytorch implementation of Neural Machine Translation with seq2seq and attention (en-zh) (英汉翻译)


## Introduction 

The goal of machine translation is to maximize p(**y**|**x**). Due to the infinite space of language, directly estimating this conditional probability is impossible. Thus neural networks, which are good at fitting complex functions, are introduced into machine translation. 

[Sutskever et. al 2014](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) proposed a model consisting of encoder and decoder, named `seq2seq`. Once proposed, it set off a tornado in NMT. A number of follow-up work began (e.g. [Cho et. al](https://arxiv.org/abs/1406.1078)). One of the most famous work is the `attention` mechanism ([Bahdanau et. al 2014](https://arxiv.org/abs/1409.0473)).

In this repo, I implemented the seq2seq model with attention in PyTorch for en-zh translation. 

## Requirements

python 3.6

- PyTorch>=0.4
- torchtext
- nltk
- jieba
- subword-nmt

## Data

neu2017 from [CWMT corpus 2017](http://nlp.nju.edu.cn/cwmt-wmt/)


2 million parallel sentences (enzh)

98% of data is for training, the other for validating and testing.

## Preprocessing

- tokenizer
    - zh: jieba
    - en: nltk.word_tokenizer
- BPE: [subword-nmt](https://github.com/rsennrich/subword-nmt) (For the parameter `num_operations`, I choose 32000.)


It is worth mentioning that `BPE` reduced the vocabulary significantly, from 50000+ to 24182.

Besides, \<sos> and \<eos> symbols are conventionally prepended to each sentences. OOV is represented with \<unk>. 

## Model Architecture (seq2seq)

Similar to [Luong et. al 2015](https://arxiv.org/abs/1508.04025). 

![model](doc/model.png)

- embeddings: [glove](https://nlp.stanford.edu/projects/glove/)  (en)&  [word2vec](https://github.com/Embedding/Chinese-Word-Vectors) (zh)  (both 300-dim)
- encoder: 4-layer Bi-GRU (hidden-size 1000-dim)
- decoder: 4-layer GRU with attention (hidden-size 1000-dim)
- attention: bilinear global attention


## Training details

Hypter-Params:

- optim: Adam
- lr: 1e-4
- no L2 regularization (Since there is no obvious overfitting)
- dropout: 0.3

I found that training is a bit slow. Maybe larger lr is better. And the embeddings were fixed during training.  


## Beam Search

According to [wikipedia](https://en.wikipedia.org/wiki/Beam_search), beam search is BFS with width constraints. But I found that this method did not perform very well. 

Google's [GNMT paper](https://arxiv.org/abs/1609.08144) gave two refinements to the beam search algorithm: a coverage penalty and length normalization. The coverage penalty formula they proposed is so empirical that I just use length normalization.



## Visualization




## References


