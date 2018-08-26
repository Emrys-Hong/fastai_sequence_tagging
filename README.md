# fastai_sequence_tagging
sequence tagging for NER for ULMFiT

## data
to replicate result:
you can download the ```data/``` folder from [here](https://www.dropbox.com/sh/z308tqyqrvakl66/AADsiYKx3vfNZ3LQGInz0Q-qa?dl=0), and put it in root directory.

## run training
I am currently doing experiments in jupyter notebook ```coNLL_three_layer.ipynb```

## files modified from lesson10.ipynb
1. concat both forward and backward outputs from language model ```W_LM = [W_forward, W_backward]```

2. feeding word vectors from GloVe to a BiLSTM and get output ```W_glove```

3. concatenating these outputs ```W = [W_glove, W_LM]```

4. feeding ```W``` to another BiLSTM to get final result.

## results
F1 score of 76. 

(need to improve by fine tuning parameters, see how the toks are preprocessed, [adding char embedding](http://alanakbik.github.io/papers/coling2018.pdf), [adding CRF layer](https://arxiv.org/abs/1603.01360).

## questions
1. which layer of lanuage model should be used for Sequence tagging problem

2. how to build a better language model for sequence tagging

## relevant papers
[Regularizing and Optimizing LSTM Language Models](https://arxiv.org/pdf/1708.02182.pdf)

[deep contextualized word representations](https://arxiv.org/abs/1802.05365)

[End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](http://www.aclweb.org/anthology/P16-1101)

[Semi-supervised sequence tagging with bidirectional language models](https://arxiv.org/abs/1705.00108)

[Contextual String Embeddings for Sequence Labeling](http://alanakbik.github.io/papers/coling2018.pdf)
