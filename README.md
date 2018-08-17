# fastai_sequence_tagging
sequence tagging for NER for ULMFiT

## data
you can download the models from [here](https://www.dropbox.com/sh/tw1phbbj0mvhtwd/AADL09ugrCHozYz99knO5Nnoa?dl=0), and put it in ```data/coNLL/models ```

## run the file
run ```CUDA_VISIBLE_DEVICES=0 python train.py data/coNLL/```

## relevant papers
[Regularizing and Optimizing LSTM Language Models](https://arxiv.org/pdf/1708.02182.pdf)

[deep contextualized word representations](https://arxiv.org/abs/1802.05365)

[End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](http://www.aclweb.org/anthology/P16-1101)

[Semi-supervised sequence tagging with bidirectional language models](https://arxiv.org/abs/1705.00108)

## acknowledgements
the library is built on [pytorch](https://pytorch.org/) and [fastai](www.fast.ai), used seqdataloader from [Sebastian Ruder](http://ruder.io/) and CRF layer from [Ye-Zhixiu's repo](https://github.com/ZhixiuYe/NER-pytorch).
