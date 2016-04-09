# DER-Network

DER-Network is a neural network that read article, organize information about entities and answer questions about them,
which is proposed on our paper ["Dynamic Entity Representation with Max-pooling Improves Machine Reading."](http://www.cl.ecei.tohoku.ac.jp/publications/2016/kobayashi-dynamic-entity-naacl2016.pdf)
This repository contains code for traning new models and test them and some preprocessing for dataset.
CNN QA dataset is needed for training and test.

CNN QA dataset is available by [using download/preprocess scripts](http://cs.nyu.edu/~kcho/DMQA/) or [downloding processed dataset directly](http://cs.nyu.edu/~kcho/DMQA/). I recommend the latter way.


## Procedure

1. RNN encodes each local context for each entity's mention in each sentence.  
   An entity representation as input of RNN is constructed from preceding contexts.
2. RNN encodes the placeholder's local context in a query.
3. Attention mechanism based on the query's context merges all the local contexts of each entity by weighted mean.
4. Probability of entity as the answer is calculated by dot-product of entity's context and query's context, and softmax.


## Dependencies

These code will need:

- Python 2.7 (they may work on other versions)
- [Chainer](https://github.com/pfnet/chainer) 1.5-
- and dependencies for [Chainer](https://github.com/pfnet/chainer)


## How to run

1. Download CNN QA dataset and decompress it, following [this](http://cs.nyu.edu/~kcho/DMQA/).

2. Preprocess it.
    1. Add EOS tokens. `python -u scripts/add_eos.py dmqa_dir/cnn/questions`
    2. Preprocess. `python -u scripts/preprocess_dataset.py -dir dmqa_dir/cnn/eos_questions -save data/dataset.json`

3. Train a model. `python -u scripts/train_model.py --per 500 -savem models_dir -mn modelname -b 2 -np 6 -fe 8 --load-corpus data/dataset.json --plus-valid 8 --plus-test 8`


## Reference

If you use anything in this repository, please cite:

Sosuke Kobayashi, Ran Tian, Naoaki Okazaki and Kentaro Inui.
**Dynamic Entity Representation with Max-pooling Improves Machine Reading.**
*Proceedings of the NAACL HLT 2016, Jun. 2016.*

    @InProceedings{kobayashi2016dynamic,
      title={Dynamic Entity Representation with Max-pooling Improves Machine Reading},
      author={Sosuke Kobayashi and Ran Tian and Naoaki Okazaki and Kentaro Inui},
      booktitle={Proceedings of the NAACL HLT 2016},
      year={2016}
    }