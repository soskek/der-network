# DER-Network

DER-Network is a neural network that read article, organize information about entities and answer questions about them,
which is proposed on our paper ["Dynamic Entity Representation with Max-pooling Improves Machine Reading."](http://www.cl.ecei.tohoku.ac.jp/publications/2016/kobayashi-dynamic-entity-naacl2016.pdf)  
This repository will contain code for traning the new models, test for them and some preprocessing for dataset.
However, the code is under refactoring now. It will be available until main conference day of NAACL-HLT 2016, the middle of June.

CNN QA dataset is needed for training and test.

CNN QA dataset is available by [using download/preprocess scripts](http://cs.nyu.edu/~kcho/DMQA/) or [downloding processed dataset directly](http://cs.nyu.edu/~kcho/DMQA/).
I recommend the latter way.

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
