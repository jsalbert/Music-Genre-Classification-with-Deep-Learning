# Music Genre Classification with Deep Learning

## Description
We adapt the model from [Choi et al.] (https://github.com/keunwoochoi/music-auto_tagging-keras) to train a custom music genre classification system with our own genres and data. 

We fine-tuned their model with a small dataset (30 songs per genre) and test it on the GTZAN dataset with a final accuracy of 80%. 

In this repository we provide the scripts to fine-tune the pre-trained model and a quick genre prediction algorithm using our own weights. 

Currently the genres supported are the [GTZAN dataset](http://marsyasweb.appspot.com/download/data_sets/) tags:

- Blues
- Classical
- Country
- Disco
- HipHop
- Jazz
- Metal
- Pop
- Reggae
- Rock


Needed dependencies: Keras, Theano, LibRosa, h5py, numpy
