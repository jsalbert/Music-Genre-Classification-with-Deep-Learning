# Music Genre Classification with Deep Learning

## Abstract

In this project we adapt the model from [Choi et al.](https://github.com/keunwoochoi/music-auto_tagging-keras) to train a custom music genre classification system with our own genres and data. 

We fine-tuned their model with a small dataset (30 songs per genre) and test it on the GTZAN dataset providing a final accuracy of 80%. 

## Slides and Report

- [Slides](https://github.com/jsalbert/music-genre-classification/blob/master/Slides.pdf)

- [Report](https://github.com/jsalbert/music-genre-classification/blob/master/Music_genre_recognition.pdf)

## Code 

In this repository we provide the scripts to fine-tune the pre-trained model and a quick music genre prediction algorithm using our own weights. 

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

### Prerequisites

We have used Keras running over Theano to perform the experiments. Was done previous to Keras 2.0, not sure if it will work with the new version. 
Python packages necessary specified in *requirements.txt* run:
```
 pip install -r requirements.txt
 
```
## Results

[![Sea of Dreams - Oberhofer](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://www.youtube.com/watch?v=mIDWsTwstgs)

[![Sky Full of Stars- Coldplay](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://www.youtube.com/watch?v=zp7NtW_hKJI)

