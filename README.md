# ConvLSTM-CNN-for-tropical-cyclone-prediction


## Table of contents

- [Quick start](#quick-start)
- [Status](#status)
- [What's included](#whats-included)
- [Results](#results)
- [Bugs and feature requests](#bugs-and-feature-requests)
- [Creator](#creator)
- [Thanks](#thanks)
- [Copyright and license](#copyright-and-license)


## Quick start

This project was made for 2022 NTU Remote Sensing & Geospatial Information Analysis And Application.  
There are two sections for in this project, ConvLSTM for windspeed time-series prediction and CNN for cyclone intensity prediction.  
Give a :star: if you think this project is helpful:smile:  
:exclamation: You need GPU for this project, especially for ConvLSTM :exclamation:

- Section 1: ConvLSTM ---> See ```.ipynb & colab(link) ``` &nbsp;  in Section 1 folder
- Section 1: CNN ---> See ```.ipynb ``` &nbsp;  in Section 2 folder
- :point_right: V100 32G & RTX 2080ti were used for ConvLSTM ---> Reduce batch size first if OOM occurs, also try simplifying the network structure
- :point_right: GTX 950M were used for CNN 
- Project Flow Chart as below :point_down: :point_down:  
![flow](https://user-images.githubusercontent.com/58526756/169665524-7cfb2276-5581-4b30-a7ec-f678c0f0cd06.JPG)


## Status

![Status](https://img.shields.io/badge/Keras-needed-brightgreen)
![Status](https://img.shields.io/badge/netCDF4-needed-brightgreen)
![Status](https://img.shields.io/badge/ploty-needed-brightgreen)
![Status](https://img.shields.io/badge/xarray-needed-brightgreen)
![Status](https://img.shields.io/badge/requests-needed-critical)
![Status](https://img.shields.io/badge/GPU-needed-critical)

## What's includeds
### Project excecute order
```Section 1 :``` Download_gfs.ipynb -> Generate_images_sequence.ipynb -> colab example / train.py -> make_gif.py  
```Section 2 :``` Inspect_track_data.ipynb -> Download_HURSAT.py -> Process.py -> train.py -> view_pred_images.py

### Project Notice
```Section 1 :``` images.npy (2022/01 - 2022/05) --> Smaller dataset --> prepocess contains in those files    
```train.py -1``` images_all.npy (2021/05 - 2022/05) --> modified train.py at line 17 & 18 --> change data[:-3] to data[:-4]  
```train.py -2``` images_all.npy (2021/05 - 2022/05) --> also change the batchsize to 619 in np.reshape() in line 23 & 24

### Project File
```text
Section 1/
└── windspeed_timeseries/
    ├── code/
    │   ├── train.py
    |   ├── make_gif.py
    |   ├── colab_train_link.txt
    |   ├── Generate_images_sequence.ipynb (provide link since > 30 Mb)
    |   └── Download_gfs.ipynb
    └── dataset/
        └──link for images.npy (2022/01 - 2022/05) & images_all.npy (2021/05 - 2022/05)
              
Section 2/
└── cyclone_intensity/
    ├── code/
    │   ├── Download_HURSAT.py
    │   |── Process.py
    |   |── train.py
    |   |── view_pred_images.py
    |   └── Inspect_track_data.ipynb
    └── dataset/
        └── link for images.npy & labels.npy &  5 fold prediction result
        
```

## Results
Up: ConvLSTM predicts 2 in 5 frame // Down: CNN predition examples 
![ConvLSTM_Result](https://user-images.githubusercontent.com/58526756/169666668-7b7e7193-aec1-43db-8a94-c6be3771cbf9.gif)
![intensity](https://user-images.githubusercontent.com/58526756/169667014-3d562d51-60fd-4e4b-ade4-9c092ab6dda9.png)



## Bugs and feature requests

Have a bug or a feature request? Please first search for existing and closed issues.  
If your problem or idea is not addressed yet, please open a new issue.  


## Creator

**GMfatcat**

- <https://github.com/GMfatcat>
- <http://homepage.ntu.edu.tw/~r10521801/>

## Thanks

1.https://www.kaggle.com/code/kcostya/convlstm-convolutional-lstm-network-tutorial  
2.https://www.kaggle.com/code/concyclics/analysis-typhoon-size/notebook  
3.https://github.com/23ccozad/hurricane-wind-speed-cnn  
4.https://www.ncree.narl.org.tw/home  for High Performance Computing System (ConvLSTM)

## Copyright and license

Code released under the [MIT License](https://reponame/blob/master/LICENSE).  

Also Check out my related project:https://github.com/GMfatcat/py_erddap/tree/main

Enjoy :metal:
