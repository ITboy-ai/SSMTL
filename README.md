# SSMTL

Source code and annotations for :

Caihong Mu; Yu Liu; Xiangrong Yan; Aamir Ali; Yi Liu. Few-Shot Open-Set Hyperspectral Image Classification With Adaptive Threshold Using Self-Supervised Multitask Learning. IEEE TGRS, 2024. [doi:10.1109/TGRS.2024.3441617](https://doi.org/10.1109/TGRS.2024.3441617)

Contact: <liu_y@stu.xidian.edu.cn>

Code and annotations are released here, or check [https://github.com/sjliu68/MDL4OW](https://github.com/sjliu68/MDL4OW)

## Overview
Existing hyperspectral image (HSI) classification methods rarely consider open-set classification (OSC). Although some reconstruction-based methods can deal with OSC, they lack adaptive threshold strategies and heavily rely on the labeled samples. Therefore, this article proposes a self-supervised multitask learning (SSMTL) framework for few-shot open-set HSI classification, including three stages: pretraining stage (PTS), fine-tuning stage, and testing stage. The model consists of three modules: data diversification module (DDM), 3-D multiscale attention module (3D-MAM), and adaptive threshold module (ATM), as well as a backbone network: dense feature pyramid network (DFPN).


## Key packages
    tensorflow-gpu==1.9
    keras==2.1.6
    libmr

    
## How to use
#### Hyperspectral satellite images
The input image is with size of imx×imy×channel. 

The satellite images are standard data, downloaded here: [http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)

The above data is in matlab format, the numpy format can be found here (recommended):
[https://drive.google.com/file/d/1cEpTuP-trfRuphKWqKHjAaJhek5sqI3C/view?usp=sharing](https://drive.google.com/file/d/1cEpTuP-trfRuphKWqKHjAaJhek5sqI3C/view?usp=sharing)



### Quick usage
    python main.py

#### Arguments
    --DATA_KEY: dataname: 'sa'(salinas), 'pa'(paviaU), 'in'(indian)
    --nos: number of training samples per class. 20 and 10 for few-shot
    --closs: classification loss weight, default=50 (0.5)
    --patience: if the training loss does not decrease for {patience} epoches, stop training
    --output: save path for output files (model, predict probabilities, predict labels, reconstruction loss)
    --showmap: save classification map



