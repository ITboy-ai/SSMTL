# SSMTL

Source code and annotations for :

Caihong Mu; Yu Liu; Xiangrong Yan; Aamir Ali; Yi Liu. Few-Shot Open-Set Hyperspectral Image Classification With Adaptive Threshold Using Self-Supervised Multitask Learning. IEEE TGRS, 2024. [doi:10.1109/TGRS.2024.3441617](https://doi.org/10.1109/TGRS.2024.3441617)

Contact: <liu_y@stu.xidian.edu.cn>

Code and annotations are released here, or check [https://github.com/sjliu68/MDL4OW](https://github.com/sjliu68/MDL4OW)

## Overview
##### Ordinary: misclassify road, house, helicopter, and truck 
Existing hyperspectral image (HSI) classification methods rarely consider open-set classification (OSC). Although some reconstruction-based methods can deal with OSC, they lack adaptive threshold strategies and heavily rely on the labeled samples. 

![](https://sjliu.me/images/mdl4ow1.png)

##### What we do: mask out the unknown in black
What we do here is, by using multitask deep learning, enpowering the deep learning model with the ability to identify the unknown: those masked with black color. 
For the upper image (Salinas Valley), the roads and houses between farmlands are successfully identified.
For the lower image (University of Pavia Campus), helicopters and trucks are successfully identified. 

![](https://sjliu.me/images/mdl4ow2.png)




## Key packages
    tensorflow-gpu==1.9
    keras==2.1.6
    libmr
    
Tested on Python 3.6, Windows 10

Recommend Anaconda, Spyder
    
## How to use
#### Hyperspectral satellite images
The input image is with size of imx×imy×channel. 

The satellite images are standard data, downloaded here: [http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)

The above data is in matlab format, the numpy format can be found here (recommended):
[https://drive.google.com/file/d/1cEpTuP-trfRuphKWqKHjAaJhek5sqI3C/view?usp=sharing](https://drive.google.com/file/d/1cEpTuP-trfRuphKWqKHjAaJhek5sqI3C/view?usp=sharing)



### Quick usage
    python demo_salinas.py

#### Arguments
    --nos: number of training samples per class. 20 for few-shot, 200 for many-shot
    --key: dataname: 'salinas', 'paviaU', 'indian'
    --gt: gtfile path
    --closs: classification loss weight, default=50 (0.5)
    --patience: if the training loss does not decrease for {patience} epoches, stop training
    --output: save path for output files (model, predict probabilities, predict labels, reconstruction loss)
    --showmap: save classification map
    
### Evaluation code update on 18 May 2021
When using the evaluation code "z20210518a_readoa.py", you should change the parameter "mode" for different settings. The inputs are output files from the training script.

#### Mode
    mode==0: closed-set
    mode==1: MDL4OW
    mode==2: MDL4OW/C
    mode==3: closed-set with probablity
    mode==4: softmax with threshold
    mode==5: openmax


