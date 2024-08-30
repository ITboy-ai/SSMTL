# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 19:33:34 2022
@author: LY
"""

# %%
import time
import os
import copy
import libmr
import numpy as np
import argparse
import rscls_my
import ipykernel
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import network as nw
import utils_openset as u
import tensorflow as tf
import keras
from sklearn import preprocessing
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.optimizers import Adadelta
from keras import losses
from keras import backend as K
from sklearn.metrics.pairwise import paired_distances as dist
from pandas import DataFrame
from sklearn.decomposition import PCA

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='manual to this script')

vbs = 1  # if vbs==0, training in silent mode; vbs==1, print training process

# parameters  note： changeit
DATA_KEY = 'in'  # datasets: sa、in、pa
EPOCH_NUM = 200
seedx = [0, 1, 2, 3, 4]
# SAVE_MODEL = True # False
# SAVE_PRETRAINED_MODEL = True # False
# REPRETRAIN = False


parser.add_argument('--model', type=str, default='TDM_DFPN')  # network: 3DMAM + DFPN
parser.add_argument('--mode', type=int, default=2)  # 1. raw  2.pretrain
parser.add_argument('--nos', type=int, default=20)  # number of training samples per class  note： changeit

if DATA_KEY == 'sa':
    parser.add_argument('--num_evm', type=int, default=16)  # number of training samples per class  note： changeit
if DATA_KEY == 'in':
    parser.add_argument('--num_evm', type=int, default=8)  # number of training samples per class  note： changeit
elif DATA_KEY == 'pa':
    parser.add_argument('--num_evm', type=int, default=9)  # number of training samples per class  note： changeit

parser.add_argument('--closs', type=int, default=50)  # classification loss weight, 50---->0.5
parser.add_argument('--patience', type=int, default=50)  # earlystopping  note： changeit
parser.add_argument('--output', type=str, default='output/')  # save path for output files
parser.add_argument('--showmap', type=int, default=0)  # show classification map, change to 0 if run multiple times
parser.add_argument('--smoothing', type=float, default=0.1)  # help="amount of label smoothing to be applied")

# Label description: gt-raw is used for data generation and gt-17 is used for testing
if DATA_KEY == 'sa':
    parser.add_argument('--key', type=str, default='salinas')  # data name
    parser.add_argument('--gt', type=str, default='data/salinas_raw_gt.npy')  # only known training samples included
    parser.add_argument('--gt_17', type=str, default='data/salinas_gt17.npy')  # novel samples included
elif DATA_KEY == 'in':
    # It seem like Indian can not both use 200 nos and hybrid model.
    parser.add_argument('--key', type=str, default='indian')  # data name
    parser.add_argument('--gt', type=str, default='data/indian_raw_gt.npy')  # only known training samples included
    parser.add_argument('--gt_17', type=str, default='data/indian_gt9.npy')  # novel samples included
elif DATA_KEY == 'pa':
    parser.add_argument('--key', type=str, default='paviaU')  # data name
    parser.add_argument('--gt', type=str, default='data/paviaU_raw_gt.npy')  # only known training samples included
    parser.add_argument('--gt_17', type=str, default='data/paviaU_gt10.npy')  # novel samples included

args = parser.parse_args(args=[])

# metrics list
oa_close = []
aa_close = []
kappa_close = []
acc_close = []
oa_global = []
aa_global = []
f1_global = []
acc_global = []
oa_class = []
aa_class = []
f1_class = []
acc_class = []


"""Hyperparameter adaptive setting """
bsz1 = 20  # batch size
ensemble = 1  # Stick to ensemble=1
loss1 = 'categorical_crossentropy'
plane = 64 # pca dimension
patch = 9
early_stopping = EarlyStopping(monitor='loss', patience=args.patience, verbose=1)


""" saving path """
MODELNAME = '_' + args.model
key2 = args.gt.split('/')[-1].split('_')[1]
imfile = 'data/' + args.key + '_im.npy'
NUM_EVM_str = '_NUMOFEVM=' + str(args.num_evm)

if args.mode == 1:
    spath = args.output + args.key + '_' + key2 + '_' + str(args.nos) + '_closs' + str(args.closs) + MODELNAME + NUM_EVM_str + '/'  # 'output/salinas_raw_10_closs50_TDM_DFPN_pretrain_NUMOFEVM=16/'
elif args.mode == 2:
    spath = args.output + args.key + '_' + key2 + '_' + str(args.nos) + '_closs' + str(args.closs) + MODELNAME + '_pretrain' + NUM_EVM_str + '/'   # 'salinas_raw_10_closs50_TDM_DFPN_pretrain_NUMOFEVM=16'
if not os.path.exists(spath):
    os.makedirs(spath)

# novellabel = gt.max()+1
if args.key == 'indian':
    novellabel = 9
elif args.key == 'paviaU':
    novellabel = 10
elif args.key == 'salinas':
    novellabel = 17

""" added function """
# PCA transform
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[-1]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX

# F1
def F_measure(preds, labels, openset=True, unknown=-1):  # F1
    if openset:
        true_pos = 0.
        false_pos = 0.
        false_neg = 0.
        for i in range(len(labels)):
            true_pos += 1 if preds[i] == labels[i] and labels[i] != unknown else 0
            false_pos += 1 if preds[i] != labels[i] and labels[i] != unknown else 0
            false_neg += 1 if preds[i] != labels[i] and labels[i] == unknown else 0

        precision = true_pos / (true_pos + false_pos + 1e-12)
        recall = true_pos / (true_pos + false_neg + 1e-12)
        return 2 * ((precision * recall) / (precision + recall + 1e-12))

def writelogtxt(logfilepath, content, with_time=True):
    f = open(logfilepath, 'a')
    # f.write(time.asctime())
    tm_str = ""
    if with_time:
        tm_str = str(time.asctime()) + " "
    # f.write(tm_str+content+'\n')
    f.write(tm_str + content)
    f.flush()
    f.close()

# Random selection of spectral bands
def rand_select(cube, bands, band):
    print('sampling begin')
    time_sample1 = int(time.time())  # training time
    for j in range(cube.shape[0]):
        lists = sorted(np.random.choice(bands, band, replace=False))
        cube_cut = cube[np.newaxis, j, :, :, lists]
        #print(cube_cut.shape)
        try:
            output = np.concatenate((output, cube_cut), axis=1)
        except:
            output = cube_cut
        del cube_cut
    output = np.transpose(output,(1,2,3,0))
        # print(output.shape)
    print('sampling end')
    time_sample2 = int(time.time())  # training time
    print('sampling time:', time_sample2 - time_sample1)
    writelogtxt(spath + args.key + '_log_' + str(seedi) + '.txt', 'sampling time:' + str(time_sample2 - time_sample1) + '\n')
    return output

# calculate the entrop of RL
def entrop(histogram):
    pro2 = np.zeros(len(histogram), dtype=np.int8)
    pro2_5 = np.zeros(len(histogram), dtype=np.int8)
    pro3 = np.zeros(len(histogram))
    pro3_5 = np.zeros(len(histogram))
    pro4 = np.zeros(len(histogram))

    for i in range(len(histogram)):
        loc = np.where(histogram <= histogram[i])[0]
        pro2[i] = len(loc)
        pro2_5[i] = len(histogram) - pro2[i]

    for i in range(len(histogram)):
        pro3[i] = np.nan_to_num(pro2[i] / np.sum(pro2))
        pro3_5[i] = np.nan_to_num(pro2_5[i] / np.sum(pro2_5))
        pro4[i] = pro3[i] * np.nan_to_num(np.log(pro3[i] + 1e-5)) + pro3_5[i] * np.nan_to_num(np.log(pro3_5[i] + 1e-5))
    return np.nan_to_num(-pro4)

# maximum peak value of the gray histogram of RL
def threshTwoPeaks(histograms):
    histogram = entrop(histograms)
    peak = np.zeros((len(histogram)))
    for i in range(2, len(histogram) - 2):
        if histogram[i] <= histogram[i - 1] and histogram[i] <= histogram[i + 1]:
            peak[i] = 1
    # print(peak)
    minLoc = np.where(peak == 1)
    # print(minLoc)
    num1 = np.where(histogram[minLoc[0]] == np.max(histogram[minLoc[0]]))[0]
    if len(num1) > 1:
        num1 = num1[0]
    # print(num1)

    num2 = len(np.where(histograms > histograms[num1])[0])
    writelogtxt(spath + args.key + '_log_' + str(seedi) + '.txt', 'len(histograms)：' + str(len(histograms)) + '\n')
    writelogtxt(spath + args.key + '_log_' + str(seedi) + '.txt', 'minLoc[0][0]：' + str(minLoc[0][0]) + '\n')
    writelogtxt(spath + args.key + '_log_' + str(seedi) + '.txt', 'minLoc[0][-1]：' + str(minLoc[0][-1]) + '\n')
    writelogtxt(spath + args.key + '_log_' + str(seedi) + '.txt', 'max：' + str(num1) + '\n')

    sum1 = np.where(histograms < histograms[num1])[0]
    sum1 = np.sum(histograms[sum1])

    thresh = sum1 / np.sum(histograms)
    rate = num2 / len(histograms)
    return thresh, rate

# calculate the entrop of CL
def entrop_pro(histogram):
    w, h, c = histogram.shape
    pro = np.zeros((w, h))

    for i in range(w):
        for j in range(h):
            pro[i][j] = - np.sum(np.log(histogram[i, j, :] + 1e-5) * histogram[i, j, :])
    # print(pro.shape)
    return pro

# maximum peak value of the gray histogram of CL
def threshTwoPeaks_pro(histograms):
    histogram = entrop_pro(histograms)
    histogram = histogram.flatten()
    peak = np.zeros((len(histogram)))
    for i in range(1, len(histogram) - 1):
        if histogram[i] <= histogram[i - 1] and histogram[i] <= histogram[i + 1]:
            peak[i] = 1
    # print(peak)
    minLoc = np.where(peak == 1)
    # print(minLoc)
    num1 = np.where(histogram[minLoc[0]] == np.max(histogram[minLoc[0]]))[0]
    if len(num1) > 1:
        num1 = num1[0]

    mask = np.zeros(len(histogram), dtype=bool)
    for i in range(len(histogram)):
        if histogram[i] > histogram[num1]:
            mask[i] = True
    return mask


"""main function"""
# TensorFlow session start
config1 = tf.ConfigProto()
config1.gpu_options.allow_growth = True
sess = tf.Session(config=config1)
K.set_session(sess)

pretrained_model_path = args.output + 'pre_model/' + args.key + '_pretrained_model_' + str(args.model) + '0.h5'

# %% data preprocess
for seedi in seedx:
    print('Random seed:', seedi)
    print('pretrained_model_path:', pretrained_model_path)
    # load image and GT
    im = np.load(imfile)  # 'data/salinas_im.npy' 图像im (512, 217, 204)
    imx, imy, imz = im.shape  # (512, 217, 204)
    im = np.float32(im)
    dataset = np.reshape(im, [imx * imy, imz, ])
    stand_scaler = preprocessing.StandardScaler()
    dataset = stand_scaler.fit_transform(dataset)
    im = dataset.reshape([imx, imy, imz])
    imx, imy, imz = im.shape

    im1 = applyPCA(im, plane)  # PCA
    im1x, im1y, im1z = im1.shape
    gt1 = np.load(args.gt)

    clss = np.unique(gt1)[1:]  # non-background labels
    gt1[gt1 == novellabel] = 0  # novel label = 0

    cls1 = gt1.max()  # Number of image classes (excluding novel class)

    c1 = rscls_my.rscls(im, gt1, cls=cls1)  # image processing
    c1.padding(patch)  # padding (520, 225, 204)

    c11 = rscls_my.rscls(im1, gt1, cls=cls1)  # image processing
    c11.padding(patch)  # padding (520, 225, 204)

    np.random.seed(seedi)
    x1_train, y1_train = c11.train_sample(args.nos)  # load train samples: <class 'tuple'>: (320, 9, 9, 204)， <class 'tuple'>: (320,)
    x1_train, y1_train = rscls_my.make_sample(x1_train, y1_train)  # augmentation : <class 'tuple'>: (1280, 9, 9, 204)， <class 'tuple'>: (1280,)
    y1_train = to_categorical(y1_train, cls1)  # to one-hot labels: <class 'tuple'>: (1280, 16)


    if args.mode == 2:
        # Unlabeled data is used as pre-training data, and novel data is not sampled
        x0_pre_train, y0_pre_train = c1.train_sample_all_known()  # load all known train samples
        x1_pre_train = rand_select(x0_pre_train, imz, plane)
        x1_pre_train, y1_pre_train = rscls_my.make_sample(x1_pre_train, y0_pre_train)  # augmentation
        y1_pre_train = to_categorical(y1_pre_train, cls1)  # to one-hot labels

        x2_pre_train = rand_select(x0_pre_train, imz, plane)
        x2_pre_train, y2_pre_train = rscls_my.make_sample(x2_pre_train, y0_pre_train)  # augmentation
        y2_pre_train = to_categorical(y2_pre_train, cls1)  # to one-hot labels

    """model defunction"""
    if args.model == 'TDM_DFPN':
        model1, model2 = nw.TDM_DFPN(im1z, patch, cls1, 1)
    else:
        exit('Error: model defined error')
    if vbs:
        model1.summary()  # print network structure

    '''if args.mode == 2 and os.path.exists(pretrained_model_path) and not REPRETRAIN:
        print('loading pretrained model weight...')
        model1 = model1.load_model(pretrained_model_path)
        #model1 = model1.load_weights(pretrained_model_path)
    else:
        print('Training pretrained model')'''
    # Because of the custom network layer, cancel the model save operation
    time2 = int(time.time())

    """************************************************* Pretrain Begin *****************************************"""
    if args.mode == 2:
        # Note that the second label is the original data that you want to compare with the reconstructed data
        if not os.path.exists(pretrained_model_path):
            model1.compile(loss=[loss1, losses.mean_absolute_error], optimizer=Adadelta(lr=1.0), metrics=['accuracy'],
                           loss_weights=[0, 1])
            model1.fit(x1_pre_train, [np.zeros_like(y1_pre_train), x1_pre_train], batch_size=bsz1, epochs=2, verbose=vbs,
                       shuffle=True, callbacks=[early_stopping])  # 通过将预测标签置为0，实现无标签数据忽略分类损失

            model1.compile(loss=[loss1, losses.mean_absolute_error], optimizer=Adadelta(lr=0.1), metrics=['accuracy'],
                           loss_weights=[0, 1])
            model1.fit(x1_pre_train, [np.zeros_like(y1_pre_train), x1_pre_train], batch_size=bsz1, epochs=2, verbose=vbs,
                       shuffle=True, callbacks=[early_stopping])

            model1.compile(loss=[loss1, losses.mean_absolute_error], optimizer=Adadelta(lr=0.01), metrics=['accuracy'],
                           loss_weights=[0, 1])
            model1.fit(x1_pre_train, [np.zeros_like(y1_pre_train), x1_pre_train], batch_size=bsz1, epochs=2, verbose=vbs,
                       shuffle=True, callbacks=[early_stopping])

            model1.compile(loss=[loss1, losses.mean_absolute_error], optimizer=Adadelta(lr=1.0), metrics=['accuracy'],
                           loss_weights=[0, 1])
            model1.fit(x2_pre_train, [np.zeros_like(y2_pre_train), x2_pre_train], batch_size=bsz1, epochs=2, verbose=vbs,
                       shuffle=True, callbacks=[early_stopping])  # 通过将预测标签置为0，实现无标签数据忽略分类损失

            model1.compile(loss=[loss1, losses.mean_absolute_error], optimizer=Adadelta(lr=0.1), metrics=['accuracy'],
                           loss_weights=[0, 1])
            model1.fit(x2_pre_train, [np.zeros_like(y2_pre_train), x2_pre_train], batch_size=bsz1, epochs=2, verbose=vbs,
                       shuffle=True, callbacks=[early_stopping])

            model1.compile(loss=[loss1, losses.mean_absolute_error], optimizer=Adadelta(lr=0.01), metrics=['accuracy'],
                           loss_weights=[0, 1])
            model1.fit(x2_pre_train, [np.zeros_like(y2_pre_train), x2_pre_train], batch_size=bsz1, epochs=2, verbose=vbs,
                       shuffle=True, callbacks=[early_stopping])

        # if SAVE_PRETRAINED_MODEL:
        # print("save model weight ...")
        # odel1.save_weights(pretrained_model_path)
        # model1.save(pretrained_model_path)
        # else:
        # print('loading pretrained model ...')
        #     model1 = None
        # model1 = keras.models.load_model(pretrained_model_path)

    """******************************************* Pretrain End *************************************************"""

    """******************************************* Training Begin ***********************************************"""
    if args.model == 'TDM_DFPN':
        # train the model with lr=1.0, 0.1, 0.01
        model1.compile(loss=[loss1, losses.mean_absolute_error], optimizer=Adadelta(lr=1), metrics=['accuracy'], loss_weights=[args.closs / 100.0, 1 - args.closs / 100.0])
        model1.fit(x1_train,
                   [y1_train, x1_train], batch_size=bsz1, epochs=EPOCH_NUM, verbose=vbs, shuffle=True, callbacks=[early_stopping])

        # then train the model with lr=0.1
        model1.compile(loss=[loss1, losses.mean_absolute_error], optimizer=Adadelta(lr=0.1), metrics=['accuracy'], loss_weights=[args.closs / 100.0, 1 - args.closs / 100.0])
        model1.fit(x1_train,
                   [y1_train, x1_train], batch_size=bsz1, epochs=EPOCH_NUM, verbose=vbs, shuffle=True, callbacks=[early_stopping])

        # then train the model with lr=0.1
        model1.compile(loss=[loss1, losses.mean_absolute_error], optimizer=Adadelta(lr=0.01), metrics=['accuracy'], loss_weights=[args.closs / 100.0, 1 - args.closs / 100.0])
        model1.fit(x1_train,
                   [y1_train, x1_train], batch_size=bsz1, epochs=EPOCH_NUM, verbose=vbs, shuffle=True, callbacks=[early_stopping])


    time3 = int(time.time())  # training time
    print('training time:', time3 - time2)
    writelogtxt(spath + args.key + '_log_' + str(seedi) + '.txt', 'training time:' + str(time3 - time2) + '\n')

    # save model
    # model1.save(spath + args.key + '_model' + '_' + str(seedi))
    # if SAVE_MODEL:
    # model1.save(spath + args.key + '_model.h5')
    # model1.save_weights(spath + args.key + '_model.h5')

    """********************************************** Training END *********************************************"""

    """********************************************** Predict Begin ********************************************"""
    # predict part, predicting image row-by-row
    pre_all = []
    preloss = []
    for i in range(ensemble):
        pre_rows_1 = []
        for j in range(im1x):
            sam_row = c11.all_sample_row(j)
            pre_row1, _ = model1.predict(sam_row)
            _ = dist(_.reshape(im1y, -1), sam_row.reshape(im1y, -1))
            preloss.append(_)
            pre_rows_1.append(pre_row1)
        pre_all.append(np.array(pre_rows_1))

    """classification loss"""
    preloss = np.array(preloss)
    preloss = np.float64(preloss.reshape(-1))
    np.save(spath + args.key + '_predictloss' + '_' + str(seedi), preloss)

    """predicted probabilities"""
    pre = pre_all[0]
    np.save(spath + args.key + '_predict' + '_' + str(seedi), pre)

    # The classification loss histogram based on training samples automatically generates thresholds
    fig = plt.figure()
    plt.hist(preloss.flatten(), bins=len(preloss.flatten()) // 2, color='g')
    plt.savefig(spath + args.key + 'reconloss_' + str(seedi))
    T_global, _ = threshTwoPeaks(preloss.flatten())  # threshold based on entropy of CL: T_global
    writelogtxt(spath + args.key + '_log_' + str(seedi) + '.txt', 'thresh_global：' + str(T_global) + '\n')

    """Closed set classification result"""
    pre0 = np.argmax(pre, axis=-1) + 1
    np.save(spath + args.key + '_close' + '_' + str(seedi), pre0)

    # get reconstruction loss
    _, trainloss = model1.predict(x1_train)

    trainloss = dist(trainloss.reshape(trainloss.shape[0], -1),
                     x1_train.reshape(x1_train.shape[0], -1))
    np.save(spath + args.key + '_trainloss' + '_' + str(seedi), trainloss)  # 2

    # mask_pro = threshTwoPeaks_pro(pre)

    # figure of histogram based on RL
    fig = plt.figure()
    plt.hist(trainloss.flatten(), bins=len(trainloss.flatten()) // 2, color='g')
    plt.savefig(spath + args.key + 'trainloss_' + str(seedi))

    _, rate_global = threshTwoPeaks(trainloss.flatten()) # threshold based on entropy of RL: rate_global
    writelogtxt(spath + args.key + '_log_' + str(seedi) + '.txt', 'rate_global：' + str(rate_global) + '\n')

    # mask1 = threshTwoPeaks_pro(pre)
    # T_global, rate_global = thresh(preloss.flatten(), mask1)

    NUM_EVM = args.num_evm
    # set EVT tail number
    numofevm_all = int(args.nos * NUM_EVM * 6 * rate_global)  # numofevm_all for the global method

    if numofevm_all < 20:
        numofevm_all = 20

    """open set classification results globally"""
    # get unknown mask    
    mr = libmr.MR()
    mr.fit_high(trainloss, numofevm_all)
    wscore = mr.w_score_vector(preloss)
    mask2 = wscore > T_global  # results based on two adaptive threshold strategies
    # print(np.array(mask1).shape)
    # print(np.array(mask2).shape)
    # print(mask1)
    # print(mask2)
    mask = mask2
    mask = mask.reshape(im1x, im1y)

    # figure of open classification results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(mask, aspect='auto', cmap=plt.cm.gray, interpolation='nearest')
    plt.axis('equal')
    plt.axis('off')
    plt.savefig(spath + args.key + 'globalmask_' + str(seedi))

    # apply novel mask
    pre = pre_all[0]
    pre1 = np.argmax(pre, axis=-1) + 1
    pre1_thred = pre1.flatten()
    pre1[mask == 1] = novellabel
    np.save(spath + args.key + '_pre_global' + '_' + str(seedi), pre1)
    pre1_mdl = copy.deepcopy(pre1)

    """open set classification results classly (Ignore this part of the result)"""
    mrs = {}  # save libmr model
    wscores = {}
    y2_train = np.argmax(y1_train, axis=-1) + 1
    np.save(spath + args.key + '_trainlabel' + '_' + str(seedi), y2_train)  # 4

    realmask = np.zeros([im1x, im1y], np.uint8)
    for cls2 in clss:
        idx = y2_train == cls2
        tmp4 = trainloss[idx]
        # print(np.array(y2_train).shape)
        # print(np.array(idx).shape)
        # print(np.array(trainloss).shape)
        # print(np.array(tmp4).shape)

        idx_thred = pre1_thred == cls2
        tmp_thred = preloss[idx_thred]
        # print(np.array(y2_train).shape)
        # print(np.array(idx_thred).shape)
        # print(np.array(preloss).shape)
        # print(np.array(tmp_thred).shape)

        # The reconstruction loss histogram based on all samples automatically generates thresholds
        T_classes, _ = threshTwoPeaks(tmp_thred.flatten())
        writelogtxt(spath + args.key + '_log_' + str(seedi) + '.txt', 'thresh_classes =' + str(cls2) + '：' + str(T_classes) + '\n')

        _, rate_classes = threshTwoPeaks(tmp4.flatten())
        writelogtxt(spath + args.key + '_log_' + str(seedi) + '.txt', 'rate_classes =' + str(cls2) + '：' + str(rate_classes) + '\n')

        numofevm = int(args.nos * 6 * rate_classes)  # numofevm for the class-wise method
        if numofevm < 3:
            numofevm = 3

        mrs[cls2] = libmr.MR()
        mrs[cls2].fit_high(tmp4, numofevm)
        wscore = mrs[cls2].w_score_vector(preloss)
        mask3 = wscore > T_classes
        mask = mask3
        mask = mask.reshape(im1x, im1y)
        realmask[np.logical_and(mask, pre1 == cls2)] = 1

    # figure of open classification results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(realmask, aspect='auto', cmap=plt.cm.gray, interpolation='nearest')
    plt.axis('equal')
    plt.axis('off')
    plt.savefig(spath + args.key + 'classwisemask_' + str(seedi))

    # apply mask, class-wise
    pre = pre_all[0]
    pre2 = np.argmax(pre, axis=-1) + 1
    pre2[realmask == 1] = novellabel
    np.save(spath + args.key + '_pre_classwise' + '_' + str(seedi), pre2)  # 5
    pre2_mdl = copy.deepcopy(pre2)
    """(Ignore above part of the result)"""

    time4 = int(time.time())  # training time
    print('predict time:', time4 - time3)
    writelogtxt(spath + args.key + '_log_' + str(seedi) + '.txt', 'predict time:' + str(time4 - time3) + '\n')

    """gt maps"""
    if args.key == 'salinas':
        gt_17 = np.load(args.gt_17)
        gt = np.load(args.gt)  # 'data/salinas_raw_gt.npy', gt（512，217），value：0 - 16

        cfm_normal = rscls_my.gtcfm(pre0, gt_17, 17)
        cfm1 = rscls_my.gtcfm(pre1, gt_17, 17)  # dsmle-os
        cfm2 = rscls_my.gtcfm(pre2, gt_17, 17)

        if args.showmap:
            u.save_cmap_salinas16(pre0, 0, spath + args.key + '_close_' + str(seedi))
            u.save_cmap_salinas17(pre1, 0, spath + args.key + '_global_' + str(seedi))
            u.save_cmap_salinas17(pre2, 0, spath + args.key + '_classwise_' + str(seedi))
            pre0_back = copy.deepcopy(pre0)
            pre0_back[gt == 0] = 0
            u.save_cmap_salinas17(pre0_back, 0, spath + args.key + '_close_with_back_' + str(seedi))

    elif args.key == 'paviaU':
        gt_17 = np.load(args.gt_17)
        gt = np.load(args.gt)

        cfm_normal = rscls_my.gtcfm(pre0, gt_17, 10)
        cfm1 = rscls_my.gtcfm(pre1, gt_17, 10)  # dsmle-os
        cfm2 = rscls_my.gtcfm(pre2, gt_17, 10)

        if args.showmap:
            u.save_cmap_pu10(pre0, 0, spath + args.key + '_close_' + str(seedi))
            u.save_cmap_pu10(pre1, 0, spath + args.key + '_global_' + str(seedi))
            u.save_cmap_pu10(pre2, 0, spath + args.key + '_classwise_' + str(seedi))
            pre0_back = copy.deepcopy(pre0)
            pre0_back[gt == 0] = 0
            u.save_cmap_pu10(pre0_back, 0, spath + args.key + '_close_with_back_' + str(seedi))  # Note: 添加背景黑色

    elif args.key == 'indian':
        gt_17 = np.load(args.gt_17)
        gt = np.load(args.gt)

        cfm_normal = rscls_my.gtcfm(pre0, gt_17, 9)
        cfm1 = rscls_my.gtcfm(pre1, gt_17, 9)  # dsmle-os
        cfm2 = rscls_my.gtcfm(pre2, gt_17, 9)

        if args.showmap:
            u.save_cmap_indian9(pre0, 0, spath + args.key + '_close_' + str(seedi))
            u.save_cmap_indian9(pre1, 0, spath + args.key + '_global_' + str(seedi))
            u.save_cmap_indian9(pre2, 0, spath + args.key + '_classwise_' + str(seedi))
            pre0_back = copy.deepcopy(pre0)
            pre0_back[gt == 0] = 0
            u.save_cmap_indian9(pre0_back, 0, spath + args.key + '_close_with_back_' + str(seedi))  # Note: 添加背景黑色

    """save results and metrics to outpath"""
    data_normal = DataFrame(cfm_normal)
    data_normal_string = data_normal.to_csv(spath + args.key + '_normal_' + str(seedi) + '.csv', sep='\t')
    data2 = DataFrame(cfm1)
    data2_string = data2.to_csv(spath + args.key + '_' + str(seedi) + '.csv', sep='\t')
    data3 = DataFrame(cfm2)
    data3_string = data3.to_csv(spath + args.key + '_classwise_' + str(seedi) + '.csv', sep='\t')

    writelogtxt(spath + args.key + '_log_' + str(seedi) + '.txt', '------------------ Vars ------------------\n')
    vars_str = '------------------ Vars ------------------\n'
    for k in list(vars(args).keys()):
        writelogtxt(spath + args.key + '_log_' + str(seedi) + '.txt', '{}: {} \n'.format(k, vars(args)[k]))
        vars_str += '{}: {} \n'.format(k, vars(args)[k])
    vars_str += '------------------------------------------\n'

    gt = np.load(args.gt_17)
    assert novellabel == gt.max()

    # F1
    gt = gt.reshape(-1)
    pre11 = pre1_mdl.reshape(-1)
    pre11 = pre11[gt != 0]

    pre22 = pre2_mdl.reshape(-1)
    pre22 = pre22[gt != 0]

    gt = gt[gt != 0]
    f11 = F_measure(pre11, gt, openset=True, unknown=novellabel)
    f12 = F_measure(pre22, gt, openset=True, unknown=novellabel)

    oa_close.append(cfm_normal[-1, 0])
    aa_close.append(cfm_normal[-1, 2])
    kappa_close.append(cfm_normal[-1, 2])
    acc_close.append(cfm_normal[novellabel, :-1])
    writelogtxt(spath + args.key + '_log_' + str(seedi) + '.txt',
                '\nclose: class acc， oa, aa, kappa: \n' + \
                '\n'.join(list(map(str, ["{0:.2f}".format(e * 100) for e in cfm_normal[novellabel, :-1]]))) + '\n' + \
                "{0:.2f}".format(cfm_normal[-1, 0] * 100) + '\n' + \
                "{0:.2f}".format(cfm_normal[-1, 1] * 100) + '\n' + \
                "{0:.2f}".format(cfm_normal[-1, 2] * 100) + '\n'
                )
    writelogtxt(spath + args.key + '_log_' + str(seedi) + '.txt',
                'Close classificiation CF:' + data_normal.to_string() + '\n'
                )

    vars_str += '\nDSML: oa, aa, kappa: \n' + \
                "{0:.2f}".format(cfm_normal[-1, 0] * 100) + '\n' + \
                "{0:.2f}".format(cfm_normal[-1, 1] * 100) + '\n' + \
                "{0:.2f}".format(cfm_normal[-1, 2] * 100) + '\n'

    vars_str += 'Close classificiation CF:' + data_normal.to_string() + '\n'

    oa_global.append(cfm1[-1, 0])
    aa_global.append(cfm1[-1, 1])
    f1_global.append(f11)
    acc_global.append(cfm1[novellabel, :-1])
    writelogtxt(spath + args.key + '_log_' + str(seedi) + '.txt',
                '\nopen_global: class acc, oa, aa, f1: \n' + \
                '\n'.join(list(map(str, ["{0:.2f}".format(e * 100) for e in cfm1[novellabel, :-1]]))) + '\n' + \
                "{0:.2f}".format(cfm1[-1, 0] * 100) + '\n' + \
                "{0:.2f}".format(cfm1[-1, 1] * 100) + '\n' + \
                "{0:.2f}".format(f11 * 100) + '\n'
                )
    vars_str += '\nopen_class: class acc, oa, aa, f1: \n' + \
                '\n'.join(list(map(str, ["{0:.2f}".format(e * 100) for e in cfm1[novellabel, :-1]]))) + '\n' + \
                "{0:.2f}".format(cfm1[-1, 0] * 100) + '\n' + \
                "{0:.2f}".format(cfm1[-1, 1] * 100) + '\n' + \
                "{0:.2f}".format(f11 * 100) + '\n'

    oa_class.append(cfm2[-1, 0])
    aa_class.append(cfm2[-1, 1])
    f1_class.append(f12)
    acc_class.append(cfm2[novellabel, :-1])
    writelogtxt(spath + args.key + '_log_' + str(seedi) + '.txt',
                '\nDSMLE-OS: class acc, oa, aa, f1: \n' + \
                '\n'.join(list(map(str, ["{0:.2f}".format(e * 100) for e in cfm2[novellabel, :-1]]))) + '\n' + \
                "{0:.2f}".format(cfm2[-1, 0] * 100) + '\n' + \
                "{0:.2f}".format(cfm2[-1, 1] * 100) + '\n' + \
                "{0:.2f}".format(f12 * 100) + '\n'
                )
    vars_str += '\nDSMLE-OS: class acc, oa, aa, f1: \n' + \
                '\n'.join(list(map(str, ["{0:.2f}".format(e * 100) for e in cfm2[novellabel, :-1]]))) + '\n' + \
                "{0:.2f}".format(cfm2[-1, 0] * 100) + '\n' + \
                "{0:.2f}".format(cfm2[-1, 1] * 100) + '\n' + \
                "{0:.2f}".format(f12 * 100) + '\n'


    writelogtxt(spath + args.key + '_log_' + str(seedi) + '.txt', 'Open classificiation CF:' + data2.to_string() + '\n')
    writelogtxt(spath + args.key + '_log_' + str(seedi) + '.txt', 'Open classificiation CF_os:' + data3.to_string() + '\n')
    vars_str += 'Open classificiation CF:' + data2.to_string() + '\n'
    vars_str += 'Open classificiation CF_os:' + data3.to_string() + '\n'

    print(vars_str)
    print("Output path: ", spath)

writelogtxt(spath + args.key + '_log_' + 'avarage_result' + '.txt', 'Random seed list:' + str(seedx) + '\n')
writelogtxt(spath + args.key + '_log_' + 'avarage_result' + '.txt', "oa_close:" + str(oa_close) + '\n')
writelogtxt(spath + args.key + '_log_' + 'avarage_result' + '.txt', "aa_close:" + str(aa_close) + '\n')
writelogtxt(spath + args.key + '_log_' + 'avarage_result' + '.txt', "kappa_close:" + str(kappa_close) + '\n')
writelogtxt(spath + args.key + '_log_' + 'avarage_result' + '.txt', "acc_close:" + str(acc_close) + '\n')
writelogtxt(spath + args.key + '_log_' + 'avarage_result' + '.txt',
            "mean_close:oa,aa,kp: " + str(np.mean(oa_close) * 100) + " " + str(np.mean(aa_close) * 100) + " " + str(np.mean(kappa_close) * 100) + '\n')
writelogtxt(spath + args.key + '_log_' + 'avarage_result' + '.txt',
            "std:oa,aa,kp: " + str(np.std(oa_close) * 100) + " " + str(np.std(aa_close) * 100) + " " + str(np.std(kappa_close) * 100) + '\n')
writelogtxt(spath + args.key + '_log_' + 'avarage_result' + '.txt', "mean_acc_close: " + '\n')
m_acc_close = np.mean(acc_close, axis=0)
for a in m_acc_close:
    writelogtxt(spath + args.key + '_log_' + 'avarage_result' + '.txt', str(a * 100) + '\n', with_time=False)

writelogtxt(spath + args.key + '_log_' + 'avarage_result' + '.txt', "oa_global:" + str(oa_global) + '\n')
writelogtxt(spath + args.key + '_log_' + 'avarage_result' + '.txt', "aa_global:" + str(aa_global) + '\n')
writelogtxt(spath + args.key + '_log_' + 'avarage_result' + '.txt', "f1_global:" + str(f1_global) + '\n')
writelogtxt(spath + args.key + '_log_' + 'avarage_result' + '.txt', "acc_global:" + str(acc_global) + '\n')
writelogtxt(spath + args.key + '_log_' + 'avarage_result' + '.txt',
            "mean_global:oa,aa,kp: " + str(np.mean(oa_global) * 100) + " " + str(np.mean(aa_global) * 100) + " " + str(np.mean(f1_global) * 100) + '\n')
writelogtxt(spath + args.key + '_log_' + 'avarage_result' + '.txt',
            "std:oa,aa,f1: " + str(np.std(oa_global) * 100) + " " + str(np.std(aa_global) * 100) + " " + str(np.std(f1_global) * 100) + '\n')
writelogtxt(spath + args.key + '_log_' + 'avarage_result' + '.txt', "mean_acc_global: " + '\n')
m_acc_global = np.mean(acc_global, axis=0)
for a in m_acc_global:
    writelogtxt(spath + args.key + '_log_' + 'avarage_result' + '.txt', str(a * 100) + '\n', with_time=False)

writelogtxt(spath + args.key + '_log_' + 'avarage_result' + '.txt', "oa_class:" + str(oa_class) + '\n')
writelogtxt(spath + args.key + '_log_' + 'avarage_result' + '.txt', "aa_class:" + str(aa_class) + '\n')
writelogtxt(spath + args.key + '_log_' + 'avarage_result' + '.txt', "f1_class:" + str(f1_class) + '\n')
writelogtxt(spath + args.key + '_log_' + 'avarage_result' + '.txt', "acc_class:" + str(acc_class) + '\n')
writelogtxt(spath + args.key + '_log_' + 'avarage_result' + '.txt',
            "mean_class:oa,aa,f1: " + str(np.mean(oa_class) * 100) + " " + str(np.mean(aa_class) * 100) + " " + str(np.mean(f1_class) * 100) + '\n')
writelogtxt(spath + args.key + '_log_' + 'avarage_result' + '.txt',
            "std:oa,aa,f1: " + str(np.std(oa_class) * 100) + " " + str(np.std(aa_class) * 100) + " " + str(np.std(f1_class) * 100) + '\n')
writelogtxt(spath + args.key + '_log_' + 'avarage_result' + '.txt', "mean_acc_class: " + '\n')
m_acc_class = np.mean(acc_class, axis=0)
for a in m_acc_class:
    writelogtxt(spath + args.key + '_log_' + 'avarage_result' + '.txt', str(a * 100) + '\n', with_time=False)
