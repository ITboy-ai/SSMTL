# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 19:33:34 2022
@author: LY

"""
from __future__ import print_function
import keras
from keras.models import Model
from keras.layers import concatenate, Dense, Dropout, Flatten, Add, SpatialDropout2D, Conv3D
from keras.layers import Conv1D, Conv2D, Conv3D, MaxPooling2D, Input, Activation, AveragePooling2D, BatchNormalization, AveragePooling1D
from keras.layers import MaxPooling3D, AveragePooling3D, Conv2DTranspose, Reshape, Conv3DTranspose
from keras import backend as K
from keras import regularizers
from keras import initializers
from keras.initializers import he_normal, RandomNormal
from keras.layers import multiply, GlobalAveragePooling2D, GlobalAveragePooling3D
from keras.layers.core import Reshape, Dropout
from sklearn.decomposition import PCA

import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import Lambda, Softmax, Concatenate, add
from keras import Sequential
from keras.layers import SimpleRNN, LSTM, GRU
from keras.regularizers import l2
from einops import rearrange
import numpy as np


def SpectralAttention(x):
    patch = int(x.shape[1])
    conv_weight1 = Conv1D(1, kernel_size=5, strides=1, padding='same', activation='sigmoid')

    out1 = AveragePooling3D(pool_size=(patch, patch, 1))(x)  # 1*1*64*8
    out2 = MaxPooling3D(pool_size=(patch, patch, 1))(x)
    out = add([out1, out2])
    # print(out.shape)
    weight = Reshape((64*8, 1))(out)
    weight = conv_weight1(weight)
    weight = Reshape((1, 1, 64, 8))(weight)
    # print(weight.shape)
    return weight


def SpatialAttention(y):
    imy = int(y.shape[1])
    conv_weight1 = Conv2D(8, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='sigmoid')

    out1 = AveragePooling3D(pool_size=(1, 1, 64))(y)  # 9*9*1*8
    out2 = MaxPooling3D(pool_size=(1, 1, 64))(y)  # 9*9*1*8
    out = add([out1, out2])
    # print(out.shape)
    weight = Reshape((imy, imy, 1 * 8))(out)
    weight = conv_weight1(weight)
    weight = Reshape((imy, imy, 1, 8))(weight)
    # print(weight.shape)
    return weight


def group_conv1(x):
    out = Conv3D(filters=8, kernel_size=(3, 3, 3), padding='same', activation='relu')(x)
    return out


def group_conv2(x):
    out = Conv3D(filters=8, kernel_size=(5, 5, 5), padding='same', activation='relu')(x)
    return out


def group_conv3(x):
    out = Conv3D(filters=8, kernel_size=(7, 7, 7), padding='same', activation='relu')(x)
    return out


def group_conv4(x):
    out = Conv3D(filters=8, kernel_size=(9, 9, 9), padding='same', activation='relu')(x)
    return out


def CPSA(put1):
    # print("put1 shape", put1.shape)
    imxy = int(put1.shape[1])
    put1 = Reshape((imxy, imxy, 64, 1))(put1)
    planes = 64
    # 卷积前后图像大小不变，通道数变为planes // 4
    my_group_conv1 = Lambda(lambda x: group_conv1(x))
    my_group_conv2 = Lambda(lambda x: group_conv2(x))
    my_group_conv3 = Lambda(lambda x: group_conv3(x))
    my_group_conv4 = Lambda(lambda x: group_conv4(x))
    x1 = my_group_conv1(put1)  # batch*9*9*64*16
    x2 = my_group_conv2(put1)
    x3 = my_group_conv3(put1)
    x4 = my_group_conv4(put1)

    feats = Concatenate(axis=3)([x1, x2, x3, x4])  # batch*9*9*256*8
    feats = Reshape((feats.shape[1], feats.shape[2], 4, planes, 8))(feats)  # batch*9*9*4*64*8

    my_SEWeightModule = Lambda(lambda x: SpectralAttention(x))
    x1_se = my_SEWeightModule(x1)  # batch*1*1*64*8
    x2_se = my_SEWeightModule(x2)
    x3_se = my_SEWeightModule(x3)
    x4_se = my_SEWeightModule(x4)
    # print(x1_se.shape)

    my_SAWeightModule = Lambda(lambda x: SpatialAttention(x))
    x1_sa = my_SAWeightModule(x1)  # batch*9*9*1*8
    x2_sa = my_SAWeightModule(x2)
    x3_sa = my_SAWeightModule(x3)
    x4_sa = my_SAWeightModule(x4)
    # print(x1_sa.shape)

    # print("x1_sa type", type(x1_sa))
    # print("x1_sa shape", x1_sa.shape)
    x_sa = Concatenate(axis=3)([x1_sa, x2_sa, x3_sa, x4_sa])  ##batch*9*9*4*8
    # print("x_sa shape", x_sa.shape)
    #
    # print("x1_se type", type(x1_se))
    # print("x1_se shape", x1_se.shape)
    x_se = Concatenate(axis=3)([x1_se, x2_se, x3_se, x4_se])  #batch*1*1*256*8
    # print("x_se shape", x_se.shape)

    attention_vectors = Reshape((1, 1, 4, planes, 8))(x_se)  # batch*1*1*4*64*8
    # print(attention_vectors.shape)

    attention_sa = Reshape((imxy, imxy, 4, 1, 8))(x_sa)  # batch*9*9*4*1*8
    # print(attention_sa.shape)

    attention_vectors = Softmax(axis=3)(attention_vectors)  # softmax归一化

    attention_sa = Softmax(axis=3)(attention_sa)  # softmax归一化

    mydot = Lambda(lambda x: [x[0] * x[1]])
    feats_weight = mydot([feats, attention_vectors])

    mydot = Lambda(lambda x: [x[0] * x[1]])
    feats_weight = mydot([feats_weight, attention_sa])  # batch*9*9*4*64*8

    for i in range(4):
        x_se_weight_fp = Lambda(lambda z: z[:, :, :, i, :])(feats_weight)
        if i == 0:
            out = x_se_weight_fp
        else:
            out = Concatenate(axis=3)([x_se_weight_fp, out])  # batch*9*9*256*8

    "New Add (end)"

    return out


def TDM_DFPN(band, imx, ncla1, l=1):
    input1 = Input(shape=(imx, imx, band))

    conv_6 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', kernel_initializer='he_normal')
    BN6 = BatchNormalization(momentum=0.8, name='normalization6')
    ACT6 = Activation('relu')
    conv_7 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', kernel_initializer='he_normal')
    BN7 = BatchNormalization(momentum=0.8, name='normalization7')
    ACT7 = Activation('relu')
    conv_8 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', kernel_initializer='he_normal')
    BN8 = BatchNormalization(momentum=0.8, name='normalization8')
    ACT8 = Activation('relu')

    conv_9 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', kernel_initializer='he_normal')
    BN9 = BatchNormalization(momentum=0.8, name='normalization9')
    ACT9 = Activation('relu')
    conv_10 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', kernel_initializer='he_normal')
    BN10 = BatchNormalization(momentum=0.8, name='normalization10')
    ACT10 = Activation('relu')

    dense = Dense(ncla1, activation='softmax', name='output')

    # latent attention feature
    # Y1
    # print(out.shape)
    attention = CPSA(input1)
    attention = Reshape((imx, imx, 256 * 8))(attention)
    y1 = conv_6(attention)  # batch_size*(9*9*30)  ------>  batch_size*(7*7*28)*8
    y1 = BN6(y1)
    y1_ACT1 = ACT6(y1)
    print('*' * 30)
    print(y1_ACT1.shape)
    y1 = conv_7(y1_ACT1)  # batch_size*[7*(7*28)]*8  ------>  batch_size*5*5*64
    y1 = BN7(y1)
    y1_ACT2 = ACT7(y1)
    y1 = conv_8(concatenate([y1_ACT1, y1_ACT2]))  # batch_size*[7*(7*28)]*8  ------>  batch_size*5*5*64
    y1 = BN8(y1)
    y1_ACT3 = ACT8(y1)

    # deeper attention feature
    # y1
    # print(FEY1.shape)
    y1 = conv_9(concatenate([y1_ACT1, y1_ACT2, y1_ACT3]))  # batch_size*(9*9*30)  ------>  batch_size*(7*7*28)*8
    y1 = BN9(y1)
    y1_ACT4 = ACT9(y1)
    print('*' * 30)
    print(y1_ACT4.shape)
    y1 = conv_10(concatenate([y1_ACT1, y1_ACT2, y1_ACT3, y1_ACT4]))  # batch_size*[7*(7*28)]*8  ------>  batch_size*5*5*64
    y1 = BN10(y1)
    y1_ACT5 = ACT10(y1)

    pool = GlobalAveragePooling2D(name='ave_pool')(y1_ACT5)
    pre = dense(pool)

    dconv1 = Conv2DTranspose(64, kernel_size=(1, 1), padding='same')
    dconv2 = Conv2DTranspose(64, kernel_size=(3, 3), padding='same')
    dconv3 = Conv2DTranspose(64, kernel_size=(3, 3), padding='same')
    dconv4 = Conv2DTranspose(64, kernel_size=(3, 3), padding='same')
    dconv5 = Conv2DTranspose(band, kernel_size=(3, 3), padding='same')
    bn1_de = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True,
                                beta_initializer='zeros', gamma_initializer='ones',
                                moving_mean_initializer='zeros',
                                moving_variance_initializer='ones')
    bn2_de = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True,
                                beta_initializer='zeros', gamma_initializer='ones',
                                moving_mean_initializer='zeros',
                                moving_variance_initializer='ones')

    # reconstrution
    r1 = dconv1(y1_ACT5)

    r1 = bn1_de(r1)
    r1 = Activation('relu')(r1)
    r1 = dconv2(add([r1, y1_ACT4]))

    r1 = Activation('relu')(r1)
    r1 = dconv3(add([r1, y1_ACT3]))

    r1 = bn2_de(r1)
    r1 = Activation('relu')(r1)
    r1 = dconv4(add([r1, y1_ACT2]))

    r1 = Activation('relu')(r1)
    r1 = dconv5(r1)


    model1 = Model(inputs=input1, outputs=[pre, r1])
    model2 = Model(inputs=input1, outputs=[pre, r1, pool])
    return model1, model2
