# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 19:33:34 2022
@author: LY
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#import rscls

colors = ['#000000','#CACACA', '#02FF00', '#00FFFF', '#088505', '#FF00FE', '#AA562E', '#8C0085', '#FD0000', '#FFFF00']
cmap = ListedColormap(colors)

#0000FF, #228B22, #7BFC00, #FF0000, #724A12, #C0C0C0, #00FFFF, #FF8000, #FFFF00

def save_cmap_hk(img, cmap, fname):
    colors = ['#000000','#008000','#808080','#FFF700','#0290DE','#EDC9Af','#F3F2E7']
    cmap = ListedColormap(colors)
   
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])
     
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
 
    ax.imshow(img, cmap=cmap, vmin=0, vmax=6)
    plt.savefig(fname, dpi = height)
    plt.close()

def save_cmap_pc(img, cmap, fname):
    colors = ['#000000','#0000FF','#228B22','#7BFC00', '#FF0000', '#724A12', '#C0C0C0',
              '#00FFFF', '#FF8000', '#FFFF00']
    cmap = ListedColormap(colors)
   
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])
     
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
 
    ax.imshow(img, cmap=cmap, vmin=0, vmax=9)
    plt.savefig(fname, dpi = height)
    plt.close()
    
def save_cmap_salinas16(img,cmap,fname):
    colors = ['#000000','#DCB809','#03009A','#FE0000','#FF349B','#FF66FF',
              '#0000FD','#EC8101','#00FF00','#838300','#990099','#00F7F1',
              '#009999','#009900','#8A5E2D','#67FECB','#F6EF00']
    cmap = ListedColormap(colors)
   
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])
     
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
 
    ax.imshow(img, cmap=cmap, vmin=0, vmax=16)
    plt.savefig(fname, dpi = height)
    plt.close()
    
def save_cmap_indian16(img,cmap,fname):
    colors = ['#000000','#FFFC86','#0037F3','#FF5D00','#00FB84','#FF3AFC',
              '#4A32FF','#00ADFF','#00FA00','#AEAD51','#A2549E','#54B0FF',
              '#375B70','#65BD3C','#8F462C','#6CFCAB','#FFFC00']
    cmap = ListedColormap(colors)
   
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])
     
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
 
    ax.imshow(img, cmap=cmap, vmin=0, vmax=16)
    plt.savefig(fname, dpi = height)
    plt.close()
 
# New add
def save_cmap_salinas17(img, cmap, fname):
    # colors = ['#000000', '#DCB809', '#03009A', '#FE0000', '#FF349B', '#FF66FF',
    #           '#0000FD', '#EC8101', '#00FF00', '#838300', '#990099', '#00F7F1',
    #           '#009999', '#009900', '#8A5E2D', '#67FECB', '#F6EF00', '#FFFFFF']
    colors = ['#FFFFFF', '#FF6666', '#0037F3', '#FF9900', '#CCCCCC', '#FF3AFC',
              '#FFFF66', '#00ADFF', '#00FA00', '#AEAD51', '#A2549E', '#54B0FF',
              '#375B70', '#666699', '#8F462C', '#669966', '#CC0033', '#000000']
    cmap = ListedColormap(colors)

    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img, cmap=cmap, vmin=0, vmax=17)   # 注意要改
    plt.savefig(fname, dpi=height)
    plt.close()

# New add
def save_cmap_indian17(img, cmap, fname):
    colors = ['#000000', '#FFFC86', '#0037F3', '#FF5D00', '#00FB84', '#FF3AFC',
              '#4A32FF', '#00ADFF', '#00FA00', '#AEAD51', '#A2549E', '#54B0FF',
              '#375B70', '#65BD3C', '#8F462C', '#6CFCAB', '#FFFC00', '#FFFFFF']

    cmap = ListedColormap(colors)

    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img, cmap=cmap, vmin=0, vmax=17)
    plt.savefig(fname, dpi=height)
    plt.close()

# New add
def save_cmap_indian9(img, cmap, fname):
    #colors = ['#000000','#0037F3', '#FF5D00', '#FF3AFC','#00FA00', '#A2549E', '#54B0FF', '#375B70','#8F462C', '#FFFFFF']
    colors = ['#FFFFFF', '#FF6666', '#0037F3', '#FF9900', '#CCCCCC', '#FF3AFC', '#FFFF66', '#00ADFF', '#00FA00', '#000000']

    cmap = ListedColormap(colors)

    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img, cmap=cmap, vmin=0, vmax=9)
    plt.savefig(fname, dpi=height)
    plt.close()
    
# 自己添加的没有未知类的图像
def save_cmap_indian8(img, cmap, fname):
    colors = ['#000000','#0037F3', '#FF5D00', '#FF3AFC','#00FA00', '#A2549E', '#54B0FF', '#375B70','#8F462C']
    cmap = ListedColormap(colors)

    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img, cmap=cmap, vmin=0, vmax=9)
    plt.savefig(fname, dpi=height)
    plt.close()

def save_cmap_pu9(img, cmap, fname):
    colors = ['#000000','#CACACA','#02FF00','#00FFFF','#088505','#FF00FE','#AA562E','#8C0085','#FD0000', '#FFFF00']
    cmap = ListedColormap(colors)
   
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])
     
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
 
    ax.imshow(img, cmap=cmap, vmin=0, vmax=9)
    plt.savefig(fname, dpi = height)
    plt.close()
    
 # new add
def save_cmap_hu15(img, cmap, fname):
    colors = ['#000000', '#FFFC86', '#0037F3', '#FF5D00', '#00FB84', '#FF3AFC',
              '#4A32FF', '#00ADFF', '#00FA00', '#AEAD51', '#A2549E', '#54B0FF',
              '#375B70', '#65BD3C', '#8F462C', '#FFFFFF']

    cmap = ListedColormap(colors)

    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img, cmap=cmap, vmin=0, vmax=15)
    plt.savefig(fname, dpi=height)
    plt.close()

def save_cmap_hu14(img, cmap, fname):
    colors = ['#000000', '#FFFC86', '#0037F3', '#FF5D00', '#00FB84', '#FF3AFC',
              '#4A32FF', '#00ADFF', '#00FA00', '#AEAD51', '#A2549E', '#54B0FF',
              '#375B70', '#65BD3C', '#8F462C']

    cmap = ListedColormap(colors)

    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img, cmap=cmap, vmin=0, vmax=14)
    plt.savefig(fname, dpi=height)
    plt.close()

def save_cmap_pu10(img, cmap, fname):
    # colors = ['#000000', '#CACACA', '#02FF00', '#00FFFF', '#088505', '#FF00FE', '#AA562E', '#8C0085', '#FD0000',
    #           '#FFFF00', '#FFFFFF']
    colors = ['#FFFFFF', '#FF6666', '#0037F3', '#FF9900', '#CCCCCC', '#FF3AFC', '#FFFF66', '#00ADFF', '#00FA00', '#AEAD51', '#000000']

    cmap = ListedColormap(colors)

    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img, cmap=cmap, vmin=0, vmax=10)  # gai!!
    plt.savefig(fname, dpi=height)
    plt.close()
    
def save_im(img,fname):
   
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])
     
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
 
    ax.imshow(img)
    plt.savefig(fname, dpi = height)
    plt.close()
    
#save_im(rscls.strimg255(im[:,:,[50,34,20]],5),'indian_im')
#plt.imshow(rscls.strimg255(im[:,:,[50,34,20]],5))
    
#save_cmap(pre1,cmap,'a')
    

    


