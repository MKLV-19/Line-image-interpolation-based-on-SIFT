# -*- coding: utf-8 -*-
"""
自动中割工程文件

@author: 吕铭凯
"""
#%% 导入包
import sys #路径设置
sys.path.append('调用路径') #修改调用路径
import autoMidCut_function as MidCut #自动中割函数


#%%预设
gifpath="path/img" #原gif的路径
gifname="new_img.gif" #新gif的名称
savepath="path/" #新gif的存储路径
imgpath1="path/img_1.jpg" #前后两张图像的路径
imgpath2="path/img_2.jpg"

#%% 中割
MidCut.midCut(imgpath1, imgpath2, gifname, savepath,cutnum=2,fps=3,alpha=0.99,r=10,sense=3)

#%% 图像处理流程
imgarray=MidCut.gif2arr(gifpath) #将gif转化为01矩阵序列
MidCut.getArrayBoundary(imgarray) #保留图像边缘
MidCut.array2img(imgarray) #将01矩阵序列转换为图像矩阵序列
MidCut.createGif(imgarray, gifname, savepath,fps=24) #将图像序列保存为gif
