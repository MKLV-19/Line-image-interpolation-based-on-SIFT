# -*- coding: utf-8 -*-
"""
SIFT特征匹配算法

"""


#%% 导入包
import cv2
import numpy as np
import matplotlib.pyplot as plt #绘图

import sys
sys.path.append('D:\大学课程\毕设\工程文件') #修改调用路径
import draw_line as draw
import image_processing as ip
import getAdjMat
import autoMidCut_function as MidCut
from PIL import Image #导入图像处理模块

#导入图片
img = cv2.imread('C:/Users/20161/Desktop/10001.jpg') 
img2 = cv2.imread('C:/Users/20161/Desktop/10002.jpg')
#转灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


# 提取SURF或SIFT特征点
# sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.SIFT_create() #SIFT特征，参数为特征数量
# surf.setExtended(True)
kp1, des1 = surf.detectAndCompute(gray, mask=None)
kp2, des2 = surf.detectAndCompute(gray2, mask=None)

anot1 = cv2.drawKeypoints(gray, kp1, None)
anot2 = cv2.drawKeypoints(gray2, kp2, None)
plt.subplot(121)
plt.imshow(anot1)
plt.subplot(122)
plt.imshow(anot2)

# 特征点匹配
matcher = cv2.BFMatcher()
raw_matches = matcher.knnMatch(des1, des2, k=2)
good_matches = []
for m1, m2 in raw_matches:
    #  如果最接近和次接近的比值大于一个既定的值，那么我们保留这个最接近的值，认为它和其匹配的点为good_match
    if m1.distance < 0.99* m2.distance:
        good_matches.append([m1])
# good_matches = sorted(raw_matches, key=lambda x: x[0].distance)[:300]

# # RANSAC
assert len(good_matches) > 4, "Too few matches."
kp1_array = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
kp2_array = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
H, status = cv2.findHomography(kp1_array, kp2_array, cv2.RANSAC, ransacReprojThreshold=4)
good_matches = [good_matches[i] for i in range(len(good_matches)) if status[i] == 1]
imgOut = cv2.warpPerspective(gray2, H, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
#print(H)  # 变换矩阵H，透视变换？
# plt.figure()
# plt.imshow(imgOut)
# cv2.findFundamentalMat()  # 用于3D

matches = cv2.drawMatchesKnn(anot1, kp1, anot2, kp2, good_matches, None, flags = 2)

plt.figure()
plt.imshow(matches)

plt.show()

#%% 匹配点绘制测试
#二值化
new_gray=gray.copy()
for i in range(new_gray.shape[0]):
    for j in range(new_gray.shape[1]):
        if new_gray[i,j]>127:
            new_gray[i,j]=0
        else:
            new_gray[i,j]=1
ip.getArrayBoundary([new_gray])

new_gray2=gray2.copy()
for i in range(new_gray2.shape[0]):
    for j in range(new_gray2.shape[1]):
        if new_gray2[i,j]>127:
            new_gray2[i,j]=0
        else:
            new_gray2[i,j]=1
ip.getArrayBoundary([new_gray2])

kp1_array = np.int0(kp1_array) #变为整数
kp2_array = np.int0(kp2_array)

#%%
rowarray=[]
colarray=[]
row=0
col=0
for i in kp1_array:
    row,col=getAdjMat.findAnotherPoint(gray,i[0,1],i[0,0],r=10)
    rowarray.append(row)
    colarray.append(col)
    
rowarray2=[]
colarray2=[]
row2=0
col2=0
for i in kp2_array:
    row2,col2=getAdjMat.findAnotherPoint(gray2,i[0,1],i[0,0],r=10)
    rowarray2.append(row2)
    colarray2.append(col2)
#%%
r=3
adjmat=getAdjMat.getAdjMat_new(new_gray, rowarray, colarray,sense=r)
adjmat2=getAdjMat.getAdjMat_new(new_gray2, rowarray2, colarray2,sense=r)

#%%
b=np.zeros_like(new_gray)
#b=new_gray.copy() #原图绘制

#按照邻接矩阵绘图
for i in range(adjmat.shape[0]):
    for j in range(i+1,adjmat.shape[1]):
        if adjmat[i,j]:
            draw.draw_line_bold(b, kp1_array[i][0,1], kp1_array[i][0,0], kp1_array[j][0,1], kp1_array[j][0,0],pix=2,inplace=True)
for i in range(kp1_array.shape[0]):
    if status[i]:
        draw.draw_circle(b, kp1_array[i][0,1], kp1_array[i][0,0],r=10,inplace=True)
ip.array2img([b])
x=Image.fromarray(b)
x.show()
    
#%% 特征点二次匹配
#记录成功匹配点的下标
goodmatchpoint=[]
for i in range(len(rowarray)):
    if status[i]:
        goodmatchpoint.append(i)
#记录真匹配点
realpoint=[]
sdrealpoint=[]
for i in range(len(rowarray)):
    dislist=[]
    ind=0
    realpoint.append(i) #记录每个特征点真正对应的匹配点，若为成功匹配点，则为本身
    sdrealpoint.append(i)
    if not status[i]:
        #当一个点不为成功匹配的特征点时，需要根据与其最近的已匹配特征点确定其新坐标
        dislist=getAdjMat.dijkstra(adjmat, i, goodmatchpoint).copy()
        ind=dislist.index(min(dislist))
        #寻找第二近的点
        sdmin=100000
        sdind=0
        for j in range(len(dislist)):
            if j==ind:
                continue
            if dislist[j]<sdmin:
                sdmin=dislist[j]
                sdind=j
        realpoint[i]=goodmatchpoint[ind]
        sdrealpoint[i]=goodmatchpoint[sdind]
        
#%% 中间张绘制
gifname="mid2.gif"
savepath="C:/Users/20161/Desktop"
#循环作图
t=0
imgarray=[] #图像序列
w=1
for m in range(0,11):
    t=0.1*m #时间参数，取值在0-1
    c=np.zeros_like(new_gray)
    midrowarray=[] #中间点行列坐标
    midcolarray=[] 
    #生成t时刻的中间点序列
    for i in range(len(rowarray)):
        midrowarray.append(round(rowarray[i]*(1-t)+rowarray2[i]*t))
        midcolarray.append(round(colarray[i]*(1-t)+colarray2[i]*t))
    for i in range(len(rowarray)):
        if status[i]<1:
            midrowarray[i]=rowarray[i]+w*(midrowarray[realpoint[i]]-rowarray[realpoint[i]])
            +(1-w)*(midrowarray[sdrealpoint[i]]-rowarray[sdrealpoint[i]])
            midrowarray[i]=round(midrowarray[i])
            midcolarray[i]=colarray[i]+w*(midcolarray[realpoint[i]]-colarray[realpoint[i]])
            +(1-w)*(midcolarray[sdrealpoint[i]]-colarray[sdrealpoint[i]])
            midcolarray[i]=round(midcolarray[i])
    #中间点绘制
    # for i in range(len(midrowarray)):
    #     if status[i]:
    #         draw.draw_circle(c, midrowarray[i], midcolarray[i],r=10,inplace=True)

    #按照邻接矩阵绘图
    for i in range(adjmat.shape[0]):
        #根据邻接矩阵绘制图形
        for j in range(i+1,adjmat.shape[1]):
            if adjmat[i,j]:
                c=draw.draw_line_bold(c, midrowarray[i],  midcolarray[i], midrowarray[j], midcolarray[j],pix=2,inplace=False)
    imgarray.append(c.copy()) #添加新图片
ip.array2img(imgarray)
ip.createGif(imgarray, gifname, savepath,fps=20)


