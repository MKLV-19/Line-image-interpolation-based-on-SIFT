"""
用于图像处理，将一个图像序列二值化后转换为数值矩阵

@author:吕铭凯
"""

#%% 导入包
from PIL import Image #导入图像处理模块
import os #os系统操作模块，对文件夹或文件进行操作
import numpy as np #用于矩阵处理

import sys
sys.path.append('调用路径') #修改调用路径
import draw_line as draw

#%% gif转矩阵序列
def gif2arr(gif_path,format="png"):
    """
    将动图转换为矩阵序列
    gif_path:图片路径
    format:分解成的图片格式，默认为PNG
    """
    image_series=[] #图像序列
    #gif_file_name=os.path.basename(gif_path) #加载文件路径
    #base_name=str(gif_file_name).split('.')[0] #获取文件名称
    image=Image.open(gif_path) #打开图片
    for n in range(image.n_frames):
        #gen_file_name=base_name+'_'+str(n) #图像序列名称定义
        image.seek(n) #图像定位
        image_new=Image.new("L", image.size) #定义图像色彩模式(灰度图），大小（与原图相同）
        image_new.paste(image) #将图像粘贴到新定义的图像中
        image_new=image_new.point(lambda x:0 if x>127 else 1) #图像二值化并转换为矩阵
        #亮度大于127时二值化为白色
        arr = np.array(image_new) #图像矩阵化
        image_series.append(arr) #矩阵存入序列
        #image_new.save(os.path.join(tar_dir, "%s.%s" % (gen_file_name,format))) #将图像序列按路径存储
        #print(f'文件名称:{gen_file_name}.{format} 已生成')
    return image_series

#%% 判断是否为边缘,同时保留01矩阵边缘  
def isBoundary(np_mat,x,y):
    """
    判断一个01矩阵的(x,y)处元素是否为边缘
    当且仅当该点数值为1且周围一圈元素存在0时，认为该点为边缘
    为了去噪点，孤立像素不作为边缘
    """
    if(np_mat[x,y]==0):
        return False #当该点为0，其必定不为边缘
    M=np_mat.shape[0] #总行数
    N=np_mat.shape[1] #总列数
    #判断图片的四个角点
    if ((x==0)and(y==0)):
        if 0<(np_mat[x+1,y]+np_mat[x+1,y+1]+np_mat[x,y+1])<3:
            return True #当周围存在0但不全为0时，为边缘
        else:
            return False
    if ((x==M-1)and(y==0)):
        if 0<(np_mat[x-1,y]+np_mat[x-1,y+1]+np_mat[x,y+1])<3:
            return True #当周围存在0但不全为0时，为边缘
        else:
            return False
    if ((x==0)and(y==N-1)):
        if 0<(np_mat[x+1,y]+np_mat[x+1,y-1]+np_mat[x,y-1])<3:
            return True #当周围存在0但不全为0时，为边缘
        else:
            return False
    if ((x==M-1)and(y==N-1)):
        if 0<(np_mat[x-1,y]+np_mat[x-1,y-1]+np_mat[x,y-1])<3:
            return True #当周围存在0但不全为0时，为边缘
        else:
            return False
    #判断图片的四个边缘
    if (x==0):
        if 0<(np_mat[x,y-1]+np_mat[x,y+1]+np_mat[x+1,y-1]+np_mat[x+1,y]+np_mat[x+1,y+1])<5:
            return True
        else:
            return False
    if (x==M-1):
        if 0<(np_mat[x,y-1]+np_mat[x,y+1]+np_mat[x-1,y-1]+np_mat[x-1,y]+np_mat[x-1,y+1])<5:
            return True
        else:
            return False
    if (y==0):
        if 0<(np_mat[x-1,y]+np_mat[x+1,y]+np_mat[x-1,y+1]+np_mat[x,y+1]+np_mat[x+1,y+1])<5:
            return True
        else:
            return False
    if (y==N-1):
        if 0<(np_mat[x-1,y]+np_mat[x+1,y]+np_mat[x-1,y-1]+np_mat[x,y-1]+np_mat[x+1,y-1])<5:
            return True
        else:
            return False
    #判断其它点
    sum=np_mat[x-1,y]+np_mat[x+1,y]+np_mat[x-1,y-1]+np_mat[x,y-1]+np_mat[x+1,y-1]+np_mat[x-1,y+1]+np_mat[x,y+1]+np_mat[x+1,y+1]
    if 0<sum<8:
        return True
    else:
        return False
      
def getBoundary(np_mat):
    """
    将01矩阵的边缘保留下来
    """
    flagmat=np.zeros_like(np_mat) #创建一个与原矩阵相同的矩阵
    for i in range(np_mat.shape[0]):
        for j in range(np_mat.shape[1]):
            if isBoundary(np_mat, i, j):
                flagmat[i,j]=1
    np_mat=np_mat*flagmat
    return np_mat

#%% 批量处理01矩阵序列，使其保留边缘
def getArrayBoundary(arrays):
    """
    将01矩阵列全部保留边缘
    """
    for i in range(len(arrays)):
        arrays[i]=getBoundary(arrays[i])

#%% 将01矩阵序列赋值为255和0
def array2img(arrays):
    """
    将01矩阵序列映射为255和0
    """
    for n in range(len(arrays)):
        for i in range(arrays[n].shape[0]):
            for j in range(arrays[n].shape[1]):
                if arrays[n][i,j]==0:
                    arrays[n][i,j]=255
                else:
                    arrays[n][i,j]=0

#%% 矩阵序列转gif
def createGif(image_list,gif_name,tar_dir,fps=24):
    """
    将一个矩阵序列转换为gif
    image_list:用于生成动图的图片
    gif_name:字符串，生成gif的文件名
    tar_dir:目标文件夹路径
    fps:帧速率
    """
    frames=[]
    for x in image_list:
        frames.append(Image.fromarray(x))
    the_duration=1000/fps
    picname=tar_dir+'/'+gif_name
    frames[0].save(picname,save_all=True,append_images=frames,duration=the_duration,loop=0)

#%% 测试
gifpath="path/img.gif"
gifname="new_cat.gif"
savepath="path"
imgarray=gif2arr(gifpath)
getArrayBoundary(imgarray)
array2img(imgarray)
createGif(imgarray, gifname, savepath,24)
