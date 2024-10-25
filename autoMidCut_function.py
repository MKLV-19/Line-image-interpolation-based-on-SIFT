# -*- coding: utf-8 -*-
"""
自动中割所需函数包

@author: 吕铭凯
"""
#%% 导入包
from PIL import Image #导入图像处理模块
import numpy as np #用于矩阵处理
import cv2 #计算机视觉

import sys #路径设置
sys.path.append('调用路径') #修改调用路径

#%% 动态图像拆分、处理、合成相关函数
#gif转矩阵序列
def gif2arr(gif_path,format="png"):
    """
    将动图切分成图像序列，二值化并转换为01矩阵序列
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

#判断是否为边缘,同时保留01矩阵边缘  
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

#批量处理01矩阵序列，使其保留边缘
def getArrayBoundary(arrays):
    """
    将01矩阵列全部保留边缘
    """
    for i in range(len(arrays)):
        arrays[i]=getBoundary(arrays[i])

#找到0点范围内的1点
def findAnotherPoint(img,rowpoint,colpoint,r=1):
    """
    给定一个01矩阵img以及一个点的坐标，并将该点坐标改为范围内与之最近的1点坐标
    r为搜索半径，默认为1
    """
    #输出错误信息
    if r<=0:
        raise ValueError('半径数值不合法')
    #检测初始点
    if img[rowpoint,colpoint]==1:
        return [rowpoint,colpoint] #若初始点为1点，则不需要做任何处理
    #检测周边点
    Maxrow=img.shape[0] #图像的最大行列
    Maxcol=img.shape[1]
    for i in range(1,r+1):
        #当初始点不为1点，开始搜索半径为1（八向）内所有像素内的1点
        for j in range(-i,i+1):
            if (rowpoint+j<0 or rowpoint+j>=Maxrow):
                continue #避免行出界
            for k in range(-i,i+1):
                if abs(k)<i and abs(j)<i:
                    continue #当已经搜索过，跳过当前像素
                if (colpoint+k<0 or colpoint+k>=Maxcol):
                    continue #避免列出界
                if img[rowpoint+j,colpoint+k]==1:
                    #将点替换
                    rowpoint=rowpoint+j
                    colpoint=colpoint+k
                    return [rowpoint,colpoint]

#将01矩阵序列赋值为255和0
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

#矩阵序列转gif
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

#%% 绘图相关函数
#画线
def draw_line(mat, x0, y0, x1, y1, inplace=False):
    """
    在矩阵mat的坐标(x0,y0)和(x1,y1)之间绘制一条线段
    x为纵向，y为横向
    inplace: 是否替代原矩阵，默认为false
    """
    if not (0 <= x0 < mat.shape[0] and 0 <= x1 < mat.shape[0] and
            0 <= y0 < mat.shape[1] and 0 <= y1 < mat.shape[1]):
        raise ValueError('坐标位置不合法')
    if not inplace:
        mat = mat.copy()
    if (x0, y0) == (x1, y1):
        mat[x0, y0] = 1
        return mat if not inplace else None
    # 当x变化率小于y变化率时，将矩阵转置使线条平缓
    transpose = abs(x1 - x0) < abs(y1 - y0)
    if transpose:
        mat = mat.T
        x0, y0, x1, y1 = y0, x0, y1, x1
    # 使(x0, y0)始终在(x1, y1)左边
    if x0 > x1:
        x0, y0, x1, y1 = x1, y1, x0, y0
    # 标记起点和终点
    mat[x0, y0] = 1
    mat[x1, y1] = 1
    # 绘制两点之间的像素
    x = np.arange(x0 + 1, x1)
    y = np.round(((y1 - y0) / (x1 - x0)) * (x - x0) + y0).astype(x.dtype) #类型和x保持一致
    # 因为已经排除两点重合的情况以及斜率大于1的情况，故此处斜率必定存在
    mat[x, y] = 1
    if transpose:
        mat = mat.T
    return mat if not inplace else None
    #if not inplace:
    #    return mat if not transpose else mat
  
#画粗线条
def draw_line_bold(mat, x0, y0, x1, y1, pix=1, inplace=False):
    """
    在矩阵mat的坐标(x0,y0)和(x1,y1)之间绘制一条可控粗细的线段
    x为纵向，y为横向
    inplace: 是否替代原矩阵，默认为false
    pix: 线的粗细，默认为1像素
    """
    if not (0 <= x0 < mat.shape[0] and 0 <= x1 < mat.shape[0] and
            0 <= y0 < mat.shape[1] and 0 <= y1 < mat.shape[1]):
        raise ValueError('坐标位置不合法')
    if not inplace:
        mat = mat.copy()
    if (x0, y0) == (x1, y1):
        mat[x0, y0] = 1
        return mat if not inplace else None
    # 当x变化率小于y变化率时，将矩阵转置使线条平缓
    transpose = abs(x1 - x0) < abs(y1 - y0)
    if transpose:
        mat = mat.T
        x0, y0, x1, y1 = y0, x0, y1, x1
    # 使(x0, y0)始终在(x1, y1)左边
    if x0 > x1:
        x0, y0, x1, y1 = x1, y1, x0, y0
    # 标记起点和终点
    mat[x0, y0] = 1
    mat[x1, y1] = 1
    # 绘制两点之间的像素
    x = np.arange(x0, x1)
    y = np.round(((y1 - y0) / (x1 - x0)) * (x - x0) + y0).astype(x.dtype) #类型和x保持一致
    # 因为已经排除两点重合的情况以及斜率大于1的情况，故此处斜率必定存在
    r_up=round(pix/2) #上侧粗细
    r_down=pix-r_up #下侧粗细
    for i in range(r_up):
        mat[x,y-i-1]=1
    for i in range(r_down):
        mat[x,y+i+1]=1
    mat[x, y] = 1
    if transpose:
        mat = mat.T
    return mat if not inplace else None
    #if not inplace:
    #    return mat if not transpose else mat
    
#画三角形
def draw_triangle(mat,x,y,h=3,inplace=False):
    """
    在矩阵mat的第x行y列绘制一个高为h的等边三角形，(x,y)为重心所在位置
    """
    if not (0 <= x < mat.shape[0] and 0 <= y < mat.shape[1]):
        raise ValueError('坐标位置不合法')
    if not inplace:
        mat = mat.copy()
    up=int(2*h/3) #重心上下的像素数量
    down=h-up
    Maxrow=mat.shape[0]
    Maxcol=mat.shape[1]
    for i in range(-up+1,down+1):
        if (x+i<0 or x+i>=Maxrow):
            continue
        for j in range(-(up+i-1),up+i-1):
            r=round(j/(3**0.5)) #斜率
            if (y+r<0 or y+r>=Maxcol):
                continue
            mat[x+i,y+r]=1
    return mat if not inplace else None

#画矩形
def draw_rectangle(mat,x,y,w=3,h=3,inplace=False):
    """
    在矩阵mat的第x行y列绘制一个宽为w，高为h的矩形，(x,y)为重心所在位置
    """
    if not (0 <= x < mat.shape[0] and 0 <= y < mat.shape[1]):
        raise ValueError('坐标位置不合法')
    if not inplace:
        mat = mat.copy()
    up=round(h/2) #重心上下左右的像素数量
    down=h-up
    left=round(w/2)
    right=w-left
    Maxrow=mat.shape[0]
    Maxcol=mat.shape[1]
    for i in range(-up+1,down+1):
        if (x+i<0 or x+i>=Maxrow):
            continue
        for j in range(-left+1,right+1):
            if (y+j<0 or y+j>=Maxcol):
                continue
            mat[x+i,y+j]=1
    return mat if not inplace else None

#%% Shi-Tomasi角点识别
def getCorner_ShiTomasi(gray,light=127,findr=1,pointnum=0,quality=0.01,mindistance=10):
    """
    gray: 灰度图像
    light: 二值化亮度阈值，默认为127
    findr: 近似角点搜索半径，默认为1
    pointnum: 角点数量，默认为0，此时搜索所有角点
    quality: 角点品质因子，默认为0.01
    mindistance: 角点间的最小距离，默认为10像素
    使用Shi-Tomasi角点检测算法，识别图片img中的角点，并返回其角点的行列坐标数组
    图像经过灰度处理后进行二值化，亮度高于127则赋值为0（白色），低于127则赋值为1（黑色）
    同时搜寻距离角点最近的边缘点作为新的角点近似，搜寻范围findr
    """
    #二值化
    gray=gray.copy()
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if gray[i,j]>127:
                gray[i,j]=0
            else:
                gray[i,j]=1
    #获取角点
    corners = cv2.goodFeaturesToTrack(gray,pointnum,quality,mindistance) #corner的第一个数字为列，第二个数字为行
    corners = np.int0(corners) #变为整数
    #寻找临近点
    rowarray=[]
    colarray=[]
    row=0
    col=0
    for i in corners:
        row,col=findAnotherPoint(gray,i[0,1],i[0,0],r=findr)
        rowarray.append(row)
        colarray.append(col)
    return [rowarray,colarray]

#%% SIFT特征点匹配
def midCut(imgpath1,imgpath2,gifname,savepath,fps=24,cutnum=10,alpha=0.85,r=10,sense=3):
    """
    使用SIFT匹配两张图像的特征点，并利用特征点完成图像的中割绘制
    imgpath: 图像路径
    alpha: 特征点选取比例系数
    r: 特征点搜索半径
    sense: 邻接矩阵感知域
    gifname: gif的名称
    savepath: 保存路径
    """
    #导入图片
    img = cv2.imread(imgpath1) 
    img2 = cv2.imread(imgpath2)
    #转灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # 提取SURF或SIFT特征点
    # sift = cv2.xfeatures2d.SIFT_create()
    surf = cv2.SIFT_create() #SIFT特征，参数为特征数量
    # surf.setExtended(True)
    kp1, des1 = surf.detectAndCompute(gray, mask=None)
    kp2, des2 = surf.detectAndCompute(gray2, mask=None)
    
    # 特征点匹配
    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m1, m2 in raw_matches:
        #  如果最接近和次接近的比值大于一个既定的值，那么我们保留这个最接近的值，认为它和其匹配的点为good_match
        if m1.distance < alpha* m2.distance:
            good_matches.append([m1])
    # good_matches = sorted(raw_matches, key=lambda x: x[0].distance)[:300]
    # # RANSAC
    assert len(good_matches) > 4, "Too few matches."
    kp1_array = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    kp2_array = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, status = cv2.findHomography(kp1_array, kp2_array, cv2.RANSAC, ransacReprojThreshold=4)
    good_matches = [good_matches[i] for i in range(len(good_matches)) if status[i] == 1]
    
    #print(H)  # 变换矩阵H，透视变换？
    # plt.figure()
    # plt.imshow(imgOut)
    # cv2.findFundamentalMat()  # 用于3D
    #二值化
    new_gray=gray.copy()
    for i in range(new_gray.shape[0]):
        for j in range(new_gray.shape[1]):
            if new_gray[i,j]>127:
                new_gray[i,j]=0
            else:
                new_gray[i,j]=1
    getArrayBoundary([new_gray])

    new_gray2=gray2.copy()
    for i in range(new_gray2.shape[0]):
        for j in range(new_gray2.shape[1]):
            if new_gray2[i,j]>127:
                new_gray2[i,j]=0
            else:
                new_gray2[i,j]=1
    getArrayBoundary([new_gray2])
    
    kp1_array = np.int0(kp1_array) #变为整数
    kp2_array = np.int0(kp2_array)
    
    rowarray=[]
    colarray=[]
    row=0
    col=0
    for i in kp1_array:
        row,col=findAnotherPoint(gray,i[0,1],i[0,0],r)
        rowarray.append(row)
        colarray.append(col)
    
    rowarray2=[]
    colarray2=[]
    row2=0
    col2=0
    for i in kp2_array:
        row2,col2=findAnotherPoint(gray2,i[0,1],i[0,0],r)
        rowarray2.append(row2)
        colarray2.append(col2)
    
    #邻接矩阵构造
    adjmat=getAdjMat_new(new_gray, rowarray, colarray,sense)
    adjmat2=getAdjMat_new(new_gray2, rowarray2, colarray2,sense)
    
    #记录成功匹配点的下标
    goodmatchpoint=[]
    for i in range(len(rowarray)):
        if status[i]:
            goodmatchpoint.append(i)
    #记录真匹配点
    realpoint=[]
    for i in range(len(rowarray)):
        dislist=[]
        ind=0
        realpoint.append(i) #记录每个特征点真正对应的匹配点，若为成功匹配点，则为本身
        if not status[i]:
            #当一个点不为成功匹配的特征点时，需要根据与其最近的已匹配特征点确定其新坐标
            dislist=dijkstra(adjmat, i, goodmatchpoint).copy()
            ind=dislist.index(min(dislist))
            realpoint[i]=goodmatchpoint[ind]
    #中间张绘制
    #循环作图
    t=0
    step=1/cutnum
    imgarray=[] #图像序列
    for m in range(0,cutnum+1):
        t=step*m #时间参数，取值在0-1
        c=np.zeros_like(new_gray)
        midrowarray=[] #中间点行列坐标
        midcolarray=[] 
        #生成t时刻的中间点序列
        for i in range(len(rowarray)):
            midrowarray.append(round(rowarray[i]*(1-t)+rowarray2[i]*t))
            midcolarray.append(round(colarray[i]*(1-t)+colarray2[i]*t))
        for i in range(len(rowarray)):
            if status[i]<1:
                midrowarray[i]=rowarray[i]+(midrowarray[realpoint[i]]-rowarray[realpoint[i]])
                midcolarray[i]=colarray[i]+(midcolarray[realpoint[i]]-colarray[realpoint[i]])

        #按照邻接矩阵绘图
        for i in range(adjmat.shape[0]):

            #根据邻接矩阵绘制图形
            for j in range(i+1,adjmat.shape[1]):
                if adjmat[i,j]:
                    c=draw_line_bold(c, midrowarray[i],  midcolarray[i], midrowarray[j], midcolarray[j],pix=2,inplace=False)
        imgarray.append(c.copy()) #添加新图片
    array2img(imgarray)
    createGif(imgarray, gifname, savepath,fps=fps)

#%% 邻接矩阵及最短路
#DFS构造邻接矩阵
def getAdjMat(image,rowarray,colarray):
    """
    将01矩阵img中选定的各个坐标按照其连接关系转化为邻接矩阵
    rowarray为行坐标，colarray为列坐标
    """
    img=image.copy() #拷贝矩阵
    #输出错误信息
    if len(rowarray) != len(colarray):
        raise ValueError('行列坐标数目不匹配')
    pointnum=len(rowarray) #坐标点的数量
    Maxrow=img.shape[0] #图像的最大行列
    Maxcol=img.shape[1]
    adjmat=np.zeros([pointnum,pointnum]) #初始化邻接矩阵
    #检测一个点是否在list中
    def getPointIndex(row,col):
        """
        判断一个点是否在列表当中，如果在，返回其索引+1，否则返回0
        """
        for i in range(pointnum):
            if (row==rowarray[i] and col==colarray[i]):
                return i+1
        return 0
    #深度优先搜索
    #用递归写DFS（有bug）
    # def DFS(row,col,lastnum):
    #     """
    #     使用深度优先搜索遍历矩阵img，lastnum为当前搜寻出发点的编号
    #     当搜索到列表中的点时，修改邻接矩阵，同时改变lastnum的值
    #     规定：每个点的编号为其序号+1后取负值
    #     """
    #     if (row<0 or row>=Maxrow or col<0 or col>=Maxcol):
    #         return None #若出界则不进行搜索
    #     if img[row,col]<=0:
    #         return None #若不为通路则不进行搜索
    #     thisnum=getPointIndex(row, col) #获得当前点的索引值
    #     if not thisnum:
    #         img[row,col]=lastnum #若不为列表中的点，则当前点的值改为当前出发点的值
    #         thisnum=-lastnum #改变当前点的值为上一个搜索点的值
    #     else:
    #         img[row,col]=-thisnum #若为列表中的点，则当前点的值改为新的索引值
    #         adjmat[-lastnum-1,thisnum-1]=1 #更新邻接矩阵
    #         adjmat[thisnum-1,-lastnum-1]=1

    #     #八向寻路，先沿直线搜索，再沿斜线搜索
    #     DFS(row+1, col, -thisnum) #向下
    #     DFS(row, col+1, -thisnum) #向右
    #     DFS(row-1, col, -thisnum) #向上
    #     DFS(row, col-1, -thisnum) #向左
    #     #DFS(img, row+1, col+1, -thisnum) #向右下
    #     #DFS(img, row-1, col+1, -thisnum) #向右上
    #     #DFS(img, row+1, col-1, -thisnum) #向左下
    #     #DFS(img, row-1, col-1, -thisnum) #向左上
    # DFS(rowarray[0], colarray[0], -1)
    #用栈写的DFS
    s=[] #定义一个栈
    thiscol=0 #当前行列值
    thisrow=0
    thisnum=0 #当前和之前的染色值
    lastnum=0
    canvisit=img.copy() #判断是否搜索过的矩阵
    for i in range(pointnum):
        #对每一个顶点进行循环
        s.append([rowarray[i],colarray[i],-i-1]) #入栈
        canvisit[rowarray[i],colarray[i]]=0
        while s:
            #当栈不为空时进行搜索
            [thisrow,thiscol,lastnum]=s.pop() #出栈
            canvisit[thisrow,thiscol]=0
            if (thisrow<0 or thisrow>=Maxrow or thiscol<0 or thiscol>=Maxcol):
                continue #若出界则不进行搜索
            if img[thisrow,thiscol]<=0 :
                continue #若不为通路则不进行搜索
            thisnum=getPointIndex(thisrow, thiscol)
            if not thisnum:
                #不在列表中
                img[thisrow,thiscol]=int(lastnum)
                thisnum=-int(lastnum)
            else:
                #在列表中
                img[thisrow,thiscol]=-int(thisnum)
                adjmat[-lastnum-1,thisnum-1]=1 #更新邻接矩阵
                adjmat[thisnum-1,-lastnum-1]=1
            #八向寻路（加判断）
            # if thisrow+1<Maxrow:
            #      if canvisit[thisrow+1,thiscol]:
            #          s.append([thisrow+1,thiscol,-int(thisnum)])           
            # if thisrow+1<Maxrow and thiscol+1<Maxcol:
            #     if canvisit[thisrow+1,thiscol+1]:
            #         s.append([thisrow+1,thiscol+1,-int(thisnum)])
            # if thiscol+1<Maxcol:
            #     if canvisit[thisrow,thiscol+1]:
            #         s.append([thisrow,thiscol+1,-int(thisnum)])                
            # if thisrow-1>=0 and thiscol+1<Maxcol:
            #     if canvisit[thisrow-1,thiscol+1]:
            #         s.append([thisrow-1,thiscol+1,-int(thisnum)])  
            # if thisrow-1>=0:
            #     if canvisit[thisrow-1,thiscol]:
            #         s.append([thisrow-1,thiscol,-int(thisnum)])                 
            # if thisrow-1>=0 and thiscol-1>=0:
            #     if canvisit[thisrow-1,thiscol-1]:
            #         s.append([thisrow-1,thiscol-1,-int(thisnum)])   
            # if thiscol-1>=0:
            #     if canvisit[thisrow,thiscol-1]:
            #         s.append([thisrow,thiscol-1,-int(thisnum)])                  
            # if thisrow+1<Maxrow and thiscol-1>=0:
            #     if canvisit[thisrow+1,thiscol-1]:
            #         s.append([thisrow+1,thiscol-1,-int(thisnum)])                
            #八向寻路（不加判断）
            if canvisit[thisrow+1,thiscol]:
                s.append([thisrow+1,thiscol,-int(thisnum)])           
            if canvisit[thisrow+1,thiscol+1]:
                s.append([thisrow+1,thiscol+1,-int(thisnum)])
            if canvisit[thisrow,thiscol+1]:
                s.append([thisrow,thiscol+1,-int(thisnum)])                
            if canvisit[thisrow-1,thiscol+1]:
                s.append([thisrow-1,thiscol+1,-int(thisnum)])  
            if canvisit[thisrow-1,thiscol]:
                s.append([thisrow-1,thiscol,-int(thisnum)])                 
            if canvisit[thisrow-1,thiscol-1]:
                s.append([thisrow-1,thiscol-1,-int(thisnum)])   
            if canvisit[thisrow,thiscol-1]:
                s.append([thisrow,thiscol-1,-int(thisnum)])                  
            if canvisit[thisrow+1,thiscol-1]:
                s.append([thisrow+1,thiscol-1,-int(thisnum)])    
    return adjmat

#BFS构造邻接矩阵
def getAdjMat_BFS(image,rowarray,colarray):
    """
    将01矩阵img中选定的各个坐标按照其连接关系转化为邻接矩阵
    rowarray为行坐标，colarray为列坐标
    """
    img=image.copy() #拷贝矩阵
    #输出错误信息
    if len(rowarray) != len(colarray):
        raise ValueError('行列坐标数目不匹配')
    pointnum=len(rowarray) #坐标点的数量
    Maxrow=img.shape[0] #图像的最大行列
    Maxcol=img.shape[1]
    adjmat=np.zeros([pointnum,pointnum]) #初始化邻接矩阵
    #检测一个点是否在list中
    def getPointIndex(row,col):
        """
        判断一个点是否在列表当中，如果在，返回其索引+1，否则返回0
        """
        for i in range(pointnum):
            if (row==rowarray[i] and col==colarray[i]):
                return i+1
        return 0
    #用队列写的BFS
    s=[] #定义一个队列
    thiscol=0 #当前行列值
    thisrow=0
    thisnum=0 #当前和之前的染色值
    lastnum=0
    canvisit=img.copy() #判断是否搜索过的矩阵
    
    for i in range(pointnum):
        #对每一个顶点进行循环
        s.append([rowarray[i],colarray[i],-i-1]) #入队列
        while s:
            #当栈不为空时进行搜索
            [thisrow,thiscol,lastnum]=s[0].copy() #出队列
            s.remove(s[0])
            if not canvisit[thisrow,thiscol]:
                continue
            canvisit[thisrow,thiscol]=0
            if (thisrow<0 or thisrow>=Maxrow or thiscol<0 or thiscol>=Maxcol):
                continue #若出界则不进行搜索
            if img[thisrow,thiscol]<=0 :
                continue #若不为通路则不进行搜索
            thisnum=getPointIndex(thisrow, thiscol)
            if not thisnum:
                #不在列表中
                img[thisrow,thiscol]=int(lastnum)
                thisnum=-int(lastnum)
            else:
                #在列表中
                img[thisrow,thiscol]=-int(thisnum)
                adjmat[-lastnum-1,thisnum-1]=1 #更新邻接矩阵
                adjmat[thisnum-1,-lastnum-1]=1
            #八向寻路
            if canvisit[thisrow+1,thiscol]:
                s.append([thisrow+1,thiscol,-int(thisnum)])
                
            if canvisit[thisrow+1,thiscol+1]:
                s.append([thisrow+1,thiscol+1,-int(thisnum)])
            
            if canvisit[thisrow,thiscol+1]:
                s.append([thisrow,thiscol+1,-int(thisnum)])
            
            if canvisit[thisrow-1,thiscol+1]:
                s.append([thisrow-1,thiscol+1,-int(thisnum)])
                
            if canvisit[thisrow-1,thiscol]:
                s.append([thisrow-1,thiscol,-int(thisnum)])
                
            if canvisit[thisrow-1,thiscol-1]:
                s.append([thisrow-1,thiscol-1,-int(thisnum)])
                
            if canvisit[thisrow,thiscol-1]:
                s.append([thisrow,thiscol-1,-int(thisnum)])
                
            if canvisit[thisrow+1,thiscol-1]:
                s.append([thisrow+1,thiscol-1,-int(thisnum)])   
    return adjmat

#感知+DFS构造邻接矩阵
def getAdjMat_new(image,rowarray,colarray,sense=1):
    """
    将01矩阵img中选定的各个坐标按照其连接关系转化为邻接矩阵
    rowarray为行坐标，colarray为列坐标
    在遍历每个每个格子时进行八向搜索，判断是否有列表中的其它坐标
    sense为感知半径，默认为1
    """
    img=image.copy() #拷贝矩阵
    #输出错误信息
    if len(rowarray) != len(colarray):
        raise ValueError('行列坐标数目不匹配')
    pointnum=len(rowarray) #坐标点的数量
    Maxrow=img.shape[0] #图像的最大行列
    Maxcol=img.shape[1]
    adjmat=np.zeros([pointnum,pointnum]) #初始化邻接矩阵
    #检测一个点是否在list中
    def getPointIndex(row,col):
        """
        判断一个点是否在列表当中，如果在，返回其索引+1，否则返回0
        """
        for i in range(pointnum):
            if (row==rowarray[i] and col==colarray[i]):
                return i+1
        return 0
    #检测八向的点是否在list中
    def findPointInList(row,col):
        """
        返回当前点周围八个方向在列表中的点，若不存在，则返回空值
        """
        flag=0
        for k in range(1,sense+1):
            #选取不同的半径，先从小半径开始
            
            for i in range(-k,k+1):
                if (row+i<0 or row+i>=Maxrow):
                    continue #防出界
                for j in range(-k,k+1):
                    if abs(i)<k and abs(j)<k:
                        continue #当已经搜索过，跳过当前像素
                    if (col+j<0 or col+j>=Maxcol):
                        continue #防出界
                    if (i==0 and j==0):
                        continue #防止返回本点
                    flag=getPointIndex(row+i, col+j)
                    if flag:
                        #当该点在list中
                        return [row+i,col+j,flag] #返回坐标值以及索引
        return None #若不存在则返回空
    
    #用栈写的DFS
    s=[] #定义一个栈
    thiscol=0 #当前行列值
    thisrow=0
    thisnum=0 #当前和之前的染色值
    lastnum=0
    canvisit=img.copy() #判断是否搜索过的矩阵
    nextrow=0 #将要搜寻下一个点的值
    nextcol=0
    nextnum=0
    flag=0
    
    for i in range(pointnum):
        #对每一个顶点进行循环
        s.append([rowarray[i],colarray[i],-i-1]) #入栈
        while s:
            #当栈不为空时进行搜索
            [thisrow,thiscol,lastnum]=s.pop() #出栈
            canvisit[thisrow,thiscol]=0
            if (thisrow<0 or thisrow>=Maxrow or thiscol<0 or thiscol>=Maxcol):
                continue #若出界则不进行搜索
            if img[thisrow,thiscol]<=0 :
                continue #若不为通路则不进行搜索
            thisnum=getPointIndex(thisrow, thiscol)
            if not thisnum:
                #不在列表中
                img[thisrow,thiscol]=int(lastnum)
                thisnum=-int(lastnum)
            else:
                #在列表中
                img[thisrow,thiscol]=-int(thisnum)
                adjmat[-lastnum-1,thisnum-1]=1 #更新邻接矩阵
                adjmat[thisnum-1,-lastnum-1]=1
            #预先判断周围八点是否有列表中的点
            flag=findPointInList(thisrow, thiscol)
            if flag:
                #当找到点
                [nextrow,nextcol,nextnum]=flag.copy() #传回点的坐标
                if canvisit[nextrow,nextcol]:
                    adjmat[nextnum-1,-img[thisrow,thiscol]-1]=1 #更新邻接矩阵
                    adjmat[-img[thisrow,thiscol]-1,nextnum-1]=1
                    s.append([nextrow,nextcol,-int(nextnum)])
                    continue                    
                
            #八向寻路
            if canvisit[thisrow+1,thiscol]:
                s.append([thisrow+1,thiscol,-int(thisnum)])
                
            if canvisit[thisrow+1,thiscol+1]:
                s.append([thisrow+1,thiscol+1,-int(thisnum)])
            
            if canvisit[thisrow,thiscol+1]:
                s.append([thisrow,thiscol+1,-int(thisnum)])
            
            if canvisit[thisrow-1,thiscol+1]:
                s.append([thisrow-1,thiscol+1,-int(thisnum)])
                
            if canvisit[thisrow-1,thiscol]:
                s.append([thisrow-1,thiscol,-int(thisnum)])
                
            if canvisit[thisrow-1,thiscol-1]:
                s.append([thisrow-1,thiscol-1,-int(thisnum)])
                
            if canvisit[thisrow,thiscol-1]:
                s.append([thisrow,thiscol-1,-int(thisnum)])
                
            if canvisit[thisrow+1,thiscol-1]:
                s.append([thisrow+1,thiscol-1,-int(thisnum)])
    return adjmat

#获取距离矩阵
def getDisMat(adjmat,rowarray,colarray):
    """
    根据邻接矩阵获取距离矩阵
    """
    #输出错误信息
    if len(rowarray) != len(colarray):
        raise ValueError('行列坐标数目不匹配')
    pointnum=len(rowarray)
    dismat=np.zeros_like(adjmat)
    d=0
    for i in range(pointnum):
        for j in range(i+1,pointnum):
            if adjmat[i,j]>0:
                #当两点之间发生连接时，计算两点之间的距离
                d=round(((rowarray[i]-rowarray[j])**2+(colarray[i]-colarray[j])**2)**(1/2))#欧氏距离
                dismat[i,j]=d
                dismat[j,i]=d
    return dismat

#Dijkstra算法计算两点之间的最短路
def dijkstra(dismat,startnum,endnumlist,rowarray=[],colarray=[]):
    """
    使用Dijkstra算法计算两点之间的最短路线
    adjmat:邻接矩阵
    startnum:起始点的序号
    endnumlist:终点的序号列表
    当传入的坐标为空时，使用邻接数作为距离，否则使用距离矩阵
    """
    if len(rowarray) != len(colarray):
        raise ValueError('行列坐标数目不匹配')
    pointnum=dismat.shape[0]
    if startnum<0 or startnum>=pointnum:
        raise ValueError('起始点不合法')
    for i in range(len(endnumlist)):
        if endnumlist[i]<0 or endnumlist[i]>=pointnum:
            raise ValueError('终点不合法')  
    #邻接矩阵初始化
    themat=np.zeros_like(dismat)
    if not rowarray:
        #当坐标为空时，使用邻接矩阵
        for i in range(pointnum):
            for j in range(i+1,pointnum):
                if dismat[i,j]>0:
                    themat[i,j]=1
                    themat[j,i]=1
    else:
        #当坐标不为空时，使用距离矩阵
        themat=dismat.copy()
    #距离列表初始化
    INF=100000
    dis=[]
    visited=[]
    for i in range(pointnum):
        dis.append(INF)
        visited.append(0)
    dis[startnum]=0
    visited[startnum]=1
    for i in range(pointnum):
        if themat[startnum,i]>0:
            dis[i]=themat[startnum,i]
    #更新距离列表
    ind=startnum #当前最短路径对应终点的索引
    mindis=INF #最小距离
    for i in range(pointnum-1):
        #每次循环更新一个最近距离
        #找出当前dis列表的最小值以及最小下标
        mindis=INF
        for j in range(pointnum):
            if visited[j]:
                continue #若已经搜索过，则直接跳过
            if dis[j]<mindis:
                mindis=dis[j]
                ind=j
        visited[ind]=1 #标记
        for j in range(pointnum):
            if themat[ind,j]>0:
                sumdis=themat[ind,j]+dis[ind]
                if sumdis<dis[j]:
                    dis[j]=sumdis #更新最短距离列表
    finaldis=[] #返回最小距离列表
    for i in range(len(endnumlist)):
        finaldis.append(dis[endnumlist[i]])
    return finaldis
