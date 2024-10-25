# -*- coding: utf-8 -*-
"""
矩阵两坐标点之间画线

"""
#%% 导入包
import numpy as np

#%% 画线
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
    
#%%画图形
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

def draw_circle(mat,x,y,r=1,inplace=False):
    """
    在矩阵mat的第x行y列绘制一个半径为r的圆，(x,y)为重心所在位置
    """
    if not (0 <= x < mat.shape[0] and 0 <= y < mat.shape[1]):
        raise ValueError('坐标位置不合法')
    if not inplace:
        mat = mat.copy()
    Maxrow=mat.shape[0]
    Maxcol=mat.shape[1]
    for i in range(-r,r+1):
        if (x+i<0 or x+i>=Maxrow):
            continue
        for j in range(-r,r+1):
            if (y+j<0 or y+j>=Maxcol):
                continue
            if (i**2+j**2<=r**2):
                mat[x+i,y+j]=1
    return mat if not inplace else None