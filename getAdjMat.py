# -*- coding: utf-8 -*-
"""
通过01矩阵中一系列坐标的连接状况，得到这些点的邻接矩阵

"""

#%% 导入包
import numpy as np

#%% 找到0点范围内的1点
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
    return [rowpoint,colpoint]

#%% 构造邻接矩阵
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

#%% 获取距离矩阵
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

#%% Dijkstra算法计算两点之间的最短路
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