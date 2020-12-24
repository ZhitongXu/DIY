# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:01:43 2019

@author: Rhodia
"""

import numpy as np
import operator
import copy
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.stats import chi2     

def genDataSet(filename):
    datalist = []
    labels = []
    with open(filename, 'r') as f:
        data = f.readlines()  #txt中所有字符串读入data
    for line in data:
        odom = line.strip('\n')       #去掉每行的换行符'\n'
        odom = odom.split(',')        #将单个数据分隔开存好
        datalist.append(odom)
    data = datalist[1:]
    
    labels = datalist[:1][0]
    del labels[-1]
    
    return labels, data

labels, data = genDataSet('german.csv')

choose = []

def step1(dataSet, X):
    # X 为待分类的属性，如果只有一类，停止，设 p 值为 1
    labels = []
    labels.append(dataSet[0][X])

    choose.append([X, 1, labels])

def step2(dataSet, X, alpha, featType):
    # X 为待分类的属性，如果有 2 类，转到 step8
    # 否则进行 step3
    labels = []
    
    for featVec in dataSet:
        currentLabel = featVec[X]
        if currentLabel not in labels:
            labels.append(currentLabel)
    
    if len(labels) > 2:
        step3(dataSet, X, alpha, featType)
    else:
        step8(dataSet, X)

# this is ordinal 连续型变量 的情况
def step3(dataSet, X, alpha, featType):
    # X 有 ≥3 类
    # 找到 p值 最大的那一对
    # 如果是类别型变量，选取对时为任意选取两个为一对
    # 如果是连续型变量，选取对时为相邻的两个为一对
    
    # 如果 p值 最大的那一对的 p值 比给定的 3.84 大，合并这两类为一类，回到 step2
    # 否则，转到 step8


    labels = []
    outputs = []
    
    for featVec in dataSet:
        currentLabel = featVec[X]
        if currentLabel not in labels:
            labels.append(currentLabel)
            
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in outputs:
            outputs.append(currentLabel)
    
    d = {}
    for featVec in dataSet:
        for feature in labels:
            if featVec[X] == feature:
                key = feature
                value = featVec[-1]
                d.setdefault(key,[]).append(value)
                # d[key]为一个列表
                # 里面是所有在 X 列上取值为key的数据在status上的取值
                
    newdic = {} 
    # 为一个嵌套的字典 
    # 键为属性上的具体取值
    # 值为每个具体属性类上统计出的各个status的个数，也是一个字典
    for item in d:
        dsub = {}
        uniqueVals = list(set(d[item]))
        for uniqueVal in uniqueVals:
            key = uniqueVal
            value = d[item].count(uniqueVal)
            dsub[key] = value
        newkey = item
        newvalue = dsub
        newdic[newkey] = newvalue
    
    # 连续型变量需要键有序
    srtdic = sorted(newdic.items(),key = lambda item:item[0])
    #print(srtdic)
    
    if featType == 'ordinal':
        # 相邻为一对计算p值
        ka2list = []
        for i in range(len(srtdic)-1):
            ka2 = calKa2(srtdic[i][1], srtdic[i+1][1]) # 取元组的第二个元素 字典
            ka2list.append(ka2)
            
        # p值最大的一对以及对应的p值
        feat1 = srtdic[ka2list.index(max(ka2list))][0]
        feat2 = srtdic[ka2list.index(max(ka2list))+1][0]
        pair = srtdic[ka2list.index(max(ka2list))][0] + srtdic[ka2list.index(max(ka2list))+1][0]
        p_value = max(ka2list)
            
    elif featType == 'nominal':
        # 任意一对计算p值
        ka2list = []
        for i in range(len(srtdic)):
            for j in range(i+1,len(srtdic)):
                ka2 = calKa2(srtdic[i][1], srtdic[j][1]) # 取元组的第二个元素 字典
                ka2list.append([ka2, i, j])
        
        pick = sorted(ka2list, key=lambda s: s[0], reverse=True)[0]
        feat1 = srtdic[pick[1]][0]
        feat2 = srtdic[pick[2]][0]
        pair = feat1 + feat2
        p_value = pick[0]
    
        
    
    if p_value > chi2.ppf(1-alpha, len(outputs)-1):
        for featVec in dataSet:
            currentLabel = featVec[X]
            if currentLabel == feat1 or currentLabel == feat2:
                featVec[X] = pair
        step2(dataSet, X, alpha, featType)
    else:
        step8(dataSet, X)

def step8(dataSet, X):
    # 计算 merge操作 后的 p值
    labels = []
    outputs = []
    
    for featVec in dataSet:
        currentLabel = featVec[X]
        if currentLabel not in labels:
            labels.append(currentLabel)
            
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in outputs:
            outputs.append(currentLabel)
    
    d = {}
    for featVec in dataSet:
        for feature in labels:
            if featVec[X] == feature:
                key = feature
                value = featVec[-1]
                d.setdefault(key,[]).append(value)
                
    newdic = {} 
    for item in d:
        dsub = {}
        uniqueVals = list(set(d[item]))
        for uniqueVal in uniqueVals:
            key = uniqueVal
            value = d[item].count(uniqueVal)
            dsub[key] = value
        newkey = item
        newvalue = dsub
        newdic[newkey] = newvalue
        
    matrix = []
    for key in newdic:
        subdic = newdic[key]
        if len(subdic) == len(outputs):
            for subkey in subdic:
                matrix.append(subdic[subkey])
        else:
            for output in outputs:
                if output in subdic.keys():
                    matrix.append(subdic[output])
                else:
                    matrix.append(0)
    matrix = np.mat(matrix)
    matrix = matrix.reshape((len(newdic),len(outputs)))
    
    matrow = matrix.sum(axis = 1)
    matcol = matrix.sum(axis = 0)
    
    total = matrow.sum()

    newmatrix = []
    for i in range(len(matcol.T)):
        for j in range(len(matrow)):
            newmatrix.append(float(matcol[0,i]*matrow[j,0])/total)
    newmatrix = np.mat(newmatrix)
    newmatrix = newmatrix.reshape((len(matcol.T),len(matrow))).T
    
    mtr = np.multiply((matrix - newmatrix),(matrix - newmatrix)) / newmatrix
    ka2 = mtr.sum()
    
    choose.append([X, ka2, labels])



def calKa2(a,b):
    # a, b 为字典
    # 这里为 2*n 的情况 
    # 2表示要判断的能否合并的两个类
    # n=2即为二分类问题，n>2则为多分类问题
    
    # 将字典中的数据整理成 2*n 的matrix来处理
    # 要考虑传进来的字典的各种情况
    
    x = copy.copy(a)
    y = copy.copy(b)
    matrix = []
                
    totalkey = []
    for key in x:
        totalkey.append(key)
    for key in y:
        totalkey.append(key)
    totalkey = list(set(totalkey))


    if len(x) < len(totalkey):
        for key in totalkey:
            if key not in x:
                x[key] = 0
    if len(y) < len(totalkey):
        for key in totalkey:
            if key not in y:
                y[key] = 0
    
    for key in x:
        matrix.append([x[key],y[key]])
                
    matrix = np.mat(matrix)
    matrix = matrix.T
    
    matrow = matrix.sum(axis = 1)
    matcol = matrix.sum(axis = 0)
    
    total = matrow.sum()

    newmatrix = []
    for i in range(len(matcol.T)):
        for j in range(len(matrow)):
            newmatrix.append(float(matcol[0,i]*matrow[j,0])/total)
    newmatrix = np.mat(newmatrix)
    newmatrix = newmatrix.reshape((len(matcol.T),len(matrow))).T
    
    mtr = np.multiply((matrix - newmatrix),(matrix - newmatrix)) / newmatrix
    ka2 = mtr.sum()
    return ka2
    







       
def singleRun(dataSet, X, alpha, featType):
    labels = []
    
    for featVec in dataSet:
        currentLabel = featVec[X]
        if currentLabel not in labels:
            labels.append(currentLabel)
    
    if len(labels) == 1:
        step1(dataSet, X)
    else:
        step2(dataSet, X, alpha, featType)
    

# singleRun(data, 0, 3.84) 
# 不知道为什么可以在函数中打印，可以声明全局变量 在step1和step8中append
# 但是不能return 然后把wholeRun函数赋值给一个变量，结果为NoneType

def chooseBestFeatureToSplit(dataSet, alpha):
    numFeatures = len(dataSet[0]) - 1
    copied = copy.deepcopy(dataSet) # 深拷贝，不对原数据进行修改
    for i in range(numFeatures):
        if i == 2:
            singleRun(copied, i, alpha, 'nominal')
        else:
            singleRun(copied, i, alpha, 'ordinal')

    # 每个属性上的最佳划分及其对应的p值 已经找到
    # 选取p值最小的属性
    pick = sorted(choose, key=lambda s: s[1])[0]
    X = pick[0]
    updatedlabels = pick[2]
    
    # 在原始数据中修改标签
    for featVec in dataSet:
        currentLabel = featVec[X]
        if currentLabel not in updatedlabels:
            for updatedlabel in updatedlabels:
                if featVec[X] in updatedlabel:
                    featVec[X] = updatedlabel
                    break
                
    # 计算卡方自由度
    outputs = []
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in outputs:
            outputs.append(currentLabel)
    v = (len(outputs)-1) * (len(updatedlabels)-1)
    
    del choose[:]
    return pick[0], pick[1], v # 返回最佳划分属性、对应的p值、merge后的自由度
    
        

# bestFeature = chooseBestFeatureToSplit(data, 3.84)
    


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels, alpha): # alpha为alpha_merge 判断是否merge
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0] # 节点纯了
    if len(dataSet[0]) == 1:
        return majorityCnt(classList) # 没有属性可以划分了
    bestFeat, p_value, n = chooseBestFeatureToSplit(dataSet, alpha)
    #if p_value > chi2.ppf(1-alpha_split, n): # 最佳划分的p值大于给定的值就停止划分
        #return majorityCnt(classList) # 需要给出alpha_split
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels, alpha)
    return myTree


Mytree = createTree(data, labels, 0.05)
# Mytree = createTree(data, labels, 0.05, 0.01)

def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

def retrieveTree(i):
    listOfTrees = [Mytree]
    return listOfTrees[i]



def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-")
    #font = FontProperties(fname=r'C:\Windows\Fonts\STXINGKA.TTF', size=15)
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, \
                            xycoords='axes fraction', \
                            xytext=centerPt, textcoords='axes fraction', \
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)   

def plotMidText(cntrPt, parentPt, txtString):
    #font = FontProperties(fname=r'C:\Windows\Fonts\MSYH.TTC', size=10)
    xMid = (parentPt[0]-cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center")
    
def plotTree(myTree, parentPt, nodeTxt):
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")
    leafNode = dict(boxstyle="round4", fc="0.8")
    numLeafs = getNumLeafs(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
    
createPlot(Mytree)