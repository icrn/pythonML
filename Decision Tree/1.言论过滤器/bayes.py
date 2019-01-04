# -*- coding: UTF-8 -*-
import numpy as np
from functools import reduce



"""
函数说明:创建实验样本
Parameters:
	无
Returns:
	postingList - 实验样本切分的词条
	classVec - 类别标签向量
"""

def loadDataSet():
        postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # 切分的词条
                       ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                       ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                       ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                       ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                       ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
        classVec = [0, 1, 0, 1, 0, 1]  # 类别标签向量，1代表侮辱性词汇，0代表不是
        return postingList, classVec

"""
函数说明:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
Parameters:
	dataSet - 整理的样本数据集
Returns:
	vocabSet - 返回不重复的词条列表，也就是词汇表
"""
def createVocabList(dataSet):
        vocabSet = set([])
        for document in dataSet:
            vocabSet = vocabSet | set(document)
        return vocabSet

"""
函数说明:根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0
Parameters:
	vocabList - createVocabList返回的列表
	inputSet - 切分的词条列表
Returns:
	returnVec - 文档向量,词集模型
"""
def setOfWords2Vec():
