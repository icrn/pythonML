# -*- coding:UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np

"""
函数说明:梯度上升算法测试函数
求函数f(x) = -x^2 + 4x的极大值
Parameters:
	无
Returns:
	无

"""

def Gradient_Ascent_test():
    def f_prime(x_old):
        return -2 * x_old + 4
    x_old = -1
    x_new = 0
    alpha = 0.01
    presision = 0.0000001
    while abs(x_new - x_old) > presision:
        x_old = x_new
        x_new = x_old + alpha * f_prime(x_old)
    print(x_new)

"""
函数说明:加载数据
Parameters:
	无
Returns:
	dataMat - 数据列表
	labelMat - 标签列表
"""
def loadDataSet():
	dataMat = []														#创建数据列表
	labelMat = []														#创建标签列表
	fr = open('testSet.txt')											#打开文件
	for line in fr.readlines():											#逐行读取
		lineArr = line.strip().split()									#去回车，放入列表
		dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])		#添加数据
		labelMat.append(int(lineArr[2]))								#添加标签
	fr.close()															#关闭文件
	return dataMat, labelMat


"""
函数说明:sigmoid函数
Parameters:
	inX - 数据
Returns:
	sigmoid函数
Author:
	Jack Cui
"""
def sigmoid(inX):
	return 1.0 / (1 + np.exp(-inX))
