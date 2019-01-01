
import numpy as np
import collections


def createDataSet():
        # 数据集合
        group = np.array([[1,101],[5,89],[108,5],[115,8]])
        # 标签
        labels = ['爱情片','爱情片','动作片','动作片']
        return group, labels

# knn 分类器算法实现
def classify0(inx, dataset, labels, k):
        # 计算距离
        dist = np.sum((inx - dataset)**2, axis=1)**0.5
        # k个最近的标签
        k_labels = [labels[index] for index in dist.argsort()[0: k]]
        # 出现次数最多的标签即为最终类别
        label = collections.Counter(k_labels).most_common(1)[0][0]
        return label


if __name__ == '__main__':
        #建数据集
        group, labels = createDataSet()
        # 测试集
        test = [101, 20]
        # kNN分类
        test_class = classify0(test, group, labels, 3)
        # 打印分类结果
        print(test_class)