#-*- coding:utf-8 -*-
# 可以使用上面提到的各种推荐系统算法
# from surprise import SVD
# from surprise import Dataset, print_perf, Reader
# from surprise.model_selection import GridSearchCV
# from surprise.model_selection import cross_validate
# import os
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# # 默认载入movielens数据集
# data = Dataset.load_builtin('ml-100k')
# # k折交叉验证(k=3),此方法现已弃用
# # data.split(n_folds=3)
# # 试一把SVD矩阵分解
# algo = SVD()
# # 在数据集上测试一下效果
# perf = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3)
# #输出结果
# print_perf(perf)

# # 指定文件所在路径
# file_path = os.path.expanduser('data.csv')
# # 告诉文本阅读器，文本的格式是怎么样的
# reader = Reader(line_format='user item rating', sep=',')
# # 加载数据
# data = Dataset.load_from_file(file_path, reader=reader)
# algo = SVD()
# # 在数据集上测试一下效果
# perf = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3)
# #输出结果
# print_perf(perf)

# 定义好需要优选的参数网格
# param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
#               'reg_all': [0.4, 0.6]}
# # 使用网格搜索交叉验证
# grid_search = GridSearchCV(SVD, param_grid, measures=['RMSE', 'FCP'], cv=3)
# # 在数据集上找到最好的参数
# data = Dataset.load_builtin('ml-100k')
# # pref = cross_validate(grid_search, data, cv=3)
# grid_search.fit(data)
# # 输出调优的参数组
# # 输出最好的RMSE结果
# print(grid_search.best_score)

# from surprise import Dataset, print_perf
# from surprise.model_selection import cross_validate
# data = Dataset.load_builtin('ml-100k')
# ### 使用NormalPredictor
# from surprise import NormalPredictor
# algo = NormalPredictor()
# perf = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3)
# print_perf(perf)
#
# ### 使用BaselineOnly
# from surprise import BaselineOnly
# algo = BaselineOnly()
# perf = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3)
# print_perf(perf)
#
# ### 使用基础版协同过滤
# from surprise import KNNBasic, evaluate
# algo = KNNBasic()
# perf = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3)
# print_perf(perf)
#
# ### 使用均值协同过滤
# from surprise import KNNWithMeans, evaluate
# algo = KNNWithMeans()
# perf = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3)
# print_perf(perf)
#
# ### 使用协同过滤baseline
# from surprise import KNNBaseline, evaluate
# algo = KNNBaseline()
# perf = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3)
# print_perf(perf)
#
# ### 使用SVD
# from surprise import SVD, evaluate
# algo = SVD()
# perf = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3)
# print_perf(perf)
#
# ### 使用SVD++
# from surprise import SVDpp, evaluate
# algo = SVDpp()
# perf = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3)
# print_perf(perf)
#
# ### 使用NMF
# from surprise import NMF
# algo = NMF()
# perf = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3)
# print_perf(perf)


# """
# 以下的程序段告诉大家如何在协同过滤算法建模以后，根据一个item取回相似度最高的item，主要是用到algo.get_neighbors()这个函数
# """
#
# from __future__ import (absolute_import, division, print_function,
#                         unicode_literals)
# import os
# import io
#
# from surprise import KNNBaseline
# from surprise import Dataset
#
#
# def read_item_names():
#     """
#     获取电影名到电影id 和 电影id到电影名的映射
#     """
#
#     file_name = (os.path.expanduser('~') +
#                  '/.surprise_data/ml-100k/ml-100k/u.item')
#     rid_to_name = {}
#     name_to_rid = {}
#     with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
#         for line in f:
#             line = line.split('|')
#             rid_to_name[line[0]] = line[1]
#             name_to_rid[line[1]] = line[0]
#
#     return rid_to_name, name_to_rid
#
#
# # 首先，用算法计算相互间的相似度
# data = Dataset.load_builtin('ml-100k')
# trainset = data.build_full_trainset()
# sim_options = {'name': 'pearson_baseline', 'user_based': False}
# algo = KNNBaseline(sim_options=sim_options)
# algo.fit(trainset)
#
# # 获取电影名到电影id 和 电影id到电影名的映射
# rid_to_name, name_to_rid = read_item_names()
#
# # Retieve inner id of the movie Toy Story
# toy_story_raw_id = name_to_rid['Toy Story (1995)']
# toy_story_inner_id = algo.trainset.to_inner_iid(toy_story_raw_id)
#
# # Retrieve inner ids of the nearest neighbors of Toy Story.
# toy_story_neighbors = algo.get_neighbors(toy_story_inner_id, k=10)
#
# # Convert inner ids of the neighbors into names.
# toy_story_neighbors = (algo.trainset.to_raw_iid(inner_id)
#                        for inner_id in toy_story_neighbors)
# toy_story_neighbors = (rid_to_name[rid]
#                        for rid in toy_story_neighbors)
#
# print()
# print('The 10 nearest neighbors of Toy Story are:')
# for movie in toy_story_neighbors:
#     print(movie)





import os
import io
from surprise import KNNBaseline
from surprise import Dataset

import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a， %d %b %Y %H:%M:%S')


# 训练推荐模型 步骤:1
def getSimModle():
    # 默认载入movielens数据集
    data = Dataset.load_builtin('ml-100k')
    trainset = data.build_full_trainset()
    #使用pearson_baseline方式计算相似度  False以item为基准计算相似度 本例为电影之间的相似度
    sim_options = {'name': 'pearson_baseline', 'user_based': False}
    ##使用KNNBaseline算法
    algo = KNNBaseline(sim_options=sim_options)
    #训练模型
    algo.fit(trainset)
    return algo


# 获取id到name的互相映射  步骤:2
def read_item_names():
    """
    获取电影名到电影id 和 电影id到电影名的映射
    """
    file_name = (os.path.expanduser('~') +
                 '/.surprise_data/ml-100k/ml-100k/u.item')
    rid_to_name = {}
    name_to_rid = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]
    return rid_to_name, name_to_rid


# 基于之前训练的模型 进行相关电影的推荐  步骤：3
def showSimilarMovies(algo, rid_to_name, name_to_rid):
    # 获得电影Toy Story (1995)的raw_id
    toy_story_raw_id = name_to_rid['Toy Story (1995)']
    logging.debug('raw_id=' + toy_story_raw_id)
    #把电影的raw_id转换为模型的内部id
    toy_story_inner_id = algo.trainset.to_inner_iid(toy_story_raw_id)
    logging.debug('inner_id=' + str(toy_story_inner_id))
    #通过模型获取推荐电影 这里设置的是10部
    toy_story_neighbors = algo.get_neighbors(toy_story_inner_id, 10)
    logging.debug('neighbors_ids=' + str(toy_story_neighbors))
    #模型内部id转换为实际电影id
    neighbors_raw_ids = [algo.trainset.to_raw_iid(inner_id) for inner_id in toy_story_neighbors]
    #通过电影id列表 或得电影推荐列表
    neighbors_movies = [rid_to_name[raw_id] for raw_id in neighbors_raw_ids]
    print('The 10 nearest neighbors of Toy Story are:')
    for movie in neighbors_movies:
        print(movie)


if __name__ == '__main__':
    # 获取id到name的互相映射
    rid_to_name, name_to_rid = read_item_names()

    # 训练推荐模型
    algo = getSimModle()

    ##显示相关电影
    showSimilarMovies(algo, rid_to_name, name_to_rid)