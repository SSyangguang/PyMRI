import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_validate
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import Lasso, LassoCV, LogisticRegressionCV, LogisticRegression
from sklearn.linear_model import ElasticNet, ElasticNetCV, enet_path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import auc, roc_curve

from utils import kernel
from mics import classifier_mics
'''
函数名尽量保持了和scikit-learn相同的函数名，便于理解函数的作用
没有写留一法实现的函数，如果要用留一法直接在K折交叉验证参数中将折数设置为样本个数即实现了留一法（scikit-learn官方文件推荐）
不推荐在网格搜索法中使用留一法，当待选参数较多时会让模型开销极大
'''




class lasso():
    '''LASSO特征选择的方法集锦，直接在class中选择是否进行交叉验证
        输入：
            X_train, X_test, y_train, y_test: 训练集和测试集的特征与标签
            feature_name: 特征名称，顺序和X的列必须对应
            path: 记录文件的存储路径，自行定义
            cv_val:布尔型，是否进行网格搜索交叉验证

        '''

    def __init__(self, X_train, X_test, y_train, y_test, feature_name, path, cv_val=True):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.name = feature_name
        self.cv_val = cv_val
        self.path = path

    def lasso(self, alpha, cv):
        '''使用LASSO进行特征选择，只进行一次，选择特征系数不为0的特征作为结果
           得到的结果包括特征选择后的训练集和测试集特征，同时还有特征名和权重，每个特征名有一个权重值，顺序是对应的
        输入：
            alpha: 参数alpha
            cv: int, 如果进行交叉验证，cv的折数

        输出：
            best_alpha（只有使用交叉验证时才有）: 最优lasso惩罚参数
            new_train_feature: 选择的训练集特征矩阵
            new_test_feature: 选择后的测试集特征矩阵
            new_feature_name: 选择后的特征名称
            feature_weight: 选择后特征对应的系数

        '''
        if self.cv_val is True:
            model_lasso = LassoCV(alphas=alpha, cv=cv)
            model_lasso.fit(self.X_train, self.y_train)
            coef = pd.Series(model_lasso.coef_)
            print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(
                sum(coef == 0)) + " variables")

            img_path = os.path.join(self.path, 'lassoCV')
            os.makedirs(img_path, exist_ok=True)

            # 交叉验证得到的最佳lasso惩罚参数
            best_alpha = model_lasso.alpha_
            print('-----------------------------')
            print('Best LASSO alpha:')
            print(best_alpha)

            # 将lasso中权重不为0的特征选择出来
            model = SelectFromModel(model_lasso, prefit=True)
            # 分别将训练集和测试集的特征使用上述lasso进行筛选
            X_new_train = model.transform(self.X_train)
            X_new_test = model.transform(self.X_test)
            # 所有特征的mask，保留的特征用True，被筛掉的特征用False
            mask = model.get_support()

            new_feature_name = []
            feature_weight = []

            # 根据mask将保留特征的名字和权重分别存储到
            for bool, feature, coef in zip(mask, self.name, coef):
                if bool:
                    new_feature_name.append(feature)
                    feature_weight.append(coef)

            # 将训练集和测试集的保留特征加上特征名
            new_train_feature = pd.DataFrame(data=X_new_train, columns=new_feature_name)
            new_test_feature = pd.DataFrame(data=X_new_test, columns=new_feature_name)
            feature_weight = pd.Series(feature_weight)

            return best_alpha, new_train_feature, new_test_feature, new_feature_name, feature_weight

        else:
            model_lasso = Lasso(alpha=alpha)
            model_lasso.fit(self.X_train, self.y_train)
            coef = pd.Series(model_lasso.coef_)
            print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(
                sum(coef == 0)) + " variables")

            img_path = os.path.join(self.path, 'lasso_only')
            os.makedirs(img_path, exist_ok=True)

            # 将lasso中权重不为0的特征选择出来
            model = SelectFromModel(model_lasso, prefit=True)
            # 分别将训练集和测试集的特征使用上述lasso进行筛选
            X_new_train = model.transform(self.X_train)
            X_new_test = model.transform(self.X_test)
            # 所有特征的mask，保留的特征用True，被筛掉的特征用False
            mask = model.get_support()

            new_feature_name = []
            feature_weight = []

            # 根据mask将保留特征的名字和权重分别存储到
            for bool, feature, coef in zip(mask, self.name, coef):
                if bool:
                    new_feature_name.append(feature)
                    feature_weight.append(coef)

            # 将训练集和测试集的保留特征加上特征名
            new_train_feature = pd.DataFrame(data=X_new_train, columns=new_feature_name)
            new_test_feature = pd.DataFrame(data=X_new_test, columns=new_feature_name)
            feature_weight = pd.Series(feature_weight)

            return new_train_feature, new_test_feature, new_feature_name, feature_weight

    def lasso_shuffle(self, shuffle_time, alpha_range, cv=10):
        '''通过多次循环，每次循环都将数据集进行打乱，最后统计每个特征出现的次数
        输入：
            shuffle_time: 进行shuffle循环的次数
            alpha_range: alpha的值，如果不进行网格搜索为int，如果进行网格搜索为list
            cv: 如果进行交叉验证的话，折数

        输出：
            new_train_feature: 特征选择后的训练集特征（其实这个和下面的特征矩阵不重要，最后还是要用索引重新对原始特征矩阵进行抽取）
            new_test_feature: 特征选择后的测试集特征
            select_feature_name: 选择出来的特征名
            select_feature_name_freq: 对应特征名，每个特征在多次shuffle循环中出现的次数
            feature_weight: 对应特征名，每个特征的系数
            select_feature_index: 对应特征名，每个特征在原始特征矩阵中的索引，可以在特征选择完成后直接进行矩阵特征的抽取

        '''
        # 将返回的值存入txt文件中
        lasso_txt = open(os.path.join(self.path, 'lasso_shuffle.txt'), 'w')
        lasso_txt.write('LASSO parameters set:\n')
        lasso_txt.write('\n---------------------------------------------\n')
        lasso_txt.write('Grid search: % s' % self.cv_val)
        lasso_txt.write('\nAlpha range: % s' % alpha_range)
        lasso_txt.write('\nShuffle time: % s' % shuffle_time)
        lasso_txt.write('\nGrid search cv-fold: % s' % cv)
        lasso_txt.write('\n---------------------------------------------\n')

        if self.cv_val is True:
            # 初始化权重为0，初始化特征列表为空
            coef_sum = 0
            select_list = []

            # 初始化最佳参数alpha
            alpha_list = []

            # 开始shuffle循环，每次都存储选择后的特征名
            for i in range(shuffle_time):
                # 将数据进行shuffle
                X, y = shuffle(self.X_train, self.y_train)
                kfold = StratifiedKFold(n_splits=cv, shuffle=False)

                model_lasso = LassoCV(alphas=alpha_range, cv=cv)
                model_lasso.fit(X, y)
                coef = pd.Series(model_lasso.coef_)
                print("% s th shuffle, Lasso picked " % i + str(
                      sum(coef != 0)) + " variables and eliminated the other " + str(
                      sum(coef == 0)) + " variables")

                # 交叉验证得到的最佳lasso惩罚参数
                alpha = model_lasso.alpha_
                alpha_list.append(alpha)
                print('best alpha value is % s' % alpha)
                # 将每一次循环的coef都进行相加
                coef_sum += model_lasso.coef_
                # 提取非零特征的mask
                model = SelectFromModel(model_lasso, prefit=True)
                # 所有特征的mask，保留的特征用True，被筛掉的特征用False
                mask = model.get_support()

                # 根据mask将保留特征的名字存储到select_list
                for bool, name in zip(mask, self.name):
                    if bool:
                        select_list.append(name)

            # 求全部特征的coef平均值
            coef_mean = coef_sum / shuffle_time

            # 每次的特征都存储在select_list中，然后统计每个特征出现的次数，存在feature_freq中，dict形式
            feature_freq = dict(zip(*np.unique(select_list, return_counts=True)))
            # 每次的alpha都存储在select_list中，然后统计每个特征出现的次数，存在feature_freq中，dict形式
            alpha_freq = dict(zip(*np.unique(alpha_list, return_counts=True)))

            # 按照特征出现的频率，从大到小进行排序，分别存储特征名和出现次数
            select_feature_name = []
            select_feature_name_freq = []

            for k in sorted(feature_freq, key=feature_freq.__getitem__, reverse=True):
                # 特征名相对应的顺序，将每个特征出现的次数存在select_feature_name_freq中
                select_feature_name_freq.append(feature_freq[k])
                # 将特征名存在select_feature_name中，list形式
                select_feature_name.append(k)

            # 获取lasso后特征的索引
            select_feature_index = []
            # 将lasso后特征的名字转为list
            name_list = list(select_feature_name)
            # 将原始所有特征的名字转为list
            all_name_list = list(self.name)
            # 获取特征选择后特征在原始特征list中的索引位置，将所有索引位置存在select_feature_index中
            for i in range(len(select_feature_name)):
                index = all_name_list.index(name_list[i])
                select_feature_index.append(index)

            # 按照alpha出现的频率，从大到小进行排序，分别存储alpha的大小和出现次数
            alpha_value = []
            alpha_value_freq = []

            for k in sorted(alpha_freq, key=alpha_freq.__getitem__, reverse=True):
                # alpha值相对应的顺序，将每个alpha值出现的次数存在alpha_value_freq中
                alpha_value_freq.append(alpha_freq[k])
                # 将alpha的值存在alpha_value中，list形式
                alpha_value.append(k)
                print('alpha value % s appeared % s times in the loop' % (k, alpha_freq[k]))

            # 通过索引将选择后的特征矩阵赋值给select_feature
            new_train_feature = self.X_train[:, select_feature_index]
            new_test_feature = self.X_test[:, select_feature_index]
            feature_weight = coef_mean[select_feature_index]

            # 将输出值存入txt文件
            lasso_txt.write('\nSelected feature index:\n')
            lasso_txt.write(str(select_feature_index))
            lasso_txt.write('\n---------------------------------------------\n')
            lasso_txt.write('\nSelected feature weight: \n')
            lasso_txt.write(str(feature_weight))
            lasso_txt.write('\n---------------------------------------------\n')
            lasso_txt.write('\nSelected feature name:\n')
            lasso_txt.write(str(select_feature_name))
            lasso_txt.write('\n---------------------------------------------\n')
            lasso_txt.write('\nSelected feature appearance frequency:\n')
            lasso_txt.write(str(select_feature_name_freq))
            lasso_txt.write('\n---------------------------------------------\n')

            return new_train_feature, new_test_feature, select_feature_name, \
                   select_feature_name_freq, feature_weight, select_feature_index

        else:
            # 初始化权重为0，初始化特征列表为空
            coef_sum = 0
            select_list = []

            # 开始shuffle循环，每次都存储选择后的特征名
            for i in range(shuffle_time):
                # 将数据进行shuffle
                X, y = shuffle(self.X_train, self.y_train)

                model_lasso = Lasso(alpha=alpha_range)
                model_lasso.fit(X, y)
                coef = pd.Series(model_lasso.coef_)
                print("% s th shuffle, Lasso picked " % i + str(
                      sum(coef != 0)) + " variables and eliminated the other " + str(
                      sum(coef == 0)) + " variables")

                # 将每一次循环的coef都进行相加
                coef_sum += model_lasso.coef_
                # 提取非零特征的mask
                model = SelectFromModel(model_lasso, prefit=True)
                # 所有特征的mask，保留的特征用True，被筛掉的特征用False
                mask = model.get_support()

                # 根据mask将保留特征的名字存储到select_list
                for bool, name in zip(mask, self.name):
                    if bool:
                        select_list.append(name)

            # 求全部特征的coef平均值
            coef_mean = coef_sum / shuffle_time

            # 每次的特征都存储在select_list中，然后统计每个特征出现的次数，存在feature_freq中，dict形式
            feature_freq = dict(zip(*np.unique(select_list, return_counts=True)))

            # 按照特征出现的频率，从大到小进行排序，分别存储特征名和出现次数
            select_feature_name = []
            select_feature_name_freq = []

            for k in sorted(feature_freq, key=feature_freq.__getitem__, reverse=True):
                # 特征名相对应的顺序，将每个特征出现的次数存在select_feature_name_freq中
                select_feature_name_freq.append(feature_freq[k])
                # 将特征名存在select_feature_name中，list形式
                select_feature_name.append(k)

            # 获取lasso后特征的索引
            select_feature_index = []
            # 将lasso后特征的名字转为list
            name_list = list(select_feature_name)
            # 将原始所有特征的名字转为list
            all_name_list = list(self.name)
            # 获取特征选择后特征在原始特征list中的索引位置，将所有索引位置存在select_feature_index中
            for i in range(len(select_feature_name)):
                index = all_name_list.index(name_list[i])
                select_feature_index.append(index)

            # 通过索引将选择后的特征矩阵赋值给select_feature
            new_train_feature = self.X_train[:, select_feature_index]
            new_test_feature = self.X_test[:, select_feature_index]
            feature_weight = coef_mean[select_feature_index]

            # 将输出值存入txt文件
            lasso_txt.write('\nSelected feature index:\n')
            lasso_txt.write(str(select_feature_index))
            lasso_txt.write('\n---------------------------------------------\n')
            lasso_txt.write('\nSelected feature weight: \n')
            lasso_txt.write(str(feature_weight))
            lasso_txt.write('\n---------------------------------------------\n')
            lasso_txt.write('\nSelected feature name:\n')
            lasso_txt.write(str(select_feature_name))
            lasso_txt.write('\n---------------------------------------------\n')
            lasso_txt.write('\nSelected feature appearance frequency:\n')
            lasso_txt.write(str(select_feature_name_freq))
            lasso_txt.write('\n---------------------------------------------\n')

            return new_train_feature, new_test_feature, select_feature_name, \
                   select_feature_name_freq, feature_weight, select_feature_index

    def logis_lasso(self, alpha, cv):
        '''使用logistic LASSO进行特征选择，可以选择是否使用交叉验证选择惩罚参数alpha
           得到的结果包括特征选择后的训练集和测试集特征，同时还有特征名和权重，每个特征名有一个权重值，顺序是对应的
        输入：
            alpha:  惩罚参数，这里因为是LASSO所以就相当于是alpha
            cv：如果进行交叉验证，次数

        输出：
            best alpha（只有使用交叉验证时才有）: 最优lasso惩罚参数
            new_train_feature: 训练集特征选择后的特征矩阵
            new_train_feature: 测试集特征选择后的特征矩阵
            new_feature_name: 特征选择后的特征名称
            feature_weight: 选择后每个特征对应的权重

        '''
        if self.cv_val is True:
            logis_lasso = LogisticRegressionCV(Cs=alpha, cv=cv, penalty='l1')
            logis_lasso.fit(self.X_train, self.y_train)
            coef = pd.Series(np.ravel(logis_lasso.coef_))
            print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(
                sum(coef == 0)) + " variables")

            img_path = os.path.join(self.path, 'lassoCV')
            os.makedirs(img_path, exist_ok=True)

            # 交叉验证得到的最佳lasso惩罚参数
            best_alpha = logis_lasso.Cs_
            print('-----------------------------')
            print('Best LASSO alpha:')
            print(best_alpha)

            # 将lasso中权重不为0的特征选择出来
            model = SelectFromModel(logis_lasso, prefit=True)
            # 分别将训练集和测试集的特征使用上述lasso进行筛选
            X_new_train = model.transform(self.X_train)
            X_new_test = model.transform(self.X_test)
            # 所有特征的mask，保留的特征用True，被筛掉的特征用False
            mask = model.get_support()

            new_feature_name = []
            feature_weight = []

            # 根据mask将保留特征的名字和权重分别存储到
            for bool, feature, coef in zip(mask, self.name, coef):
                if bool:
                    new_feature_name.append(feature)
                    feature_weight.append(coef)

            # 将训练集和测试集的保留特征加上特征名
            new_train_feature = pd.DataFrame(data=X_new_train, columns=new_feature_name)
            new_test_feature = pd.DataFrame(data=X_new_test, columns=new_feature_name)
            feature_weight = pd.Series(feature_weight)

            return best_alpha, new_train_feature, new_test_feature, new_feature_name, feature_weight

        else:
            logis_lasso = LogisticRegression(C=alpha, penalty='l1')
            logis_lasso.fit(self.X_train, self.y_train)
            coef = pd.Series(logis_lasso.coef_)
            print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(
                sum(coef == 0)) + " variables")

            img_path = os.path.join(self.path, 'lasso_only')
            os.makedirs(img_path, exist_ok=True)

            # 将lasso中权重不为0的特征选择出来
            model = SelectFromModel(logis_lasso, prefit=True)
            # 分别将训练集和测试集的特征使用上述lasso进行筛选
            X_new_train = model.transform(self.X_train)
            X_new_test = model.transform(self.X_test)
            # 所有特征的mask，保留的特征用True，被筛掉的特征用False
            mask = model.get_support()

            new_feature_name = []
            feature_weight = []

            # 根据mask将保留特征的名字和权重分别存储到
            for bool, feature, coef in zip(mask, self.name, coef):
                if bool:
                    new_feature_name.append(feature)
                    feature_weight.append(coef)

            # 将训练集和测试集的保留特征加上特征名
            new_train_feature = pd.DataFrame(data=X_new_train, columns=new_feature_name)
            new_test_feature = pd.DataFrame(data=X_new_test, columns=new_feature_name)
            feature_weight = pd.Series(feature_weight)

            return new_train_feature, new_test_feature, new_feature_name, feature_weight

    def logis_lasso_shuffle(self, alpha_range, shuffle_time=100, cv=10):
        '''使用logistic lasso进行特征选择，通过多次循环，每次循环都将数据集进行打乱，最后统计每个特征出现的次数
        输入：
            shuffle_time: 进行shuffle循环的次数
            alpha_range: alpha的值，如果不进行网格搜索为int，如果进行网格搜索为list
            cv: 如果进行交叉验证的话，折数

        输出：
            new_train_feature: 特征选择后的训练集特征（其实这个和下面的特征矩阵不重要，最后还是要用索引重新对原始特征矩阵进行抽取）
            new_test_feature: 特征选择后的测试集特征
            select_feature_name: 选择出来的特征名
            select_feature_name_freq: 对应特征名，每个特征在多次shuffle循环中出现的次数
            feature_weight: 对应特征名，每个特征的系数
            select_feature_index: 对应特征名，每个特征在原始特征矩阵中的索引，可以在特征选择完成后直接进行矩阵特征的抽取

        '''
        # 将返回的值存入txt文件中
        lasso_txt = open(os.path.join(self.path, 'logistic lasso_shuffle.txt'), 'w')
        lasso_txt.write('LASSO parameters set:\n')
        lasso_txt.write('\n---------------------------------------------\n')
        lasso_txt.write('Grid search: % s' % self.cv_val)
        lasso_txt.write('\nAlpha range: % s' % alpha_range)
        lasso_txt.write('\nShuffle time: % s' % shuffle_time)
        lasso_txt.write('\nGrid search cv-fold: % s' % cv)
        lasso_txt.write('\n---------------------------------------------\n')

        if self.cv_val is True:
            # 初始化权重为0，初始化特征列表为空
            coef_sum = 0
            select_list = []

            # 初始化最佳参数alpha
            alpha_list = []

            # 开始shuffle循环，每次都存储选择后的特征名
            for i in range(shuffle_time):
                # 将数据进行shuffle
                X, y = shuffle(self.X_train, self.y_train)
                kfold = StratifiedKFold(n_splits=cv, shuffle=False)

                model_lasso = LogisticRegressionCV(Cs=alpha_range, cv=cv, penalty='l1')
                model_lasso.fit(X, y)
                coef = pd.Series(np.ravel(model_lasso.coef_))
                print("% s th shuffle, Lasso picked " % i + str(
                      sum(coef != 0)) + " variables and eliminated the other " + str(
                      sum(coef == 0)) + " variables")

                # 交叉验证得到的最佳lasso惩罚参数
                alpha = model_lasso.Cs_
                alpha_list.append(alpha)
                print('best alpha value is % s' % alpha)
                # 将每一次循环的coef都进行相加
                coef_sum += model_lasso.coef_
                # 提取非零特征的mask
                model = SelectFromModel(model_lasso, prefit=True)
                # 所有特征的mask，保留的特征用True，被筛掉的特征用False
                mask = model.get_support()

                # 根据mask将保留特征的名字存储到select_list
                for bool, name in zip(mask, self.name):
                    if bool:
                        select_list.append(name)

            # 求全部特征的coef平均值
            coef_mean = coef_sum / shuffle_time

            # 每次的特征都存储在select_list中，然后统计每个特征出现的次数，存在feature_freq中，dict形式
            feature_freq = dict(zip(*np.unique(select_list, return_counts=True)))
            # 每次的alpha都存储在select_list中，然后统计每个特征出现的次数，存在feature_freq中，dict形式
            alpha_freq = dict(zip(*np.unique(alpha_list, return_counts=True)))

            # 按照特征出现的频率，从大到小进行排序，分别存储特征名和出现次数
            select_feature_name = []
            select_feature_name_freq = []

            for k in sorted(feature_freq, key=feature_freq.__getitem__, reverse=True):
                # 特征名相对应的顺序，将每个特征出现的次数存在select_feature_name_freq中
                select_feature_name_freq.append(feature_freq[k])
                # 将特征名存在select_feature_name中，list形式
                select_feature_name.append(k)

            # 获取lasso后特征的索引
            select_feature_index = []
            # 将lasso后特征的名字转为list
            name_list = list(select_feature_name)
            # 将原始所有特征的名字转为list
            all_name_list = list(self.name)
            # 获取特征选择后特征在原始特征list中的索引位置，将所有索引位置存在select_feature_index中
            for i in range(len(select_feature_name)):
                index = all_name_list.index(name_list[i])
                select_feature_index.append(index)

            # 按照alpha出现的频率，从大到小进行排序，分别存储alpha的大小和出现次数
            alpha_value = []
            alpha_value_freq = []

            for k in sorted(alpha_freq, key=alpha_freq.__getitem__, reverse=True):
                # alpha值相对应的顺序，将每个alpha值出现的次数存在alpha_value_freq中
                alpha_value_freq.append(alpha_freq[k])
                # 将alpha的值存在alpha_value中，list形式
                alpha_value.append(k)
                print('alpha value % s appeared % s times in the loop' % (k, alpha_freq[k]))

            # 通过索引将选择后的特征矩阵赋值给select_feature
            new_train_feature = self.X_train[:, select_feature_index]
            new_test_feature = self.X_test[:, select_feature_index]
            feature_weight = coef_mean[select_feature_index]

            # 将输出值存入txt文件
            lasso_txt.write('\nSelected feature index:\n')
            lasso_txt.write(str(select_feature_index))
            lasso_txt.write('\n---------------------------------------------\n')
            lasso_txt.write('\nSelected feature weight: \n')
            lasso_txt.write(str(feature_weight))
            lasso_txt.write('\n---------------------------------------------\n')
            lasso_txt.write('\nSelected feature name:\n')
            lasso_txt.write(str(select_feature_name))
            lasso_txt.write('\n---------------------------------------------\n')
            lasso_txt.write('\nSelected feature appearance frequency:\n')
            lasso_txt.write(str(select_feature_name_freq))
            lasso_txt.write('\n---------------------------------------------\n')

            return new_train_feature, new_test_feature, select_feature_name, \
                   select_feature_name_freq, feature_weight, select_feature_index

        else:
            # 初始化权重为0，初始化特征列表为空
            coef_sum = 0
            select_list = []

            # 开始shuffle循环，每次都存储选择后的特征名
            for i in range(shuffle_time):
                # 将数据进行shuffle
                X, y = shuffle(self.X_train, self.y_train)

                model_lasso = LogisticRegression(C=alpha_range, penalty='l1')
                model_lasso.fit(X, y)
                coef = pd.Series(np.ravel(model_lasso.coef_))
                print("% s th shuffle, Lasso picked " % i + str(
                    sum(coef != 0)) + " variables and eliminated the other " + str(
                    sum(coef == 0)) + " variables")

                # 将每一次循环的coef都进行相加
                coef_sum += model_lasso.coef_
                # 提取非零特征的mask
                model = SelectFromModel(model_lasso, prefit=True)
                # 所有特征的mask，保留的特征用True，被筛掉的特征用False
                mask = model.get_support()

                # 根据mask将保留特征的名字存储到select_list
                for bool, name in zip(mask, self.name):
                    if bool:
                        select_list.append(name)

            # 求全部特征的coef平均值
            coef_mean = coef_sum / shuffle_time

            # 每次的特征都存储在select_list中，然后统计每个特征出现的次数，存在feature_freq中，dict形式
            feature_freq = dict(zip(*np.unique(select_list, return_counts=True)))

            # 按照特征出现的频率，从大到小进行排序，分别存储特征名和出现次数
            select_feature_name = []
            select_feature_name_freq = []

            for k in sorted(feature_freq, key=feature_freq.__getitem__, reverse=True):
                # 特征名相对应的顺序，将每个特征出现的次数存在select_feature_name_freq中
                select_feature_name_freq.append(feature_freq[k])
                # 将特征名存在select_feature_name中，list形式
                select_feature_name.append(k)

            # 获取lasso后特征的索引
            select_feature_index = []
            # 将lasso后特征的名字转为list
            name_list = list(select_feature_name)
            # 将原始所有特征的名字转为list
            all_name_list = list(self.name)
            # 获取特征选择后特征在原始特征list中的索引位置，将所有索引位置存在select_feature_index中
            for i in range(len(select_feature_name)):
                index = all_name_list.index(name_list[i])
                select_feature_index.append(index)

            # 通过索引将选择后的特征矩阵赋值给select_feature
            new_train_feature = self.X_train[:, select_feature_index]
            new_test_feature = self.X_test[:, select_feature_index]
            feature_weight = coef_mean[select_feature_index]

            # 将输出值存入txt文件
            lasso_txt.write('\nSelected feature index:\n')
            lasso_txt.write(str(select_feature_index))
            lasso_txt.write('\n---------------------------------------------\n')
            lasso_txt.write('\nSelected feature weight: \n')
            lasso_txt.write(str(feature_weight))
            lasso_txt.write('\n---------------------------------------------\n')
            lasso_txt.write('\nSelected feature name:\n')
            lasso_txt.write(str(select_feature_name))
            lasso_txt.write('\n---------------------------------------------\n')
            lasso_txt.write('\nSelected feature appearance frequency:\n')
            lasso_txt.write(str(select_feature_name_freq))
            lasso_txt.write('\n---------------------------------------------\n')

            return new_train_feature, new_test_feature, select_feature_name, \
                   select_feature_name_freq, feature_weight, select_feature_index


class elastic_net():
    '''elastic net用于特征选择，可以选择组特征
    输入：
        X_train: 输入的训练集特征矩阵
        X_test: 输入的测试集特征矩阵
        y_train: 输入的训练集标签
        y_test: 输入的测试集标签
        feature_name: 特征矩阵对应的特征名
        cv_val:布尔型，是否进行网格搜索交叉验证
        path: 结果存储的路径

    '''

    def __init__(self, X_train, X_test, y_train, y_test, feature_name, cv_val, path):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.name = feature_name
        self.cv_val = cv_val
        self.path = path

    def elastic_net(self, l1, alphas, cv):
        if self.cv_val is True:
            elas = ElasticNetCV(l1_ratio=l1, alphas=alphas, cv=cv)
            elas.fit(self.X_train, self.y_train)
            coef = pd.Series(elas.coef_)
            print("Elastic Net picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(
                sum(coef == 0)) + " variables")

            img_path = os.path.join(self.path, 'ElasticNetCV')
            os.makedirs(img_path, exist_ok=True)

            # 交叉验证得到的最佳lasso惩罚参数
            best_alpha = elas.alpha_
            best_l1_ratio = elas.l1_ratio_
            best_coef = elas.coef_
            best_alphas = elas.alphas_
            best_mse_path = elas.mse_path_

            print('-----------------------------')
            print('Best Elastic Net alpha:')
            print(best_alpha)

            # 将lasso中权重不为0的特征选择出来
            model = SelectFromModel(elas, prefit=True)
            # 分别将训练集和测试集的特征使用上述lasso进行筛选
            X_new_train = model.transform(self.X_train)
            X_new_test = model.transform(self.X_test)
            # print(X_new_test.shape)
            # print(model.get_support())
            # 所有特征的mask，保留的特征用True，被筛掉的特征用False
            mask = model.get_support()

            new_feature_name = []
            feature_weight = []

            # 根据mask将保留特征的名字和权重分别存储到
            for bool, feature, coef in zip(mask, self.name, coef):
                if bool:
                    new_feature_name.append(feature)
                    feature_weight.append(coef)

            # 将训练集和测试集的保留特征加上特征名
            new_train_feature = pd.DataFrame(data=X_new_train, columns=new_feature_name)
            new_test_feature = pd.DataFrame(data=X_new_test, columns=new_feature_name)
            feature_weight = pd.Series(feature_weight)

            return best_alpha, new_train_feature, new_test_feature, new_feature_name, feature_weight
        else:
            elas = ElasticNet(l1_ratio=l1, alpha=alphas)
            elas.fit(self.X_train, self.y_train)
            coef = pd.Series(elas.coef_)
            print("Elastic Net picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(
                sum(coef == 0)) + " variables")

            img_path = os.path.join(self.path, 'ElasticNetCV')
            os.makedirs(img_path, exist_ok=True)

            coef1 = elas.coef_
            sparse = elas.sparse_coef_
            # 将elas中权重不为0的特征选择出来
            model = SelectFromModel(elas, prefit=True)
            # 分别将训练集和测试集的特征使用上述lasso进行筛选
            X_new_train = model.transform(self.X_train)
            X_new_test = model.transform(self.X_test)
            # 所有特征的mask，保留的特征用True，被筛掉的特征用False
            mask = model.get_support()

            new_feature_name = []
            feature_weight = []

            # 根据mask将保留特征的名字和权重分别存储到
            for bool, feature, coef in zip(mask, self.name, coef):
                if bool:
                    new_feature_name.append(feature)
                    feature_weight.append(coef)

            # 将训练集和测试集的保留特征加上特征名
            new_train_feature = pd.DataFrame(data=X_new_train, columns=new_feature_name)
            new_test_feature = pd.DataFrame(data=X_new_test, columns=new_feature_name)
            feature_weight = pd.Series(feature_weight)

            return new_train_feature, new_test_feature, new_feature_name, feature_weight

    def elasticnet_shuffle(self, l1_range, alphas_range, shuffle_time=100, cv=10, freq_seq=False):
        '''通过多次shuffle循环来求特征的权重，最后通过每次循环被筛选特征出现的频率来选择
        输入:
            freq_seq: 是否根据每个特征出现的频率对特征排序，False使用原始特征顺序，只是抽调部分特征

        '''

        # 将返回的值存入txt文件中
        elas_txt = open(os.path.join(self.path, 'elastic net_shuffle.txt'), 'w')
        elas_txt.write('Elastic Net parameters set:\n')
        elas_txt.write('\n---------------------------------------------\n')
        elas_txt.write('Grid search: % s' % self.cv_val)
        elas_txt.write('\nL1_ratio range: % s' % l1_range)
        elas_txt.write('\nAlpha range: % s' % alphas_range)
        elas_txt.write('\nShuffle time: % s' % shuffle_time)
        elas_txt.write('\nGrid search cv-fold: % s' % cv)
        elas_txt.write('\n---------------------------------------------\n')

        if self.cv_val is True:
            # 初始化权重为0，初始化特征列表为空
            coef_sum = 0
            select_list = []

            # 初始化最佳参数alpha
            alpha_list = []

            # 开始shuffle循环，每次都存储选择后的特征名
            for i in range(shuffle_time):
                # 将数据进行shuffle
                X, y = shuffle(self.X_train, self.y_train)
                kfold = StratifiedKFold(n_splits=cv, shuffle=False)

                model_elas = ElasticNetCV(l1_ratio=l1_range, alphas=alphas_range, cv=cv)
                model_elas.fit(X, y)
                coef = pd.Series(model_elas.coef_)
                print("% s th shuffle, Elastic net picked " % i + str(
                      sum(coef != 0)) + " variables and eliminated the other " + str(
                      sum(coef == 0)) + " variables")

                # 交叉验证得到的最佳lasso惩罚参数
                alpha = model_elas.alpha_
                l1_ratio = model_elas.l1_ratio_
                alphas = model_elas.alphas_
                mse_path = model_elas.mse_path_

                alpha_list.append(alpha)
                print('best alpha value is % s' % alpha)
                # 将每一次循环的coef都进行相加
                coef_sum += model_elas.coef_
                # 提取非零特征的mask
                model = SelectFromModel(model_elas, prefit=True)
                # 所有特征的mask，保留的特征用True，被筛掉的特征用False
                mask = model.get_support()

                # 根据mask将保留特征的名字存储到select_list
                for bool, name in zip(mask, self.name):
                    if bool:
                        select_list.append(name)

            # 求全部特征的coef平均值，这里的平均值只是为了返回每个特征的权重均值，特征选择过程中不使用
            coef_mean = coef_sum / shuffle_time

            # 每次的特征都存储在select_list中，然后统计每个特征出现的次数，存在feature_freq中，dict形式
            feature_freq = dict(zip(*np.unique(select_list, return_counts=True)))
            # 每次的alpha都存储在select_list中，然后统计每个特征出现的次数，存在feature_freq中，dict形式
            alpha_freq = dict(zip(*np.unique(alpha_list, return_counts=True)))

            # 按照特征出现的频率，从大到小进行排序，分别存储特征名和出现次数
            select_feature_name = []
            select_feature_name_freq = []

            # 如果freq_seq为True，那么按照特征出现的频率为他们排序，否则按照原始顺序
            if freq_seq is True:
                for k in sorted(feature_freq, key=feature_freq.__getitem__, reverse=True):
                    # 特征名相对应的顺序，将每个特征出现的次数存在select_feature_name_freq中
                    select_feature_name_freq.append(feature_freq[k])
                    # 将特征名存在select_feature_name中，list形式
                    select_feature_name.append(k)
            elif freq_seq is False:
                for k in feature_freq:
                    # 特征名相对应的顺序，将每个特征出现的次数存在select_feature_name_freq中
                    select_feature_name_freq.append(feature_freq[k])
                    # 将特征名存在select_feature_name中，list形式
                    select_feature_name.append(k)

            # 获取lasso后特征的索引
            select_feature_index = []
            # 将lasso后特征的名字转为list
            name_list = list(select_feature_name)
            # 将原始所有特征的名字转为list
            all_name_list = list(self.name)
            # 获取特征选择后特征在原始特征list中的索引位置，将所有索引位置存在select_feature_index中
            for i in range(len(select_feature_name)):
                index = all_name_list.index(name_list[i])
                select_feature_index.append(index)

            # 按照alpha出现的频率，从大到小进行排序，分别存储alpha的大小和出现次数
            alpha_value = []
            alpha_value_freq = []

            for k in sorted(alpha_freq, key=alpha_freq.__getitem__, reverse=True):
                # alpha值相对应的顺序，将每个alpha值出现的次数存在alpha_value_freq中
                alpha_value_freq.append(alpha_freq[k])
                # 将alpha的值存在alpha_value中，list形式
                alpha_value.append(k)
                print('alpha value % s appeared % s times in the loop' % (k, alpha_freq[k]))

            # 通过索引将选择后的特征矩阵赋值给select_feature
            new_train_feature = self.X_train[:, select_feature_index]
            new_test_feature = self.X_test[:, select_feature_index]
            feature_weight = coef_mean[select_feature_index]

            # 将输出值存入txt文件
            elas_txt.write('\nSelected feature index:\n')
            elas_txt.write(str(select_feature_index))
            elas_txt.write('\n---------------------------------------------\n')
            elas_txt.write('\nSelected feature weight: \n')
            elas_txt.write(str(feature_weight))
            elas_txt.write('\n---------------------------------------------\n')
            elas_txt.write('\nSelected feature name:\n')
            elas_txt.write(str(select_feature_name))
            elas_txt.write('\n---------------------------------------------\n')
            elas_txt.write('\nSelected feature appearance frequency:\n')
            elas_txt.write(str(select_feature_name_freq))
            elas_txt.write('\n---------------------------------------------\n')

            return new_train_feature, new_test_feature, select_feature_name, \
                   select_feature_name_freq, feature_weight, select_feature_index

        else:
            # 初始化权重为0，初始化特征列表为空
            coef_sum = 0
            select_list = []

            # 开始shuffle循环，每次都存储选择后的特征名
            for i in range(shuffle_time):
                # 将数据进行shuffle
                X, y = shuffle(self.X_train, self.y_train)

                model_elas = ElasticNet(l1_ratio=l1_range, alpha=alphas_range)
                model_elas.fit(X, y)
                coef = pd.Series(model_elas.coef_)
                print("% s th shuffle, Elastic net picked " % i + str(
                    sum(coef != 0)) + " variables and eliminated the other " + str(
                    sum(coef == 0)) + " variables")

                # 绘制elastic net的路径
                # from itertools import cycle
                # alphas_enet, coefs_enet, _ = enet_path(X, y, eps=5e-3, l1_ratio=l1_range,
                #                                        fit_intercept=False)
                # plt.figure(1)
                # colors = cycle(['b', 'r', 'g', 'c', 'k'])
                # neg_log_alphas_enet = -np.log10(alphas_enet)
                # for coef_e in coefs_enet:
                #     l2 = plt.plot(neg_log_alphas_enet, coef_e)
                #
                # plt.xlabel('-Log(alpha)')
                # plt.ylabel('coefficients')
                # plt.xlim((0, 2.2))
                # plt.ylim((-0.1, 0.1))
                # plt.axis('tight')
                # plt.show()
                # 将每一次循环的coef都进行相加
                coef_sum += model_elas.coef_
                # 提取非零特征的mask
                model = SelectFromModel(model_elas, prefit=True)
                # 所有特征的mask，保留的特征用True，被筛掉的特征用False
                mask = model.get_support()

                # 根据mask将保留特征的名字存储到select_list
                for bool, name in zip(mask, self.name):
                    if bool:
                        select_list.append(name)

            # 求全部特征的coef平均值
            coef_mean = coef_sum / shuffle_time

            # 每次的特征都存储在select_list中，然后统计每个特征出现的次数，存在feature_freq中，dict形式
            feature_freq = dict(zip(*np.unique(select_list, return_counts=True)))

            # 按照特征出现的频率，从大到小进行排序，分别存储特征名和出现次数
            select_feature_name = []
            select_feature_name_freq = []

            # 如果freq_seq为True，那么按照特征出现的频率为他们排序，否则按照原始顺序
            if freq_seq is True:
                for k in sorted(feature_freq, key=feature_freq.__getitem__, reverse=True):
                    # 特征名相对应的顺序，将每个特征出现的次数存在select_feature_name_freq中
                    select_feature_name_freq.append(feature_freq[k])
                    # 将特征名存在select_feature_name中，list形式
                    select_feature_name.append(k)
            elif freq_seq is False:
                for k in feature_freq:
                    # 特征名相对应的顺序，将每个特征出现的次数存在select_feature_name_freq中
                    select_feature_name_freq.append(feature_freq[k])
                    # 将特征名存在select_feature_name中，list形式
                    select_feature_name.append(k)

            # 获取lasso后特征的索引
            select_feature_index = []
            # 将lasso后特征的名字转为list
            name_list = list(select_feature_name)
            # 将原始所有特征的名字转为list
            all_name_list = list(self.name)
            # 获取特征选择后特征在原始特征list中的索引位置，将所有索引位置存在select_feature_index中
            for i in range(len(select_feature_name)):
                index = all_name_list.index(name_list[i])
                select_feature_index.append(index)

            # 通过索引将选择后的特征矩阵赋值给select_feature
            new_train_feature = self.X_train[:, select_feature_index]
            new_test_feature = self.X_test[:, select_feature_index]
            feature_weight = coef_mean[select_feature_index]

            # 将输出值存入txt文件
            elas_txt.write('\nSelected feature index:\n')
            elas_txt.write(str(select_feature_index))
            elas_txt.write('\n---------------------------------------------\n')
            elas_txt.write('\nSelected feature weight: \n')
            elas_txt.write(str(feature_weight))
            elas_txt.write('\n---------------------------------------------\n')
            elas_txt.write('\nSelected feature name:\n')
            elas_txt.write(str(select_feature_name))
            elas_txt.write('\n---------------------------------------------\n')
            elas_txt.write('\nSelected feature appearance frequency:\n')
            elas_txt.write(str(select_feature_name_freq))
            elas_txt.write('\n---------------------------------------------\n')

            return new_train_feature, new_test_feature, select_feature_name, \
                   select_feature_name_freq, feature_weight, select_feature_index


class SVM():
    '''支持向量机进行分类的方法集锦，包括普通SVM, shuffle_SVM, nested SVM
    输入：
        X: 输入的特征矩阵
        y: 特征矩阵对应的标签
        path: 结果存储的路径

    属性：
        weight: 每个特征的SVM权重，因此长度和特征数量相同，list形式（注意该属性只有在核函数为linear时才有效）
    '''
    # 初始化类属性SVM特征权重
    weight = 0

    def __init__(self, X, y, path):
        self.X = X
        self.y = y
        self.path = path

    def svm_only(self, kernel='linear', ratio=0.5, gamma=0.1, C=10, cv=3, gridsearch=True):
        '''进行单次SVM，可以选择使用网格搜索法寻找最优参数
        输入：
            kernel: 核函数选择
            ratio: 训练集和测试集的比例，默认为0.5
            gamma: 超参数gamma（RBF专用），如果选择网格搜索法应该使用list，如果不使用参数搜索为int
            C: 超参数C，如果选择了网格搜索法应该使用list，如果不使用参数搜索为int
            cv: 交叉验证的次数，如果进行交叉验证网格搜索法，交叉验证的折数
            gridsearch: 布尔型，是否使用网格搜索法寻找SVM最佳超参数

        输出：
            best_para: dict型，如果进行网格搜索法，得到的最佳参数
            pred_train: 训练集预测结果
            y_score_train: 训练集预测结果的概率
            pred_test: 测试集预测结果
            y_score_test: 测试集预测结果的概率

        '''
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=ratio, stratify=self.y)

        if gridsearch is True:
            svm = SVC(kernel=kernel, gamma=gamma, C=C, probability=True)
            para = {
                    'gamma': gamma,
                    'C': C,
                    }
            grid = GridSearchCV(svm, para, n_jobs=1, verbose=1, scoring='accuracy', cv=cv)
            grid.fit(X_train, y_train)
            pred_train = grid.predict(X_train)
            pred_test = grid.predict(X_test)
            y_score_train = grid.predict_proba(X_train)
            y_score_test = grid.predict_proba(X_test)
            best_para = grid.best_params_
            # 输出SVM最佳参数
            print('SVM CV Best score: %0.3f' % grid.best_score_)
            print('SVM CV Best parameters set:')
            print('-------------------------------------------')
            for param_name in sorted(best_para.keys()):
                print('\t%s: %r' % (param_name, best_para[param_name]))

            # 将SVM的特征权重存储在类属性weight中
            if kernel == 'linear':
                weight = svm.coef_
            else:
                print('SVM coefficient is only available when using linear kernel function.')

        else:
            svm = SVC(kernel=kernel, gamma=gamma, C=C, probability=True)
            svm.fit(X_train, y_train)
            pred_train = svm.predict(X_train)
            pred_test = svm.predict(X_test)
            y_score_train = svm.predict_proba(X_train)
            y_score_test = svm.predict_proba(X_test)
            best_para = {'gamma': gamma, 'C': C}

            # 将SVM的特征权重存储在类属性weight中
            if kernel == 'linear':
                weight = svm.coef_
            else:
                print('SVM coefficient is only available when using linear kernel function.')

        return pred_train, y_score_train, pred_test, y_score_test, best_para

    def svm_shuffle(self, outer, para, svm_metrics, shuffle_time=100):
        '''SVM不进行超参数网格搜索，直接指定超参数，然后使用该模型对数据进行多次shuffle，最后取平均结果
           该函数中绘制ROC的方法中，fpr不是真正的fpr，而是自定义的等差数列，然后根据真实tpr和fpr的趋势来进行插值，获得tpr
           这样绘制的ROC是对的，不过最终的AUC计算是根据插值后的ROC计算的，和真实的AUC有微小误差，不过无妨

        输入:
            outer: 每一次shuffle进行交叉验证时，交叉验证的折数
            para: dict型，SVM的参数，包括：
                  kernel: 目前仅支持linear, rbf
                  C: 惩罚参数，linear和rbf都有
                  gamma: 如果使用rbf核函数，则有这个参数
            svm_metrics: list型，SVM输出结果后需要计算的指标，目前支持accuracy, precision, recall, f1, sensitivity, specificity
                         必须写着完整名，不能用缩写
            shuffle_time: 进行shuffle的次数

        输出：
            train_means: dict型，键值对应于svm_metrics定义的指标，对应的值为list型，具体是shuffle_time次数的训练集中对应指标
                        在交叉验证过程中的平均值，该dict返回后可以通过mean来求总体的平均值。
            train_std: 和上面的dict类似，不同的是计算的是所有shuffle的标准差而不是均值
            test_means: 和上面的dict类似，不用的是计算的是测试集中的均值
            test_std: 和上面的dict类似，计算的是测试集中的标准差
            roc_dict: dict型，返回的是与绘制ROC相关的list，包含：
                tpr_train: list，训练集中每次shuffle的tpr交叉验证平均值
                tpr_test: 测试集中每次shuffle的tpr交叉验证平均值
                tpr_list_train: 二维list，训练集每次shuffle交叉验证每折的tpr都会存储，一次shuffle一个list
                tpr_list_test: 二维list，测试集每次shuffle交叉验证每折的tpr都会存储，一次shuffle一个list
                fpr_train: list, 训练集中每次shuffle的tpr交叉验证平均值（其实是自定义长度的等差数列）
                fpr_test: list, 测试集中每次shuffle的tpr交叉验证平均值（其实是自定义长度的等差数列）
                auc_list_train: list, 记录了训练集每次shuffle计算得到的AUC
                auc_list_test: list, 记录了测试集每次shuffle计算得到的AUC
                auc_train: float, 训练集上所有shuffle的AUC的平均值
                auc_test: float, 测试集上所有shuffle的AUC的平均值

            前四个dict主要是为了绘制shuffle和每种指标的关系图，mean用于绘制指标的曲线，std可以绘制标准差的变化区域

            roc_dict真正实用的是tpr_train, tpr_test, fpr_train, fpr_test，这四个list再各自做平均后就可以获取绘制ROC的所有参数，
            auc_list可以绘制shuffle和AUC的曲线图，其他的值用处不大，仅仅以防万一要用
        '''
        from mics import classifier_mics

        # 初始化SVM权重为0
        svm_weight = 0
        svm_weight_cv = 0

        # 将svm参数写入txt文档
        svm_shuffle_path = os.path.join(self.path, 'svm_shuffle')
        os.makedirs(svm_shuffle_path, exist_ok=True)
        svm_txt = open(os.path.join(self.path, 'svm_shuffle_result.txt'), 'w')
        svm_txt.write('Support Vector Machine Shuffle parameters set:\n')
        svm_txt.write('\n---------------------------------------------\n')
        svm_txt.write('Kernel type: % s' % para['kernel'])
        svm_txt.write('\nC value: % s' % para['C'])
        if para['kernel'] == 'rbf':
            svm_txt.write('\nGamma value: % s' % para['gamma'])
        svm_txt.write('\nShuffle time: % s' % shuffle_time)
        svm_txt.write('\nCross validation-fold: % s' % outer)
        svm_txt.write('\nsvm metrics: % s\n' % svm_metrics)
        svm_txt.write('\n---------------------------------------------\n')

        # 传入svm_metrics中的每个指标都初始化空的train和test的均值和方差list
        metrics_num = len(svm_metrics)
        for name in svm_metrics:
            exec('train_{}_means = []'.format(name))
            exec('train_{}_std = []'.format(name))
            exec('test_{}_means = []'.format(name))
            exec('test_{}_std = []'.format(name))

        shuffle_path = os.path.join(self.path, 'svm', 'shuffle')
        os.makedirs(shuffle_path, exist_ok=True)

        # 直接将fpr定义为等差数列
        meanfpr_outer_train = np.linspace(0, 1, 100)
        meanfpr_outer_test = np.linspace(0, 1, 100)
        # 将tpr和auc定义为空,最终tpr_outer_test和meanfpr_outer_test的长度相同,auc和shuffle的次数相同
        tpr_outer_train = []
        auc_list_train = []
        tpr_outer_test = []
        auc_list_test = []

        for i in range(shuffle_time):
            # 外嵌套每一折的分配方法
            outer_cv = StratifiedKFold(n_splits=outer, shuffle=True, random_state=i)

            # 根据svm模型的核函数来选择具体模型形式
            if para['kernel'] == 'rbf':
                svm = SVC(kernel=para['kernel'], C=para['C'], gamma=para['gamma'], probability=True)
            elif para['kernel'] == 'linear':
                svm = SVC(kernel=para['kernel'], C=para['C'], probability=True)

            # 内循环，计算每次内循环的平均tpr
            tpr_inner_train = []
            tpr_inner_test = []

            # 每一折的四大指标进行初始化,只初始化svm_metrics中要求给的
            for name in svm_metrics:
                exec('{}_inner_train = []'.format(name))
                exec('{}_inner_test = []'.format(name))

            for train, test in outer_cv.split(self.X, self.y):
                svm.fit(self.X[train], self.y[train])
                # 求SVM的输出结果
                pred_train = svm.predict(self.X[train])
                pred_test = svm.predict(self.X[test])
                prob_train = svm.predict_proba(self.X[train])
                prob_test = svm.predict_proba(self.X[test])

                # 如果使用的SVM核函数是linear则将权重进行累加
                if para['kernel'] == 'linear':
                    svm_weight_cv += np.ravel(svm.coef_)
                else:
                    print('SVM coefficient is only available when using linear kernel function.')

                # 计算四大指标
                mics = classifier_mics(self.y[train], pred_train, prob_train,
                                       self.y[test], pred_test, prob_test, 'svm_shuffle_result')
                accuracy_train, precision_train, recall_train, f1_train = mics.mics_sum_train()
                accuracy_test, precision_test, recall_test, f1_test = mics.mics_sum_test()
                sensitivity_train, sensitivity_test = mics.sensitivity()
                specificity_train, specificity_test = mics.specificity()

                # 虽然四大指标都算了,但是只向list中添加svm_metrics中要求给的
                for name in svm_metrics:
                    exec('{}_inner_train.append({}_train)'.format(name, name))
                    exec('{}_inner_test.append({}_test)'.format(name, name))

                # 计算fpr和tpr
                fpr_train, tpr_train, thres_train = roc_curve(self.y[train], prob_train[:, 1])
                fpr_test, tpr_test, thres_test = roc_curve(self.y[test], prob_test[:, 1])
                # 根据meanfpr_outer_test的长度，通过fpr和tpr的范围进行插值
                tpr_inner_train.append(np.interp(meanfpr_outer_train, fpr_train, tpr_train))
                tpr_inner_test.append(np.interp(meanfpr_outer_test, fpr_test, tpr_test))
                tpr_inner_train[-1][0] = 0.0
                tpr_inner_test[-1][0] = 0.0

            # 计算每一次shuffle交叉验证的SVM权重平均值
            svm_weight_cv /= outer
            # 将每一次shuffle的权重值相加
            svm_weight += svm_weight_cv

            # 计算每次shuffle时，每折tpr的平均值作为该次shuffle的tpr
            meantpr_inner_train = np.mean(tpr_inner_train, axis=0)
            meantpr_inner_test = np.mean(tpr_inner_test, axis=0)
            meantpr_inner_train[-1] = 1.0
            meantpr_inner_test[-1] = 1.0

            # 计算每次shuffle的auc并存储在zuc_list中
            mean_auc_train = auc(meanfpr_outer_train, meantpr_inner_train)
            mean_auc_test = auc(meanfpr_outer_test, meantpr_inner_test)
            auc_list_train.append(mean_auc_train)
            auc_list_test.append(mean_auc_test)

            # 计算完auc之后，将每一次shuffle的tpr放进tpr_outer_test中
            tpr_outer_train.append(meantpr_inner_train)
            tpr_outer_test.append(meantpr_inner_test)

            # 将外层嵌套循环的每种指标存储在list中
            for name in svm_metrics:
                # 存储训练过程中交叉验证每个指标的平均值
                exec('{}_inner_train = np.array({}_inner_train)'.format(name, name))
                exec("train_{}_means.append({}_inner_train.mean())".format(name, name))
                # 存储训练过程中交叉验证每个指标的标准差
                exec("train_{}_std.append({}_inner_train.std())".format(name, name))
                # 存储测试过程中交叉验证每个指标的平均值
                exec('{}_inner_test = np.array({}_inner_test)'.format(name, name))
                exec("test_{}_means.append({}_inner_test.mean())".format(name, name))
                # 存储测试过程中交叉验证每个指标的标准差
                exec("test_{}_std.append({}_inner_test.std())".format(name, name))

        meantpr_outer_train = np.mean(tpr_outer_train, axis=0)
        meantpr_outer_test = np.mean(tpr_outer_test, axis=0)
        final_auc_train = auc(meanfpr_outer_train, meantpr_outer_train)
        final_auc_test = auc(meanfpr_outer_test, meantpr_outer_test)

        # 计算所有shuffle后的SVM权重平均值，并将该平均值赋给类属性weight
        svm_weight /= shuffle_time
        SVM.weight = svm_weight

        # 为了简洁,将绘制ROC曲线有关的变量用一个dict来表示
        roc_dict = {}
        roc_dict['tpr_train'] = meantpr_outer_train
        roc_dict['tpr_test'] = meantpr_outer_test
        roc_dict['tpr_list_train'] = tpr_outer_train
        roc_dict['tpr_list_test'] = tpr_outer_test
        roc_dict['fpr_train'] = meanfpr_outer_train
        roc_dict['fpr_test'] = meanfpr_outer_test
        roc_dict['auc_list_train'] = auc_list_train
        roc_dict['auc_list_test'] = auc_list_test
        roc_dict['auc_train'] = final_auc_train
        roc_dict['auc_test'] = final_auc_test

        # 为了简洁，将训练、测试过程中的指标平均值和标准差以字典形式存储，再返回
        train_means = {}
        train_std = {}
        test_means = {}
        test_std = {}
        for name in svm_metrics:
            exec("train_means['{}'] = train_{}_means".format(name, name))
            exec("train_std['{}'] = train_{}_std".format(name, name))
            exec("test_means['{}'] = test_{}_means".format(name, name))
            exec("test_std['{}'] = test_{}_std".format(name, name))

        # 将输出存在txt文件中
        for name in svm_metrics:
            svm_txt.write('\n---------------------------------------------\n')
            exec("svm_txt.write('Train set {} mean value: % s' % np.mean(train_means['{}']))".format(name, name))
            svm_txt.write('\n')
            exec("svm_txt.write('Train set {} max value: % s' % np.max(train_means['{}']))".format(name, name))
            svm_txt.write('\n')
            exec("svm_txt.write('Train set {} min value: % s' % np.min(train_means['{}']))".format(name, name))

            svm_txt.write('\n---------------------------------------------\n')
            exec("svm_txt.write('Test set {} mean value: % s' % np.mean(test_means['{}']))".format(name, name))
            svm_txt.write('\n')
            exec("svm_txt.write('Test set {} max value: % s' % np.max(test_means['{}']))".format(name, name))
            svm_txt.write('\n')
            exec("svm_txt.write('Test set {} min value: % s' % np.min(test_means['{}']))".format(name, name))
            svm_txt.write('\n---------------------------------------------\n')

        svm_txt.write('\nTrain set AUC mean value: % s' % np.mean(roc_dict['auc_list_train']))
        svm_txt.write('\nTrain set AUC max value: % s' % np.max(roc_dict['auc_list_train']))
        svm_txt.write('\nTrain set AUC min value: % s' % np.min(roc_dict['auc_list_train']))
        svm_txt.write('\n---------------------------------------------\n')
        svm_txt.write('\nTest set AUC mean value: % s' % np.mean(roc_dict['auc_list_test']))
        svm_txt.write('\nTest set AUC max value: % s' % np.max(roc_dict['auc_list_test']))
        svm_txt.write('\nTest set AUC min value: % s' % np.min(roc_dict['auc_list_test']))

        # 存储SVM权重值
        svm_txt.write('\n---------------------------------------------\n')
        svm_txt.write('\nSVM weight: % s' % svm_weight)

        return train_means, train_std, test_means, test_std, roc_dict

    def svm_nested(self, para, svm_metrics, shuffle_time=100, inner=5, outer=10, log=True):
        '''SVM内外嵌套交叉验证法，然后使用该模型对数据进行多次shuffle，最后取平均结果
           该函数中绘制ROC的方法中，fpr不是真正的fpr，而是自定义的等差数列，然后根据真实tpr和fpr的趋势来进行插值，获得tpr
           这样绘制的ROC是对的，不过最终的AUC计算是根据插值后的ROC计算的，和真实的AUC有微小误差，不过无妨

        输入:
            outer: 每一次shuffle进行交叉验证时，交叉验证的折数
            inner: 内部网格搜索法交叉验证时，交叉验证的折数
            para: dict型，SVM的参数，包括：
                kernel: 目前仅支持linear, rbf
                C: 惩罚参数，linear和rbf都有
                gamma: 如果使用rbf核函数，则有这个参数
            svm_metrics: list型，SVM输出结果后需要计算的指标，目前支持accuracy, precision, recall, f1, sensitivity, specificity。
                         必须写着完整名，不能用缩写
            shuffle_time: 进行shuffle的次数
            log: bool型，是否将网格搜索法每一折的相信信息存入文件中

        输出：
            train_means: dict型，键值对应于svm_metrics定义的指标，对应的值为list型，具体是shuffle_time次数的训练集中对应指标
                        在交叉验证过程中的平均值，该dict返回后可以通过mean来求总体的平均值。
            train_std: 和上面的dict类似，不同的是计算的是所有shuffle的标准差而不是均值
            test_means: 和上面的dict类似，不用的是计算的是测试集中的均值
            test_std: 和上面的dict类似，计算的是测试集中的标准差
            roc_dict: dict型，返回的是与绘制ROC相关的list，包含：
                tpr_train: list，训练集中每次shuffle的tpr交叉验证平均值
                tpr_test: 测试集中每次shuffle的tpr交叉验证平均值
                tpr_list_train: 二维list，训练集每次shuffle交叉验证每折的tpr都会存储，一次shuffle一个list
                tpr_list_test: 二维list，测试集每次shuffle交叉验证每折的tpr都会存储，一次shuffle一个list
                fpr_train: list, 训练集中每次shuffle的tpr交叉验证平均值（其实是自定义长度的等差数列）
                fpr_test: list, 测试集中每次shuffle的tpr交叉验证平均值（其实是自定义长度的等差数列）
                auc_list_train: list, 记录了训练集每次shuffle计算得到的AUC
                auc_list_test: list, 记录了测试集每次shuffle计算得到的AUC
                auc_train: float, 训练集上所有shuffle的AUC的平均值
                auc_test: float, 测试集上所有shuffle的AUC的平均值

            前四个dict主要是为了绘制shuffle和每种指标的关系图，mean用于绘制指标的曲线，std可以绘制标准差的变化区域

            roc_dict真正实用的是tpr_train, tpr_test, fpr_train, fpr_test，这四个list再各自做平均后就可以获取绘制ROC的所有参数，
            auc_list可以绘制shuffle和AUC的曲线图，其他的值用处不大，仅仅以防万一要用
        '''
        from mics import classifier_mics

        # 将svm参数写入txt文档
        svm_shuffle_path = os.path.join(self.path, 'svm_nested')
        os.makedirs(svm_shuffle_path, exist_ok=True)
        svm_txt = open(os.path.join(self.path, 'svm_nested_result.txt'), 'w')
        svm_txt.write('Nested Support Vector Machine parameters set:\n')
        svm_txt.write('\n---------------------------------------------\n')
        svm_txt.write('Kernel type: % s' % para['kernel'])
        svm_txt.write('\nC value: % s' % para['C'])
        if para['kernel'] == 'rbf':
            svm_txt.write('\nGamma value: % s' % para['gamma'])
        svm_txt.write('\nShuffle time: % s' % shuffle_time)
        svm_txt.write('\nGrid Search Cross validation-fold: % s' % inner)
        svm_txt.write('\nCross validation-fold: % s' % outer)
        svm_txt.write('\nsvm metrics: % s\n' % svm_metrics)
        svm_txt.write('\n---------------------------------------------\n')

        # 传入scoring中的每个指标都初始化空的train和test的均值和方差list
        metrics_num = len(svm_metrics)
        for name in svm_metrics:
            exec('train_{}_means = []'.format(name))
            exec('train_{}_std = []'.format(name))
            exec('test_{}_means = []'.format(name))
            exec('test_{}_std = []'.format(name))

        shuffle_path = os.path.join(self.path, 'svm', 'nest_cv')
        os.makedirs(shuffle_path, exist_ok=True)

        # 直接将fpr定义为等差数列
        meanfpr_outer_train = np.linspace(0, 1, 100)
        meanfpr_outer_test = np.linspace(0, 1, 100)
        # 将tpr和auc定义为空,最终tpr_outer_test和meanfpr_outer_test的长度相同,auc和shuffle的次数相同
        tpr_outer_train = []
        auc_list_train = []
        tpr_outer_test = []
        auc_list_test = []

        for i in range(shuffle_time):
            # 内嵌套和外嵌套每一折的分配方法
            inner_cv = StratifiedKFold(n_splits=inner, shuffle=True, random_state=i)
            outer_cv = StratifiedKFold(n_splits=outer, shuffle=True, random_state=i)

            # 内循环，计算每次内循环的平均tpr
            tpr_inner_train = []
            tpr_inner_test = []

            # 每一折的四大指标进行初始化,只初始化svm_metrics中要求给的
            for name in svm_metrics:
                exec('{}_inner_train = []'.format(name))
                exec('{}_inner_test = []'.format(name))

            for train, test in outer_cv.split(self.X, self.y):

                svm = SVC(probability=True)
                # 网格搜索法
                grid = GridSearchCV(svm, para, scoring='accuracy', cv=inner_cv, refit=True,
                                    return_train_score=True)
                grid.fit(self.X[train], self.y[train])

                # 返回最佳参数
                best_para = grid.best_params_
                # 如果需要记录网格法每一折的结果，则记录以下指标
                if log is True:
                    test_means_log = grid.cv_results_['mean_test_score']
                    test_std_log = grid.cv_results_['std_test_score']
                    train_means_log = grid.cv_results_['mean_train_score']
                    train_std_log = grid.cv_results_['std_train_score']
                    cv_para_log = grid.cv_results_['params']
                    # 将每一折的结果记录写入文件中
                    logfile_name = os.path.join(shuffle_path, 'log.txt')
                    file_log = open(logfile_name, mode='a', encoding='utf-8')
                    file_log.write('-------------------------------------\n')
                    file_log.write('cv results:\n')
                    file_log.write('best parameters: %s\n' % cv_para_log)
                    file_log.write('mean test score: %s\n' % test_means_log)
                    file_log.write('std_test_score: %s\n' % test_std_log)
                    file_log.write('mean_train_score: %s\n' % train_means_log)
                    file_log.write('train_std_log: %s\n' % train_std_log)

                # 输出SVM最佳参数
                print('SVM CV Best score: %0.3f' % grid.best_score_)
                print('SVM CV Best parameters set:')
                print('-------------------------------------------')
                for param_name in sorted(best_para.keys()):
                    print('\t%s: %r' % (param_name, best_para[param_name]))

                # 将每一次shuffle的结果写入文件
                logfile_name = os.path.join(shuffle_path, 'log.txt')
                file_log = open(logfile_name, mode='a', encoding='utf-8')
                file_log.write('------------------------------------\n')
                file_log.write('%s th shuffle results: \n' % i)
                file_log.write('best parameters: %s\n' % best_para)

                # 求SVM的输出结果
                pred_train = grid.predict(self.X[train])
                pred_test = grid.predict(self.X[test])
                prob_train = grid.predict_proba(self.X[train])
                prob_test = grid.predict_proba(self.X[test])

                # 计算四大指标
                mics = classifier_mics(self.y[train], pred_train, prob_train,
                                       self.y[test], pred_test, prob_test, 'svm_nest_result')
                accuracy_train, precision_train, recall_train, f1_train = mics.mics_sum_train()
                accuracy_test, precision_test, recall_test, f1_test = mics.mics_sum_test()
                sensitivity_train, sensitivity_test = mics.sensitivity()
                specificity_train, specificity_test = mics.specificity()

                # 虽然四大指标都算了,但是只向list中添加svm_metrics中要求给的
                for name in svm_metrics:
                    exec('{}_inner_train.append({}_train)'.format(name, name))
                    exec('{}_inner_test.append({}_test)'.format(name, name))

                # 计算fpr和tpr
                fpr_train, tpr_train, thres_train = roc_curve(self.y[train], prob_train[:, 1])
                fpr_test, tpr_test, thres_test = roc_curve(self.y[test], prob_test[:, 1])
                # 根据meanfpr_outer_test的长度，通过fpr和tpr的范围进行插值
                tpr_inner_train.append(np.interp(meanfpr_outer_train, fpr_train, tpr_train))
                tpr_inner_test.append(np.interp(meanfpr_outer_test, fpr_test, tpr_test))
                tpr_inner_train[-1][0] = 0.0
                tpr_inner_test[-1][0] = 0.0

            # 外层嵌套交叉验证
            # cv_outer = cross_validate(clf, X, y, cv=outer_cv, scoring=svm_metrics, return_train_score=True)

            # 计算每次shuffle时，每折tpr的平均值作为该次shuffle的tpr
            meantpr_inner_train = np.mean(tpr_inner_train, axis=0)
            meantpr_inner_test = np.mean(tpr_inner_test, axis=0)
            meantpr_inner_train[-1] = 1.0
            meantpr_inner_test[-1] = 1.0

            # 计算每次shuffle的auc并存储在auc_list中
            mean_auc_train = auc(meanfpr_outer_train, meantpr_inner_train)
            mean_auc_test = auc(meanfpr_outer_test, meantpr_inner_test)
            auc_list_train.append(mean_auc_train)
            auc_list_test.append(mean_auc_test)

            # 计算完auc之后，将每一次shuffle的tpr放进tpr_outer_test中
            tpr_outer_train.append(meantpr_inner_train)
            tpr_outer_test.append(meantpr_inner_test)

            # 将外层嵌套循环的每种指标存储在list中
            for name in svm_metrics:
                # 存储训练过程中交叉验证每个指标的平均值
                exec('{}_inner_train = np.array({}_inner_train)'.format(name, name))
                exec("train_{}_means.append({}_inner_train.mean())".format(name, name))
                # 存储训练过程中交叉验证每个指标的标准差
                exec("train_{}_std.append({}_inner_train.std())".format(name, name))
                # 存储测试过程中交叉验证每个指标的平均值
                exec('{}_inner_test = np.array({}_inner_test)'.format(name, name))
                exec("test_{}_means.append({}_inner_test.mean())".format(name, name))
                # 存储测试过程中交叉验证每个指标的标准差
                exec("test_{}_std.append({}_inner_test.std())".format(name, name))

        meantpr_outer_train = np.mean(tpr_outer_train, axis=0)
        meantpr_outer_test = np.mean(tpr_outer_test, axis=0)
        final_auc_train = auc(meanfpr_outer_train, meantpr_outer_train)
        final_auc_test = auc(meanfpr_outer_test, meantpr_outer_test)

        # 为了简洁,将绘制ROC曲线有关的变量用一个dict来表示
        roc_dict = {}
        roc_dict['tpr_train'] = meantpr_outer_train
        roc_dict['tpr_test'] = meantpr_outer_test
        roc_dict['tpr_list_train'] = tpr_outer_train
        roc_dict['tpr_list_test'] = tpr_outer_test
        roc_dict['fpr_train'] = meanfpr_outer_train
        roc_dict['fpr_test'] = meanfpr_outer_test
        roc_dict['auc_list_train'] = auc_list_train
        roc_dict['auc_list_test'] = auc_list_test
        roc_dict['auc_train'] = final_auc_train
        roc_dict['auc_test'] = final_auc_test

        # 为了简洁，将训练、测试过程中的指标平均值和标准差以字典形式存储，再返回
        train_means = {}
        train_std = {}
        test_means = {}
        test_std = {}
        for name in svm_metrics:
            exec("train_means['{}'] = train_{}_means".format(name, name))
            exec("train_std['{}'] = train_{}_std".format(name, name))
            exec("test_means['{}'] = test_{}_means".format(name, name))
            exec("test_std['{}'] = test_{}_std".format(name, name))

        # 将输出存在txt文件中
        for name in svm_metrics:
            svm_txt.write('\n---------------------------------------------\n')
            exec("svm_txt.write('Train set {} mean value: % s' % np.mean(train_means['{}']))".format(name, name))
            svm_txt.write('\n')
            exec("svm_txt.write('Train set {} max value: % s' % np.max(train_means['{}']))".format(name, name))
            svm_txt.write('\n')
            exec("svm_txt.write('Train set {} min value: % s' % np.min(train_means['{}']))".format(name, name))

            svm_txt.write('\n---------------------------------------------\n')
            exec("svm_txt.write('Test set {} mean value: % s' % np.mean(test_means['{}']))".format(name, name))
            svm_txt.write('\n')
            exec("svm_txt.write('Test set {} max value: % s' % np.max(test_means['{}']))".format(name, name))
            svm_txt.write('\n')
            exec("svm_txt.write('Test set {} min value: % s' % np.min(test_means['{}']))".format(name, name))
            svm_txt.write('\n---------------------------------------------\n')

        svm_txt.write('\nTrain set AUC mean value: % s' % np.mean(roc_dict['auc_list_train']))
        svm_txt.write('\nTrain set AUC max value: % s' % np.max(roc_dict['auc_list_train']))
        svm_txt.write('\nTrain set AUC min value: % s' % np.min(roc_dict['auc_list_train']))
        svm_txt.write('\n---------------------------------------------\n')
        svm_txt.write('\nTest set AUC mean value: % s' % np.mean(roc_dict['auc_list_test']))
        svm_txt.write('\nTest set AUC max value: % s' % np.max(roc_dict['auc_list_test']))
        svm_txt.write('\nTest set AUC min value: % s' % np.min(roc_dict['auc_list_test']))

        return train_means, train_std, test_means, test_std, roc_dict

k
class mkl_svm():
    '''multi-kernel svm的实施方案，目前只有双模态的方案，还没有相好怎么让输入可以变成无限多的模态
    输入:
        X1: 第一个模态的特征矩阵
        X2: 第二个模态的特征矩阵
        y: 特征矩阵对应的标签

    '''
    def __init__(self, X1, X2, X3=None, y=None, mkl_path=None):
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        self.y = y
        self.path = mkl_path

    def mksvm2(self, para, kernel_dict, svm_metrics, shuffle_time=100, outer=10):
        '''多核SVM的实现，只适用于双模态，网格搜索法只对融合系数有效，SVM的参数需要提前设定好
        输入：
            outer: 每一次shuffle进行交叉验证时，交叉验证的折数
            shuffle_time: 进行shuffle的次数
            kernel_dict: dict型，包含的参数有：
                'kernel_type1': 第一个核的类型，支持linear, poly, rbf
                'kernel_type2': 第二个核的类型，支持linear, poly, rbf
                'kernel_weight1': 第一个核的融合系数
                'kernel_weight2': 第二个核的融合系数，注意kernel_weight1与kernel_weight2相加必须为1
            para: dict型，包含的参数有：
                'kernel1': 第一个核的参数，如果linear没有参数，poly型即degree，rbf型即位gamma
                'kernel2': 第二个核的参数，如果linear没有参数，poly型即degree，rbf型即位gamma
                'C': 支持向量机的参数C
            svm_metrics: list型，SVM输出结果后需要计算的指标，目前支持accuracy, precision, recall, f1, sensitivity, specificity。
                     必须写着完整名，不能用缩写

        输出：
        train_means: dict型，键值对应于svm_metrics定义的指标，对应的值为list型，具体是shuffle_time次数的训练集中对应指标
                    在交叉验证过程中的平均值，该dict返回后可以通过mean来求总体的平均值。
        train_std: 和上面的dict类似，不同的是计算的是所有shuffle的标准差而不是均值
        test_means: 和上面的dict类似，不用的是计算的是测试集中的均值
        test_std: 和上面的dict类似，计算的是测试集中的标准差
        roc_dict: dict型，返回的是与绘制ROC相关的list，包含：
            tpr_train: list，训练集中每次shuffle的tpr交叉验证平均值
            tpr_test: 测试集中每次shuffle的tpr交叉验证平均值
            tpr_list_train: 二维list，训练集每次shuffle交叉验证每折的tpr都会存储，一次shuffle一个list
            tpr_list_test: 二维list，测试集每次shuffle交叉验证每折的tpr都会存储，一次shuffle一个list
            fpr_train: list, 训练集中每次shuffle的tpr交叉验证平均值（其实是自定义长度的等差数列）
            fpr_test: list, 测试集中每次shuffle的tpr交叉验证平均值（其实是自定义长度的等差数列）
            auc_list_train: list, 记录了训练集每次shuffle计算得到的AUC
            auc_list_test: list, 记录了测试集每次shuffle计算得到的AUC
            auc_train: float, 训练集上所有shuffle的AUC的平均值
            auc_test: float, 测试集上所有shuffle的AUC的平均值

        前四个dict主要是为了绘制shuffle和每种指标的关系图，mean用于绘制指标的曲线，std可以绘制标准差的变化区域

        roc_dict真正实用的是tpr_train, tpr_test, fpr_train, fpr_test，这四个list再各自做平均后就可以获取绘制ROC的所有参数，
        auc_list可以绘制shuffle和AUC的曲线图，其他的值用处不大，仅仅以防万一要用
    '''
        # 将svm参数写入txt文档
        svm_shuffle_path = os.path.join(self.path, 'mkl_svm')
        os.makedirs(svm_shuffle_path, exist_ok=True)
        svm_txt = open(os.path.join(self.path, 'mkl_svm_result.txt'), 'w')
        svm_txt.write('Multi-Kernel Support Vector Machine parameters set:\n')
        svm_txt.write('C value: % s' % para['C'])
        svm_txt.write('\nShuffle time: % s' % shuffle_time)
        svm_txt.write('\nCross validation-fold: % s' % outer)
        svm_txt.write('\nsvm metrics: % s\n' % svm_metrics)
        svm_txt.write('\n---------------------------------------------\n')
        svm_txt.write('Kernel-1 parameter set:')
        svm_txt.write('\nKernel type: % s' % kernel_dict['kernel_type1'])
        svm_txt.write('\nKernel weight: % s' % kernel_dict['kernel_weight1'])
        svm_txt.write('\nKernel parameter: % s' % para['kernel1'])
        svm_txt.write('\n---------------------------------------------\n')
        svm_txt.write('Kernel-2 parameter set:\n')
        svm_txt.write('\nKernel type: % s' % kernel_dict['kernel_type2'])
        svm_txt.write('\nKernel weight: % s' % kernel_dict['kernel_weight2'])
        svm_txt.write('\nKernel parameter: % s' % para['kernel2'])

        # 初始化每次shuffle的的各类指标
        for name in svm_metrics:
            exec('train_{}_means = []'.format(name))
            exec('train_{}_std = []'.format(name))
            exec('test_{}_means = []'.format(name))
            exec('test_{}_std = []'.format(name))

        # 直接将fpr定义为等差数列
        meanfpr_shuffle_train = np.linspace(0, 1, 100)
        meanfpr_shuffle_test = np.linspace(0, 1, 100)
        # 将tpr和auc定义为空,最终tpr_outer_test和meanfpr_outer_test的长度相同,auc和shuffle的次数相同
        tpr_shuffle_train = []
        auc_list_train = []
        tpr_shuffle_test = []
        auc_list_test = []

        for i in range(shuffle_time):

            outer_cv = StratifiedKFold(n_splits=outer, shuffle=True, random_state=i)

            # 初始化交叉验证过程中，每一折的各类指标
            for name in svm_metrics:
                exec('{}_cv_train = []'.format(name))
                exec('{}_cv_test = []'.format(name))

            # 初始化交叉验证过程中
            tpr_fold_train = []
            tpr_fold_test = []
            # 每一次shuffle的交叉验证过程
            for train, test in outer_cv.split(self.X1, self.y):

                # 将数据分为训练集和测试集，mat1和mat2分别为训练集和测试集特征矩阵，mod1和mod2分别为两个模态
                mod1_mat1 = self.X1[train]
                mod1_mat2 = self.X1[test]
                mod2_mat1 = self.X2[train]
                mod2_mat2 = self.X2[test]
                y_train = self.y[train]
                y_test = self.y[test]

                # 调取两个核函数的融合系数
                weight1 = kernel_dict['kernel_weight1']
                weight2 = kernel_dict['kernel_weight2']

                # 计算核矩阵，先分别给两个模态定义实例
                kernel_mod1 = kernel(kernel_type=kernel_dict['kernel_type1'],
                                     kernel_para=para['kernel1'])
                kernel_mod2 = kernel(kernel_type=kernel_dict['kernel_type2'],
                                     kernel_para=para['kernel2'])

                # 计算mod1的训练集核矩阵与测试集核矩阵
                mod1_train = kernel_mod1.calckernel(mod1_mat1)
                mod1_test = kernel_mod1.calckernel(mod1_mat1, mod1_mat2)

                # 计算mod2的训练集核矩阵与测试集核矩阵
                mod2_train = kernel_mod2.calckernel(mod2_mat1)
                mod2_test = kernel_mod2.calckernel(mod2_mat1, mod2_mat2)

                # 进行双模态核矩阵融合
                train_kernel = weight1 * mod1_train + weight2 * mod2_train
                test_kernel = weight1 * mod1_test + weight2 * mod2_test

                # svm训练
                svm = SVC(kernel='precomputed', C=para['C'], probability=True)
                svm.fit(train_kernel, y_train)

                # 得到训练集和测试集上的输出
                pred_train = svm.predict(train_kernel)
                prob_train = svm.predict_proba(train_kernel)
                pred_test = svm.predict(test_kernel)
                prob_test = svm.predict_proba(test_kernel)

                mics = classifier_mics(y_train, pred_train, prob_train,
                                       y_test, pred_test, prob_test, path='mkl_svm')

                # 计算每一折的各类指标
                accuracy_train, precision_train, recall_train, f1_train = mics.mics_sum_train()
                accuracy_test, precision_test, recall_test, f1_test = mics.mics_sum_test()
                sensitivity_train, sensitivity_test = mics.sensitivity()
                specificity_train,  specificity_test = mics.specificity()

                # 虽然各类指标都算了,但是只向list中添加svm_metrics中要求给的
                for name in svm_metrics:
                    exec('{}_cv_train.append({}_train)'.format(name, name))
                    exec('{}_cv_test.append({}_test)'.format(name, name))

                # 计算fpr和tpr
                fpr_train, tpr_train, thres_train = roc_curve(y_train, prob_train[:, 1])
                fpr_test, tpr_test, thres_test = roc_curve(y_test, prob_test[:, 1])
                # 根据meanfpr_outer_test的长度，通过fpr和tpr的范围进行插值
                tpr_fold_train.append(np.interp(meanfpr_shuffle_train, fpr_train, tpr_train))
                tpr_fold_test.append(np.interp(meanfpr_shuffle_test, fpr_test, tpr_test))
                tpr_fold_train[-1][0] = 0.0
                tpr_fold_test[-1][0] = 0.0

            # 交叉验证所有折的平均值
            meantpr_fold_train = np.mean(tpr_fold_train, axis=0)
            meantpr_fold_test = np.mean(tpr_fold_test, axis=0)
            tpr_shuffle_train.append(meantpr_fold_train)
            tpr_shuffle_test.append(meantpr_fold_test)
            auc_train = auc(meanfpr_shuffle_train, meantpr_fold_train)
            auc_test = auc(meanfpr_shuffle_test, meantpr_fold_test)
            auc_list_train.append(auc_train)
            auc_list_test.append(auc_test)

            # 每次shuffle的每个指标都会计算出mean和std，并存储在train/test_{}_means, train/test_{}_std
            for name in svm_metrics:
                # 存储训练过程中交叉验证每个指标的平均值
                exec('{}_cv_train = np.array({}_cv_train)'.format(name, name))
                exec("train_{}_means.append({}_cv_train.mean())".format(name, name))
                # 存储训练过程中交叉验证每个指标的标准差
                exec("train_{}_std.append({}_cv_train.std())".format(name, name))
                # 存储测试过程中交叉验证每个指标的平均值
                exec('{}_cv_test = np.array({}_cv_test)'.format(name, name))
                exec("test_{}_means.append({}_cv_test.mean())".format(name, name))
                # 存储测试过程中交叉验证每个指标的标准差
                exec("test_{}_std.append({}_cv_test.std())".format(name, name))

        # 多次shuffle结果的平均值
        meantpr_shuffle_train = np.mean(tpr_shuffle_train, axis=0)
        meantpr_shuffle_test = np.mean(tpr_shuffle_test, axis=0)
        final_auc_train = auc(meanfpr_shuffle_train, meantpr_shuffle_train)
        final_auc_test = auc(meanfpr_shuffle_test, meantpr_shuffle_test)

        # 为了简洁，将训练、测试过程中的指标平均值和标准差以字典形式存储，再返回
        train_means = {}
        train_std = {}
        test_means = {}
        test_std = {}
        for name in svm_metrics:
            exec("train_means['{}'] = train_{}_means".format(name, name))
            exec("train_std['{}'] = train_{}_std".format(name, name))
            exec("test_means['{}'] = test_{}_means".format(name, name))
            exec("test_std['{}'] = test_{}_std".format(name, name))

        # 为了简洁,将绘制ROC曲线有关的变量用一个dict来表示
        roc_dict = {}
        roc_dict['tpr_train'] = meantpr_shuffle_train
        roc_dict['tpr_test'] = meantpr_shuffle_test
        roc_dict['tpr_list_train'] = tpr_shuffle_train
        roc_dict['tpr_list_test'] = tpr_shuffle_test
        roc_dict['fpr_train'] = meanfpr_shuffle_train
        roc_dict['fpr_test'] = meanfpr_shuffle_test
        roc_dict['auc_list_train'] = auc_list_train
        roc_dict['auc_list_test'] = auc_list_test
        roc_dict['auc_train'] = final_auc_train
        roc_dict['auc_test'] = final_auc_test

        # 将输出存在txt文件中
        for name in svm_metrics:
            svm_txt.write('\n---------------------------------------------\n')
            exec("svm_txt.write('Train set {} mean value: % s' % np.mean(train_means['{}']))".format(name, name))
            svm_txt.write('\n')
            exec("svm_txt.write('Train set {} max value: % s' % np.max(train_means['{}']))".format(name, name))
            svm_txt.write('\n')
            exec("svm_txt.write('Train set {} min value: % s' % np.min(train_means['{}']))".format(name, name))

            svm_txt.write('\n---------------------------------------------\n')
            exec("svm_txt.write('Test set {} mean value: % s' % np.mean(test_means['{}']))".format(name, name))
            svm_txt.write('\n')
            exec("svm_txt.write('Test set {} max value: % s' % np.max(test_means['{}']))".format(name, name))
            svm_txt.write('\n')
            exec("svm_txt.write('Test set {} min value: % s' % np.min(test_means['{}']))".format(name, name))
            svm_txt.write('\n---------------------------------------------\n')

        svm_txt.write('\nTrain set AUC mean value: % s' % np.mean(roc_dict['auc_list_train']))
        svm_txt.write('\nTrain set AUC max value: % s' % np.max(roc_dict['auc_list_train']))
        svm_txt.write('\nTrain set AUC min value: % s' % np.min(roc_dict['auc_list_train']))
        svm_txt.write('\n---------------------------------------------\n')
        svm_txt.write('\nTest set AUC mean value: % s' % np.mean(roc_dict['auc_list_test']))
        svm_txt.write('\nTest set AUC max value: % s' % np.max(roc_dict['auc_list_test']))
        svm_txt.write('\nTest set AUC min value: % s' % np.min(roc_dict['auc_list_test']))

        return train_means, train_std, test_means, test_std, roc_dict

    def mksvm2_grid(self, para, kernel_dict, svm_metrics, shuffle_time=100, cv=10, grid_num=10):
        '''多核SVM的实现，只适用于双模态，网格搜索法只对融合系数有效，SVM的参数需要提前设定好
        输入：
            grid_num: 网格搜索法，从0到1的参数个数，例如grid_num=10就是0到0.9这10个参数
            cv: 每一个网格参数组合进行交叉验证时，交叉验证的折数
            shuffle_time: 进行shuffle的次数

        输出:
            train_means: dict型，svm_metrics中要求的指标，训练集中所有shuffle结果的平均值存入
            test_means: dict型，svm_metrics中要求的指标，测试集中所有shuffle结果的平均值存入
            weight_name: dict型，包含weight1和weight2两个key，里面存的list是两个模态融合系数的值，顺序是对应的

        '''
        # 初始化每种权重下的指标list
        for name in svm_metrics:
            exec('mean{}_weight_train = []'.format(name))
            exec('mean{}_weight_test = []'.format(name))

        # 初始化每种特征组合的名称
        weight1_col = []
        weight2_col = []

        # 根据grid_num创建等差数列
        grid_list = np.arange(0, 1, 1/grid_num)

        # 每种融合系数
        for grid in grid_list:
            # 为两个模态的核矩阵融合系数赋值
            weight1 = grid
            weight2 = 1 - weight1

            # 将融合系数存入list
            weight1_col.append(weight1)
            weight2_col.append(weight2)

            # 根据svm_metrics中要求的指标进行初始化
            for name in svm_metrics:
                exec('mean{}_shuffle_train = []'.format(name))
                exec('mean{}_shuffle_test = []'.format(name))

            # 开始每种模态下的shuffle
            for i in range(shuffle_time):
                outer = StratifiedKFold(n_splits=cv, shuffle=True, random_state=i)

                # 初始化每次shuffle交叉验证的各类指标list
                for name in svm_metrics:
                    exec('{}_cv_train = []'.format(name))
                    exec('{}_cv_test = []'.format(name))

                # 开始每次shuffle的交叉验证过程
                for train, test in outer.split(self.X1, self.y):

                    # 将数据分为训练集和测试集，mat1和mat2分别为训练集和测试集特征矩阵，mod1和mod2分别为两个模态
                    mod1_mat1 = self.X1[train]
                    mod1_mat2 = self.X1[test]
                    mod2_mat1 = self.X2[train]
                    mod2_mat2 = self.X2[test]
                    y_train = self.y[train]
                    y_test = self.y[test]

                    # 计算核矩阵，先分别给两个模态定义实例
                    kernel_mod1 = kernel(kernel_type=kernel_dict['kernel_type1'],
                                         kernel_para=para['kernel1'])
                    kernel_mod2 = kernel(kernel_type=kernel_dict['kernel_type2'],
                                         kernel_para=para['kernel2'])

                    # 计算mod1的训练集核矩阵与测试集核矩阵
                    mod1_train = kernel_mod1.calckernel(mod1_mat1)
                    mod1_test = kernel_mod1.calckernel(mod1_mat1, mod1_mat2)

                    # 计算mod2的训练集核矩阵与测试集核矩阵
                    mod2_train = kernel_mod2.calckernel(mod2_mat1)
                    mod2_test = kernel_mod2.calckernel(mod2_mat1, mod2_mat2)

                    # 进行双模态核矩阵融合
                    train_kernel = weight1 * mod1_train + weight2 * mod2_train
                    test_kernel = weight1 * mod1_test + weight2 * mod2_test

                    # svm训练
                    svm = SVC(kernel='precomputed', C=para['C'], probability=True)
                    svm.fit(train_kernel, y_train)

                    # 得到训练集和测试集上的输出
                    pred_train = svm.predict(train_kernel)
                    prob_train = svm.predict_proba(train_kernel)
                    pred_test = svm.predict(test_kernel)
                    prob_test = svm.predict_proba(test_kernel)

                    mics = classifier_mics(y_train, pred_train, prob_train,
                                           y_test, pred_test, prob_test, path=self.path)

                    # 计算每一折的各类指标
                    accuracy_train, precision_train, recall_train, f1_train = mics.mics_sum_train()
                    accuracy_test, precision_test, recall_test, f1_test = mics.mics_sum_test()
                    sensitivity_train, sensitivity_test = mics.sensitivity()
                    specificity_train, specificity_test = mics.specificity()

                    # 虽然各类指标都算了,但是只向list中添加svm_metrics中要求给的
                    for name in svm_metrics:
                        exec('{}_cv_train.append({}_train)'.format(name, name))
                        exec('{}_cv_test.append({}_test)'.format(name, name))

                # 交叉验证结束后，计算每折的平均值作为此次交叉验证的结果，并将该结果存入shuffle的list中
                for name in svm_metrics:
                    exec('{}_cv_train = np.mean({}_cv_train)'.format(name, name))
                    exec('mean{}_shuffle_train.append({}_cv_train)'.format(name, name))
                    exec('{}_cv_test = np.mean({}_cv_test)'.format(name, name))
                    exec('mean{}_shuffle_test.append({}_cv_test)'.format(name, name))

            # 计算所有shuffle的平均值作为该参数组合下的最终结果
            for name in svm_metrics:
                exec('mean{}_shuffle_train = np.mean(mean{}_shuffle_train)'.format(name, name))
                exec('mean{}_weight_train.append(mean{}_shuffle_train)'.format(name, name))
                exec('mean{}_shuffle_test = np.mean(mean{}_shuffle_test)'.format(name, name))
                exec('mean{}_weight_test.append(mean{}_shuffle_test)'.format(name, name))

        # 为了简洁，将训练、测试过程中的指标平均值以字典形式存储，再返回
        train_means = {}
        test_means = {}
        for name in svm_metrics:
            exec("train_means['{}'] = mean{}_weight_train".format(name, name))
            exec("test_means['{}'] = mean{}_weight_test".format(name, name))

        # 将权重组合的名称存为dict
        weight_name = {}
        weight_name['weight1'] = weight1_col
        weight_name['weight2'] = weight2_col

        # 将以上指标网格以DataFrame格式存储为csv格式
        train_pd, train_ndarray = [], []
        test_pd, test_ndarray = [], []
        for name in svm_metrics:
            exec("train_ndarray.append(train_means['{}'])".format(name))
            exec("test_ndarray.append(test_means['{}'])".format(name))

        train_ndarray = np.array(train_ndarray)
        test_ndarray = np.array(test_ndarray)
        train_pd = pd.DataFrame(train_ndarray, index=svm_metrics, columns=['weight1='+str(round(i, 2)) for i in weight_name['weight1']])
        test_pd = pd.DataFrame(test_ndarray, index=svm_metrics, columns=['weight1='+str(round(i, 2)) for i in weight_name['weight1']])
        train_pd.to_csv(os.path.join(self.path, 'grid_result_train.csv'))
        test_pd.to_csv(os.path.join(self.path, 'grid_result_test.csv'))

        return train_means, test_means, weight_name

    def mksvm3(self, para, kernel_dict, svm_metrics, shuffle_time=100, outer=10):
        '''多核SVM的实现，只适用于三模态，网格搜索法只对融合系数有效，SVM的参数需要提前设定好
            输入：
                outer: 每一次shuffle进行交叉验证时，交叉验证的折数
                shuffle_time: 进行shuffle的次数
                kernel_dict: dict型，包含的参数有：
                    'kernel_type1': 第一个核的类型，支持linear, poly, rbf
                    'kernel_type2': 第二个核的类型，支持linear, poly, rbf
                    'kernel_type3': 第三个核的类型，支持linear, poly, rbf
                    'kernel_weight1': 第一个核的融合系数
                    'kernel_weight2': 第二个核的融合系数
                    'kernel_weight3': 第三个核的融合系数，注意kernel_weight1, kernel_weight2, kernel_weight3相加必须为1
                para: dict型，包含的参数有：
                    'kernel1': 第一个核的参数，如果linear没有参数，poly型即degree，rbf型即位gamma
                    'kernel2': 第二个核的参数，如果linear没有参数，poly型即degree，rbf型即位gamma
                    'kernel3': 第三个核的参数，如果linear没有参数，poly型即degree，rbf型即位gamma
                    'C': 支持向量机的参数C
                svm_metrics: list型，SVM输出结果后需要计算的指标，目前支持accuracy, precision, recall, f1, sensitivity, specificity。
                         必须写着完整名，不能用缩写

            输出：
            train_means: dict型，键值对应于svm_metrics定义的指标，对应的值为list型，具体是shuffle_time次数的训练集中对应指标
                        在交叉验证过程中的平均值，该dict返回后可以通过mean来求总体的平均值。
            train_std: 和上面的dict类似，不同的是计算的是所有shuffle的标准差而不是均值
            test_means: 和上面的dict类似，不用的是计算的是测试集中的均值
            test_std: 和上面的dict类似，计算的是测试集中的标准差
            roc_dict: dict型，返回的是与绘制ROC相关的list，包含：
                tpr_train: list，训练集中每次shuffle的tpr交叉验证平均值
                tpr_test: 测试集中每次shuffle的tpr交叉验证平均值
                tpr_list_train: 二维list，训练集每次shuffle交叉验证每折的tpr都会存储，一次shuffle一个list
                tpr_list_test: 二维list，测试集每次shuffle交叉验证每折的tpr都会存储，一次shuffle一个list
                fpr_train: list, 训练集中每次shuffle的tpr交叉验证平均值（其实是自定义长度的等差数列）
                fpr_test: list, 测试集中每次shuffle的tpr交叉验证平均值（其实是自定义长度的等差数列）
                auc_list_train: list, 记录了训练集每次shuffle计算得到的AUC
                auc_list_test: list, 记录了测试集每次shuffle计算得到的AUC
                auc_train: float, 训练集上所有shuffle的AUC的平均值
                auc_test: float, 测试集上所有shuffle的AUC的平均值

            前四个dict主要是为了绘制shuffle和每种指标的关系图，mean用于绘制指标的曲线，std可以绘制标准差的变化区域

            roc_dict真正实用的是tpr_train, tpr_test, fpr_train, fpr_test，这四个list再各自做平均后就可以获取绘制ROC的所有参数，
            auc_list可以绘制shuffle和AUC的曲线图，其他的值用处不大，仅仅以防万一要用
        '''
        # 将svm参数写入txt文档
        svm_shuffle_path = os.path.join(self.path, 'mkl_svm')
        os.makedirs(svm_shuffle_path, exist_ok=True)
        svm_txt = open(os.path.join(self.path, 'mkl_svm_result.txt'), 'w')
        svm_txt.write('Multi-Kernel Support Vector Machine parameters set:\n')
        svm_txt.write('C value: % s' % para['C'])
        svm_txt.write('\nShuffle time: % s' % shuffle_time)
        svm_txt.write('\nCross validation-fold: % s' % outer)
        svm_txt.write('\nsvm metrics: % s\n' % svm_metrics)
        svm_txt.write('\n---------------------------------------------\n')
        svm_txt.write('Kernel-1 parameter set:')
        svm_txt.write('\nKernel type: % s' % kernel_dict['kernel_type1'])
        svm_txt.write('\nKernel weight: % s' % kernel_dict['kernel_weight1'])
        svm_txt.write('\nKernel parameter: % s' % para['kernel1'])
        svm_txt.write('\n---------------------------------------------\n')
        svm_txt.write('Kernel-2 parameter set:\n')
        svm_txt.write('\nKernel type: % s' % kernel_dict['kernel_type2'])
        svm_txt.write('\nKernel weight: % s' % kernel_dict['kernel_weight2'])
        svm_txt.write('\nKernel parameter: % s' % para['kernel2'])
        svm_txt.write('\n---------------------------------------------\n')
        svm_txt.write('Kernel-3 parameter set:\n')
        svm_txt.write('\nKernel type: % s' % kernel_dict['kernel_type3'])
        svm_txt.write('\nKernel weight: % s' % kernel_dict['kernel_weight3'])
        svm_txt.write('\nKernel parameter: % s' % para['kernel3'])

        # 初始化每次shuffle的的各类指标
        for name in svm_metrics:
            exec('train_{}_means = []'.format(name))
            exec('train_{}_std = []'.format(name))
            exec('test_{}_means = []'.format(name))
            exec('test_{}_std = []'.format(name))

        # 直接将fpr定义为等差数列
        meanfpr_shuffle_train = np.linspace(0, 1, 100)
        meanfpr_shuffle_test = np.linspace(0, 1, 100)
        # 将tpr和auc定义为空,最终tpr_outer_test和meanfpr_outer_test的长度相同,auc和shuffle的次数相同
        tpr_shuffle_train = []
        auc_list_train = []
        tpr_shuffle_test = []
        auc_list_test = []

        for i in range(shuffle_time):

            outer_cv = StratifiedKFold(n_splits=outer, shuffle=True, random_state=i)

            # 初始化交叉验证过程中，每一折的各类指标
            for name in svm_metrics:
                exec('{}_cv_train = []'.format(name))
                exec('{}_cv_test = []'.format(name))

            # 初始化交叉验证过程中
            tpr_fold_train = []
            tpr_fold_test = []
            # 每一次shuffle的交叉验证过程
            for train, test in outer_cv.split(self.X1, self.y):

                # 将数据分为训练集和测试集，mat1和mat2分别为训练集和测试集特征矩阵，mod1和mod2分别为两个模态
                mod1_mat1 = self.X1[train]
                mod1_mat2 = self.X1[test]
                mod2_mat1 = self.X2[train]
                mod2_mat2 = self.X2[test]
                mod3_mat1 = self.X3[train]
                mod3_mat2 = self.X3[test]
                y_train = self.y[train]
                y_test = self.y[test]

                # 调取两个核函数的融合系数
                weight1 = kernel_dict['kernel_weight1']
                weight2 = kernel_dict['kernel_weight2']
                weight3 = kernel_dict['kernel_weight3']

                # 计算核矩阵，先分别给两个模态定义实例
                kernel_mod1 = kernel(kernel_type=kernel_dict['kernel_type1'],
                                     kernel_para=para['kernel1'])
                kernel_mod2 = kernel(kernel_type=kernel_dict['kernel_type2'],
                                     kernel_para=para['kernel2'])
                kernel_mod3 = kernel(kernel_type=kernel_dict['kernel_type3'],
                                     kernel_para=para['kernel3'])

                # 计算mod1的训练集核矩阵与测试集核矩阵
                mod1_train = kernel_mod1.calckernel(mod1_mat1)
                mod1_test = kernel_mod1.calckernel(mod1_mat1, mod1_mat2)

                # 计算mod2的训练集核矩阵与测试集核矩阵
                mod2_train = kernel_mod2.calckernel(mod2_mat1)
                mod2_test = kernel_mod2.calckernel(mod2_mat1, mod2_mat2)

                # 计算mod3的训练集核矩阵与测试集核矩阵
                mod3_train = kernel_mod3.calckernel(mod3_mat1)
                mod3_test = kernel_mod3.calckernel(mod3_mat1, mod3_mat2)

                # 进行三模态核矩阵融合
                train_kernel = weight1 * mod1_train + weight2 * mod2_train + weight3 * mod3_train
                test_kernel = weight1 * mod1_test + weight2 * mod2_test + weight3 * mod3_test

                # X_corr = pd.DataFrame(train_kernel)
                # corr_mat = X_corr.corr()
                # fig, axe = plt.subplots(figsize=(12, 9))
                # sns.heatmap(corr_mat, ax=axe)
                # fig.savefig(os.path.join('feature_corr_heatmap% s.png' % i))

                # svm训练
                svm = SVC(kernel='precomputed', C=para['C'], probability=True)
                svm.fit(train_kernel, y_train)

                # 得到训练集和测试集上的输出
                pred_train = svm.predict(train_kernel)
                prob_train = svm.predict_proba(train_kernel)
                pred_test = svm.predict(test_kernel)
                prob_test = svm.predict_proba(test_kernel)

                mics = classifier_mics(y_train, pred_train, prob_train,
                                       y_test, pred_test, prob_test, path='mkl_svm')

                # 计算每一折的各类指标
                accuracy_train, precision_train, recall_train, f1_train = mics.mics_sum_train()
                accuracy_test, precision_test, recall_test, f1_test = mics.mics_sum_test()
                sensitivity_train, sensitivity_test = mics.sensitivity()
                specificity_train,  specificity_test = mics.specificity()

                # 虽然各类指标都算了,但是只向list中添加svm_metrics中要求给的
                for name in svm_metrics:
                    exec('{}_cv_train.append({}_train)'.format(name, name))
                    exec('{}_cv_test.append({}_test)'.format(name, name))

                # 计算fpr和tpr
                fpr_train, tpr_train, thres_train = roc_curve(y_train, prob_train[:, 1])
                fpr_test, tpr_test, thres_test = roc_curve(y_test, prob_test[:, 1])
                # 根据meanfpr_outer_test的长度，通过fpr和tpr的范围进行插值
                tpr_fold_train.append(np.interp(meanfpr_shuffle_train, fpr_train, tpr_train))
                tpr_fold_test.append(np.interp(meanfpr_shuffle_test, fpr_test, tpr_test))
                tpr_fold_train[-1][0] = 0.0
                tpr_fold_test[-1][0] = 0.0

            # 交叉验证所有折的平均值
            meantpr_fold_train = np.mean(tpr_fold_train, axis=0)
            meantpr_fold_test = np.mean(tpr_fold_test, axis=0)
            tpr_shuffle_train.append(meantpr_fold_train)
            tpr_shuffle_test.append(meantpr_fold_test)
            auc_train = auc(meanfpr_shuffle_train, meantpr_fold_train)
            auc_test = auc(meanfpr_shuffle_test, meantpr_fold_test)
            auc_list_train.append(auc_train)
            auc_list_test.append(auc_test)

            # 每次shuffle的每个指标都会计算出mean和std，并存储在train/test_{}_means, train/test_{}_std
            for name in svm_metrics:
                # 存储训练过程中交叉验证每个指标的平均值
                exec('{}_cv_train = np.array({}_cv_train)'.format(name, name))
                exec("train_{}_means.append({}_cv_train.mean())".format(name, name))
                # 存储训练过程中交叉验证每个指标的标准差
                exec("train_{}_std.append({}_cv_train.std())".format(name, name))
                # 存储测试过程中交叉验证每个指标的平均值
                exec('{}_cv_test = np.array({}_cv_test)'.format(name, name))
                exec("test_{}_means.append({}_cv_test.mean())".format(name, name))
                # 存储测试过程中交叉验证每个指标的标准差
                exec("test_{}_std.append({}_cv_test.std())".format(name, name))

        # 多次shuffle结果的平均值
        meantpr_shuffle_train = np.mean(tpr_shuffle_train, axis=0)
        meantpr_shuffle_test = np.mean(tpr_shuffle_test, axis=0)
        final_auc_train = auc(meanfpr_shuffle_train, meantpr_shuffle_train)
        final_auc_test = auc(meanfpr_shuffle_test, meantpr_shuffle_test)

        # 为了简洁，将训练、测试过程中的指标平均值和标准差以字典形式存储，再返回
        train_means = {}
        train_std = {}
        test_means = {}
        test_std = {}
        for name in svm_metrics:
            exec("train_means['{}'] = train_{}_means".format(name, name))
            exec("train_std['{}'] = train_{}_std".format(name, name))
            exec("test_means['{}'] = test_{}_means".format(name, name))
            exec("test_std['{}'] = test_{}_std".format(name, name))

        # 为了简洁,将绘制ROC曲线有关的变量用一个dict来表示
        roc_dict = {}
        roc_dict['tpr_train'] = meantpr_shuffle_train
        roc_dict['tpr_test'] = meantpr_shuffle_test
        roc_dict['tpr_list_train'] = tpr_shuffle_train
        roc_dict['tpr_list_test'] = tpr_shuffle_test
        roc_dict['fpr_train'] = meanfpr_shuffle_train
        roc_dict['fpr_test'] = meanfpr_shuffle_test
        roc_dict['auc_list_train'] = auc_list_train
        roc_dict['auc_list_test'] = auc_list_test
        roc_dict['auc_train'] = final_auc_train
        roc_dict['auc_test'] = final_auc_test

        # 将输出存在txt文件中
        for name in svm_metrics:
            svm_txt.write('\n---------------------------------------------\n')
            exec("svm_txt.write('Train set {} mean value: % s' % np.mean(train_means['{}']))".format(name, name))
            svm_txt.write('\n')
            exec("svm_txt.write('Train set {} max value: % s' % np.max(train_means['{}']))".format(name, name))
            svm_txt.write('\n')
            exec("svm_txt.write('Train set {} min value: % s' % np.min(train_means['{}']))".format(name, name))

            svm_txt.write('\n---------------------------------------------\n')
            exec("svm_txt.write('Test set {} mean value: % s' % np.mean(test_means['{}']))".format(name, name))
            svm_txt.write('\n')
            exec("svm_txt.write('Test set {} max value: % s' % np.max(test_means['{}']))".format(name, name))
            svm_txt.write('\n')
            exec("svm_txt.write('Test set {} min value: % s' % np.min(test_means['{}']))".format(name, name))
            svm_txt.write('\n---------------------------------------------\n')

        svm_txt.write('\nTrain set AUC mean value: % s' % np.mean(roc_dict['auc_list_train']))
        svm_txt.write('\nTrain set AUC max value: % s' % np.max(roc_dict['auc_list_train']))
        svm_txt.write('\nTrain set AUC min value: % s' % np.min(roc_dict['auc_list_train']))
        svm_txt.write('\n---------------------------------------------\n')
        svm_txt.write('\nTest set AUC mean value: % s' % np.mean(roc_dict['auc_list_test']))
        svm_txt.write('\nTest set AUC max value: % s' % np.max(roc_dict['auc_list_test']))
        svm_txt.write('\nTest set AUC min value: % s' % np.min(roc_dict['auc_list_test']))

        return train_means, train_std, test_means, test_std, roc_dict

    def mksvm3_grid(self, para, kernel_dict, svm_metrics, shuffle_time=100, cv=10, grid_num=10):
        '''多核SVM的实现，只适用于三模态，网格搜索法只对融合系数有效，SVM的参数需要提前设定好
        输入：
            grid_num: 网格搜索法，从0到1的参数个数，例如grid_num=10就是0到0.9这10个参数
            cv: 每一个网格参数组合进行交叉验证时，交叉验证的折数
            shuffle_time: 进行shuffle的次数

        输出:
            train_means: dict型，svm_metrics中要求的指标，训练集中所有shuffle结果的平均值存入
            test_means: dict型，svm_metrics中要求的指标，测试集中所有shuffle结果的平均值存入
            weight_name: dict型，包含weight1和weight2两个key，里面存的list是两个模态融合系数的值，顺序是对应的

        '''
        # 初始化每种权重下的指标list
        for name in svm_metrics:
            exec('mean{}_weight_train = []'.format(name))
            exec('mean{}_weight_test = []'.format(name))

        # 初始化每种特征组合的名称
        weight1_col = []
        weight2_col = []
        weight3_col = []

        # 根据grid_num创建等差数列
        # grid_list = np.arange(0, 1, 1/grid_num)

        # 每种融合系数
        for weight1 in list(np.arange(0, 1, 1/grid_num)):
            for weight2 in list(np.arange(0, 1-weight1, 1/grid_num)):
                # 为两个模态的核矩阵融合系数赋值
                # weight1 = weight1 * 0.1
                # weight2 = weight2 * 0.1
                weight3 = 1 - weight1 - weight2

                # 将融合系数存入list
                weight1_col.append(weight1)
                weight2_col.append(weight2)
                weight3_col.append(weight3)

                # 根据svm_metrics中要求的指标进行初始化
                for name in svm_metrics:
                    exec('mean{}_shuffle_train = []'.format(name))
                    exec('mean{}_shuffle_test = []'.format(name))

                # 开始每种模态下的shuffle
                for i in range(shuffle_time):
                    outer = StratifiedKFold(n_splits=cv, shuffle=True, random_state=i)

                    # 初始化每次shuffle交叉验证的各类指标list
                    for name in svm_metrics:
                        exec('{}_cv_train = []'.format(name))
                        exec('{}_cv_test = []'.format(name))

                    # 开始每次shuffle的交叉验证过程
                    for train, test in outer.split(self.X1, self.y):

                        # 将数据分为训练集和测试集，mat1和mat2分别为训练集和测试集特征矩阵，mod1和mod2分别为两个模态
                        mod1_mat1 = self.X1[train]
                        mod1_mat2 = self.X1[test]
                        mod2_mat1 = self.X2[train]
                        mod2_mat2 = self.X2[test]
                        mod3_mat1 = self.X3[train]
                        mod3_mat2 = self.X3[test]
                        y_train = self.y[train]
                        y_test = self.y[test]

                        # 计算核矩阵，先分别给两个模态定义实例
                        kernel_mod1 = kernel(kernel_type=kernel_dict['kernel_type1'],
                                             kernel_para=para['kernel1'])
                        kernel_mod2 = kernel(kernel_type=kernel_dict['kernel_type2'],
                                             kernel_para=para['kernel2'])
                        kernel_mod3 = kernel(kernel_type=kernel_dict['kernel_type3'],
                                             kernel_para=para['kernel3'])

                        # 计算mod1的训练集核矩阵与测试集核矩阵
                        mod1_train = kernel_mod1.calckernel(mod1_mat1)
                        mod1_test = kernel_mod1.calckernel(mod1_mat1, mod1_mat2)

                        # 计算mod2的训练集核矩阵与测试集核矩阵
                        mod2_train = kernel_mod2.calckernel(mod2_mat1)
                        mod2_test = kernel_mod2.calckernel(mod2_mat1, mod2_mat2)

                        # 计算mod3的训练集核矩阵与测试集核矩阵
                        mod3_train = kernel_mod3.calckernel(mod3_mat1)
                        mod3_test = kernel_mod3.calckernel(mod3_mat1, mod3_mat2)

                        # 进行双模态核矩阵融合
                        train_kernel = weight1 * mod1_train + weight2 * mod2_train + weight3 * mod3_train
                        test_kernel = weight1 * mod1_test + weight2 * mod2_test + weight3 * mod3_test

                        # svm训练
                        svm = SVC(kernel='precomputed', C=para['C'], probability=True)
                        svm.fit(train_kernel, y_train)

                        # 得到训练集和测试集上的输出
                        pred_train = svm.predict(train_kernel)
                        prob_train = svm.predict_proba(train_kernel)
                        pred_test = svm.predict(test_kernel)
                        prob_test = svm.predict_proba(test_kernel)

                        mics = classifier_mics(y_train, pred_train, prob_train,
                                               y_test, pred_test, prob_test, path=self.path)

                        # 计算每一折的各类指标
                        accuracy_train, precision_train, recall_train, f1_train = mics.mics_sum_train()
                        accuracy_test, precision_test, recall_test, f1_test = mics.mics_sum_test()
                        sensitivity_train, sensitivity_test = mics.sensitivity()
                        specificity_train, specificity_test = mics.specificity()

                        # 虽然各类指标都算了,但是只向list中添加svm_metrics中要求给的
                        for name in svm_metrics:
                            exec('{}_cv_train.append({}_train)'.format(name, name))
                            exec('{}_cv_test.append({}_test)'.format(name, name))

                    # 交叉验证结束后，计算每折的平均值作为此次交叉验证的结果，并将该结果存入shuffle的list中
                    for name in svm_metrics:
                        exec('{}_cv_train = np.mean({}_cv_train)'.format(name, name))
                        exec('mean{}_shuffle_train.append({}_cv_train)'.format(name, name))
                        exec('{}_cv_test = np.mean({}_cv_test)'.format(name, name))
                        exec('mean{}_shuffle_test.append({}_cv_test)'.format(name, name))

                # 计算所有shuffle的平均值作为该参数组合下的最终结果
                for name in svm_metrics:
                    exec('mean{}_shuffle_train = np.mean(mean{}_shuffle_train)'.format(name, name))
                    exec('mean{}_weight_train.append(mean{}_shuffle_train)'.format(name, name))
                    exec('mean{}_shuffle_test = np.mean(mean{}_shuffle_test)'.format(name, name))
                    exec('mean{}_weight_test.append(mean{}_shuffle_test)'.format(name, name))

        # 为了简洁，将训练、测试过程中的指标平均值以字典形式存储，再返回
        train_means = {}
        test_means = {}
        for name in svm_metrics:
            exec("train_means['{}'] = mean{}_weight_train".format(name, name))
            exec("test_means['{}'] = mean{}_weight_test".format(name, name))

        # 将权重组合的名称存为dict
        weight_name = {}
        weight_name['weight1'] = weight1_col
        weight_name['weight2'] = weight2_col
        weight_name['weight3'] = weight3_col

        # 将以上指标网格以DataFrame格式存储为csv格式
        train_pd, train_ndarray = [], []
        test_pd, test_ndarray = [], []
        for name in svm_metrics:
            exec("train_ndarray.append(train_means['{}'])".format(name))
            exec("test_ndarray.append(test_means['{}'])".format(name))

        train_ndarray = np.array(train_ndarray)
        test_ndarray = np.array(test_ndarray)
        train_pd = pd.DataFrame(train_ndarray, index=svm_metrics,
                                columns=['weight1='+str(round(i, 2))+', weight2='+str(round(j, 2)) for i, j in zip(weight_name['weight1'], weight_name['weight2'])])
        test_pd = pd.DataFrame(test_ndarray, index=svm_metrics,
                               columns=['weight1='+str(round(i, 2))+', weight2='+str(round(j, 2)) for i, j in zip(weight_name['weight1'], weight_name['weight2'])])
        train_pd.to_csv(os.path.join(self.path, 'grid_result_train.csv'))
        test_pd.to_csv(os.path.join(self.path, 'grid_result_test.csv'))

        return train_means, test_means, weight_name


class RandomForest():
    '''包括网格搜索法的随机森林模型

    输入：
        分别为训练集特征、测试集特征、训练集标签、测试集标签、标签名称

    输出：
        各类特征重要性以及预测结果
    '''

    def __init__(self, X_train, X_test, y_train, y_test, labelName, path):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.labelName = labelName
        self.path = path

    def rf_baseline(self,
                    columeName,
                    n_estimators=100,
                    max_features='auto',
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    bootstrap=True
                    ):
        '''构建随机森林baseline模型

        输入：
            columeName: 即pipeline中的data.columns，每列特征的名字

        输出：
            预测结果
            特征重要性排列图
        '''
        # 构建随机森林baseline模型
        rfc = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_features=max_features,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    bootstrap=bootstrap)
        rfc.fit(self.X_train, self.y_train)
        pred_train = rfc.predict(self.X_train)
        pred_test = rfc.predict(self.X_test)

        y_score_train = rfc.predict_proba(self.X_train)
        y_score_test = rfc.predict_proba(self.X_test)

        baseline_path = os.path.join(self.path, 'baseline')
        os.makedirs(baseline_path, exist_ok=True)
        # 可视化特征重要性
        feats = {}
        for feature, importance in zip(columeName, rfc.feature_importances_):
            feats[feature] = importance

        importances = pd.DataFrame.from_dict(feats, orient='index').rename(
            columns={0: 'Importance'})
        importances = importances.sort_values(by='Importance', ascending=False).head(20)
        importances = importances.reset_index()
        importances = importances.rename(columns={'index': 'Features'})
        sns.set(style="whitegrid", color_codes=True, font_scale=1)
        fig, axe = plt.subplots()
        fig.set_size_inches(25, 20)
        sns.barplot(x=importances['Importance'], y=importances['Features'], data=importances, color='skyblue')
        axe.set_xlabel('Importance')
        axe.set_ylabel('Features')
        axe.set_title('Feature Importance', fontsize=25, weight='bold')
        img_path = os.path.join(self.path, 'baseline', 'Random Forest Baseline feature importance.png')
        fig.savefig(img_path)

        return pred_train, y_score_train, pred_test, y_score_test

    def rf_paraSearch(self,
                      n_estimators,
                      max_features,
                      max_depth,
                      min_samples_split,
                      min_samples_leaf,
                      bootstrap,
                      cross_val=3,
                      search_method='RandomizedSearch'
                      ):
        '''使用随机搜索法或者网格搜索法来获取随机森林最佳参数模型
        输入：
            search_method: 使用的参数搜索方法，default='RandomizedSearch'，可选GridSearch
        输出：
            如果为RandomizedSearch，保存每个参数对应acc的图
        '''
        if search_method == 'RandomizedSearch':
            # 定义随机森林模型
            rfc_cv = RandomForestClassifier()
            # 定义网格搜索的参数范围
            param_dist = {'n_estimators': n_estimators,
                          'max_features': max_features,
                          'max_depth': max_depth,
                          'min_samples_split': min_samples_split,
                          'min_samples_leaf': min_samples_leaf,
                          'bootstrap': bootstrap}
            # 定义参数搜索的设置
            rs = RandomizedSearchCV(rfc_cv,
                                    param_dist,
                                    n_iter=50,
                                    cv=cross_val,
                                    verbose=1,
                                    n_jobs=-1,
                                    random_state=0)
            # 使用网格搜索模型拟合数据
            rs.fit(self.X_train, self.y_train)

            # 根据rank_test_score，对每次交叉验证的结果进行重新排序
            rs_df = pd.DataFrame(rs.cv_results_).sort_values('rank_test_score').reset_index(drop=True)
            # 保留RandomizedSearchCV返回的以下这几个结果
            rs_df = rs_df[['param_n_estimators', 'param_max_features', 'param_max_depth',
                           'param_min_samples_split', 'param_min_samples_leaf', 'param_bootstrap',
                           'mean_test_score', 'rank_test_score']]
            # print(rs_df.head(10))

            os.makedirs(self.path, exist_ok=True)
            file_path = os.path.join(self.path, 'Random Forest RandomizedSearchCV Results.csv')
            rs_df.to_csv(path_or_buf=file_path)

            # 返回最佳参数下模型交叉验证中的平均得分、最佳参数组合
            best_para_random = rs.best_params_
            print('RF CV Best score: %0.3f' % rs.best_score_)
            print('RF CV Best parameters set:')
            print('-------------------------------------------')
            for param_name in sorted(best_para_random.keys()):
                print('\t%s: %r' % (param_name, best_para_random[param_name]))

            # 创建定义的6个超参数的柱状图，并针对每个值制作模型的平均得分，查看平均而言最优的值
            fig, axs = plt.subplots(ncols=3, nrows=2)
            sns.set(style="whitegrid", color_codes=True, font_scale=2)
            fig.set_size_inches(15, 10)

            sns.barplot(x='param_n_estimators', y='mean_test_score', data=rs_df, ax=axs[0, 0], color='lightgrey')
            # axs[0, 0].set_title(label='n_estimators', size=40, weight='bold')
            sns.barplot(x='param_min_samples_split', y='mean_test_score', data=rs_df, ax=axs[0, 1], color='coral')
            # axs[0, 1].set_title(label='min_samples_split', size=40, weight='bold')
            sns.barplot(x='param_min_samples_leaf', y='mean_test_score', data=rs_df, ax=axs[0, 2], color='lightgreen')
            # axs[0, 2].set_title(label='min_samples_leaf', size=40, weight='bold')
            sns.barplot(x='param_max_features', y='mean_test_score', data=rs_df, ax=axs[1, 0], color='wheat')
            # axs[1, 0].set_title(label='max_features', size=40, weight='bold')
            sns.barplot(x='param_max_depth', y='mean_test_score', data=rs_df, ax=axs[1, 1], color='lightpink')
            # axs[1, 1].set_title(label='max_depth', size=40, weight='bold')
            sns.barplot(x='param_bootstrap', y='mean_test_score', data=rs_df, ax=axs[1, 2], color='skyblue')
            # axs[1, 2].set_title(label='bootstrap', size=40, weight='bold')

            img_path = os.path.join(self.path, 'RF randomSearch results.png')
            fig.savefig(img_path)

            pred_train = rs.best_estimator_.predict(self.X_train)
            pred_test = rs.best_estimator_.predict(self.X_test)
            y_score_train = rs.best_estimator_.predict_proba(self.X_train)
            y_score_test = rs.best_estimator_.predict_proba(self.X_test)

        else:
            '不使用随机搜索，使用网格搜索法'
            # 基于RandomSearchCV搜索结果，使用GridSearchCV来详细搜索参数范围
            param_grid = {'n_estimators': n_estimators,
                          'max_features': max_features,
                          'max_depth': max_depth,
                          'min_samples_split': min_samples_split,
                          'min_samples_leaf': min_samples_leaf,
                          'bootstrap': bootstrap}
            rf_grid = RandomForestClassifier()
            gs_cv = GridSearchCV(rf_grid, param_grid, cv=cross_val, verbose=1, n_jobs=-1)
            gs_cv.fit(self.X_train, self.y_train)

            print('Best score: %0.3f' % gs_cv.best_score_)
            print('Best parameters set:')
            best_parameters_grid = gs_cv.best_params_
            print('-------------------------------------------')
            for param_name in sorted(best_parameters_grid.keys()):
                print('\t%s: %r' % (param_name, best_parameters_grid[param_name]))

            pred_train = gs_cv.best_estimator_.predict(self.X_train)
            pred_test = gs_cv.best_estimator_.predict(self.X_test)
            y_score_train = gs_cv.best_estimator_.predict_proba(self.X_train)
            y_score_test = gs_cv.best_estimator_.predict_proba(self.X_test)

        return pred_train,  y_score_train, pred_test, y_score_test


if __name__ == '__main__':
    pass
