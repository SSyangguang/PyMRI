import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import permutation_test_score
from scipy.stats import ttest_ind, mannwhitneyu


class Ttest():
    '''使用t检验进行特征选择，包含单独t检验、均值留一法t检验、频率留一法t检验
    输入：
        X: ndarray, 特征矩阵
        name: list, 特征名
        dis_num: int, 病人数量
        hc_num: int, 健康人数量
        thres: float, 用于选择特征的阈值
        value: string, 'p'或者't'
        all_value: bool,  是否返回所有特征的p值和t值

    '''
    def __init__(self, X, name, dis_num, hc_num, thres, value='p', all_value=True):
        self.X = X
        self.feature_name = name
        self.disease_num = dis_num
        self.health_num = hc_num
        self.threshold = thres
        self.value = value
        self.all_value = all_value

    def ttest_only(self):
        '''直接使用单次t检验进行特征选择
        输入：与class初始化的保持一致

        输出：
            select_feature: ndarray, 特征选择后的矩阵
            select_name: list, 特征选择后矩阵相对应的特征名称
            select_pvalue: list, 特征选择后每个特征对应的p值
            select_tvalue: list, 特征选择后每个特征对应的t值
            select_feature_index: 选择后特征在原特征矩阵中的索引
            若返回所有特征的p值和t值，则
                p: list, 所有特征的p值
                t: list, 所有特征的t值

        '''
        # 获取特征数量
        feature_num = self.X.shape[1]
        # 将病人和健康人的特征分离
        disease_num = self.disease_num
        feature_disease = self.X[: disease_num, :]
        feature_health = self.X[disease_num:, :]

        # 计算病人和健康人之间的p值和t值
        t, p = ttest_ind(feature_disease, feature_health)

        # 如果用p值来进行特征选择，则将thres设置为p值，否则为t值
        if self.value == 'p':
            thres = p
        elif self.value == 't':
            thres = t

        select_feature_index = []
        select_name = []
        select_pvalue = []
        select_tvalue = []
        # 将p值或t值低于设定阈值的特征索引和特征名称进行存储
        for col, name in zip(range(feature_num), self.feature_name):
            if thres[col] < self.threshold:
                select_feature_index.append(col)
                select_name.append(name)
                select_pvalue.append(p[col])
                select_tvalue.append(t[col])

        # 通过索引将选择后的特征矩阵赋值给select_feature
        select_feature = self.X[:, select_feature_index]

        if self.all_value is True:
            return select_feature, select_name, select_pvalue, select_tvalue, p, t, select_feature_index
        else:
            return select_feature, select_name, select_pvalue, select_tvalue, select_feature_index

    def ttest_loo_mean(self):
        '''使用留一法进行特征选择，使用每个特征所有循环p值t值的平均值作为最终值，然后使用该最终值进行筛选
           输入和输出与ttest_only保持一致
        '''
        # 获取样本数量，特征数量，病人数，健康人数
        sample_num = self.X.shape[0]
        feature_num = self.X.shape[1]
        disease_num = self.disease_num
        health = self.health_num

        # 初始化每个特征的p值和t值
        p_sum = [0] * feature_num
        t_sum = [0] * feature_num

        # 将病人和健康人的特征分离
        feature_disease = self.X[: disease_num, :]
        feature_health = self.X[disease_num:, :]

        # 使用留一法计算每个特征的p值和t值
        for i in range(sample_num):
            # 每次循环按照顺序去掉一个样本
            if i < disease_num:
                disease = np.delete(feature_disease, i, axis=0)
                health = feature_health
            else:
                disease = feature_disease
                health = np.delete(feature_health, i - disease_num, axis=0)

            # 计算p值和t值并进行累加
            t, p = ttest_ind(disease, health)
            p_sum += p
            t_sum += t
        p_mean = p_sum / sample_num
        t_mean = t_sum / sample_num

        # 如果用p值来进行特征选择，则将thres设置为p值，否则为t值
        if self.value == 'p':
            thres = p_mean
        elif self.value == 't':
            thres = t_mean

        select_feature_index = []
        select_name = []
        select_pvalue = []
        select_tvalue = []
        # 将p值或t值低于设定阈值的特征索引和特征名称进行存储
        for col, name in zip(range(feature_num), self.feature_name):
            if thres[col] < self.threshold:
                select_feature_index.append(col)
                select_name.append(name)
                select_pvalue.append(p_mean[col])
                select_tvalue.append(t_mean[col])

        # 通过索引将选择后的特征矩阵赋值给select_feature
        select_feature = self.X[:, select_feature_index]

        if self.all_value is True:
            return select_feature, select_name, select_pvalue, select_tvalue, p_mean, t_mean, select_feature_index
        else:
            return select_feature, select_name, select_pvalue, select_tvalue, select_feature_index

    def ttest_loo_freq(self, K_value, choice='percent'):
        '''使用留一法进行特征选择，每次循环都进行一次特征筛选，然后统计每个特征出现的次数，将前x%的特征筛选出来
        输入：
            K_value: 作为选择阈值，可以选择是选择后的特征占总体百分数或者选择后的特征个数
            choice: 包含'percent'和'Kbest'两种，分别为使用前百分之x和前K个出现频率最高的特征作为选择后特征

        输出：与ttest_only保持一致
        '''
        # 获取样本数量，特征数量，病人数，健康人数
        sample_num = self.X.shape[0]
        feature_num = self.X.shape[1]
        disease_num = self.disease_num
        health = self.health_num

        # 将病人和健康人的特征分离
        feature_disease = self.X[: disease_num, :]
        feature_health = self.X[disease_num:, :]

        # 使用留一法计算每个特征的p值和t值，将每次循环出现的特征存储在select_list中
        select_list = []
        for i in range(sample_num):
            if i < disease_num:
                disease = np.delete(feature_disease, i, axis=0)
                health = feature_health
            else:
                disease = feature_disease
                health = np.delete(feature_health, i - disease_num, axis=0)
            t, p = ttest_ind(disease, health)

            for col, name in zip(range(feature_num), self.feature_name):
                # 如果用p值来进行特征选择，则用p值进行筛选，否则为t值
                if self.value == 'p':
                    if p[col] < self.threshold:
                        # 每次循环p值小于阈值的特征名字都存到selectList，等下一步统计每个特征的出现频率
                        select_list.append(name)
                elif self.value == 't':
                    if t[col] < self.threshold:
                        # 每次循环p值小于阈值的特征名字都存到selectList，等下一步统计每个特征的出现频率
                        select_list.append(name)

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

        # 如果用百分比选择特征数量，那么用percent，如果用个数选择特征数量，那么用Kbest，得到实际的特征数量
        if choice == 'percent':
            select_num = int((K_value / 100) * feature_num)
            # 如果类要求的特征数量大于筛选得到的特征数量，会出错，所以将要求的特征数量设置为实际筛选得到的特征数量
            if select_num > len(select_feature_name):
                print('输入特征数量K_best大于过滤后特征数量，已将自动将特征最大值设置为过滤后特征数量% s' % len(select_feature_name))
                select_num = len(select_feature_name)
        elif choice == 'Kbest':
            select_num = K_value
            if select_num > len(select_feature_name):
                print('输入特征数量K_best大于过滤后特征数量，已自动将特征最大值设置为过滤后特征数量' % len(select_feature_name))
                select_num = len(select_feature_name)

        # 只选择预先定好数量的前x个特征
        select_feature_name = select_feature_name[: select_num]
        select_feature_name_freq = select_feature_name_freq[: select_num]

        select_feature_index = []
        name_list = list(select_feature_name)
        all_name_list = list(self.feature_name)
        for i in range(len(select_feature_name)):
            index = all_name_list.index(name_list[i])
            select_feature_index.append(index)

        # 通过索引将选择后的特征矩阵赋值给select_feature
        select_feature = self.X[:, select_feature_index]

        # 将所有特征和筛选后的特征p值与t值选出来
        t, p = ttest_ind(feature_disease, feature_health)
        select_pvalue = p[select_feature_index]
        select_tvalue = t[select_feature_index]

        if self.all_value is True:
            return select_feature, select_feature_name, select_feature_name_freq, \
                   select_pvalue, select_tvalue, p, t, select_feature_index
        else:
            return select_feature, select_feature_name, select_feature_name_freq, \
                   select_pvalue, select_tvalue, select_feature_index

class Utest():
    '''使用mann whitney U检验进行特征选择，包含单独u检验、平均值留一法u检验、频率留一法u检验
    输入：
        X: ndarray, 特征矩阵
        name: list, 特征名
        dis_num: int, 病人数量
        hc_num: int, 健康人数量
        thres: float, 阈值
        all_value: bool,  是否返回所有特征的p值和t值

    '''

    def __init__(self, X, name, dis_num, hc_num, thres, stat_path,  all_value=True):
        self.X = X
        self.feature_name = name
        self.disease_num = dis_num
        self.health_num = hc_num
        self.threshold = thres
        self.path = stat_path
        self.all_value = all_value

    # 由于scipy的mannwhitneyu函数仅支持一维数组式的输入，所以要直接把计算p值的函数放在循环中
    def utest_only(self):
        '''直接使用单次t检验进行特征选择
        输入：与class初始化的保持一致

        输出：
            select_feature: ndarray, 特征选择后的矩阵
            select_name: list, 特征选择后矩阵相对应的特征名称
            select_pvalue: list, 特征选择后每个特征对应的p值
            select_feature_index: 选择后特征在原特征矩阵中的索引
            若返回所有特征的p值和t值，则
                p: list, 所有特征的p值
        '''
        # 获取特征数量
        feature_num = self.X.shape[1]
        # 将病人和健康人的特征分离
        disease_num = self.disease_num
        feature_disease = self.X[: disease_num, :]
        feature_health = self.X[disease_num:, :]

        select_feature_index = []
        select_name = []
        select_pvalue = []
        # 将p值或t值低于设定阈值的特征索引和特征名称进行存储
        for col, name in zip(range(feature_num), self.feature_name):
            # 计算病人和健康人之间的p值和t值
            u, p = mannwhitneyu(feature_disease[:, col], feature_health[:, col])
            if p < self.threshold:
                select_feature_index.append(col)
                select_name.append(name)
                select_pvalue.append(p)

        # 通过索引将选择后的特征矩阵赋值给select_feature
        select_feature = self.X[:, select_feature_index]

        if self.all_value is True:
            return select_feature, select_name, select_pvalue, select_pvalue, select_feature_index
        else:
            return select_feature, select_name, select_pvalue, select_feature_index

    def utest_loo_mean(self):
        '''使用留一法进行特征选择，使用每个特征所有循环p值的平均值作为最终值，然后使用该最终值进行筛选
           输入和输出与utest_only保持一致
        '''
        # 获取样本数量，特征数量，病人数，健康人数
        sample_num = self.X.shape[0]
        feature_num = self.X.shape[1]
        disease_num = self.disease_num
        health = self.health_num

        # 初始化每个特征的p值
        p_sum = [0] * feature_num

        # 将病人和健康人的特征分离
        feature_disease = self.X[: disease_num, :]
        feature_health = self.X[disease_num:, :]

        # 使用留一法计算每个特征的p值和t值
        for i in range(sample_num):
            # 每次循环按照顺序去掉一个样本
            if i < disease_num:
                disease = np.delete(feature_disease, i, axis=0)
                health = feature_health
            else:
                disease = feature_disease
                health = np.delete(feature_health, i - disease_num, axis=0)

            # 计算p值并进行累加
            u, p = ttest_ind(disease, health)
            p_sum += p
        p_mean = p_sum / sample_num

        select_feature_index = []
        select_name = []
        select_pvalue = []
        # 将p值低于设定阈值的特征索引和特征名称进行存储
        for col, name in zip(range(feature_num), self.feature_name):
            if p_mean[col] < self.threshold:
                select_feature_index.append(col)
                select_name.append(name)
                select_pvalue.append(p_mean[col])

        # 通过索引将选择后的特征矩阵赋值给select_feature
        select_feature = self.X[:, select_feature_index]

        if self.all_value is True:
            return select_feature, select_name, select_pvalue, p_mean, select_feature_index
        else:
            return select_feature, select_name, select_pvalue, select_feature_index

    def utest_loo_freq(self, K_value, choice='percent'):
        '''使用留一法进行特征选择，每次循环都进行一次特征筛选，然后统计特征出现的次数，将前x%的特征筛选出来
        输入:
            K_value: 作为选择阈值，可以选择是选择后的特征占总体百分数或者选择后的特征个数
            choice: 包含'percent'和'Kbest'两种，分别为使用前百分之x和前K个出现频率最高的特征作为选择后特征

        输出：与ttest_only保持一致
        '''
        # 获取样本数量，特征数量，病人数，健康人数
        sample_num = self.X.shape[0]
        feature_num = self.X.shape[1]
        disease_num = self.disease_num
        health = self.health_num

        # 将病人和健康人的特征分离
        feature_disease = self.X[: disease_num, :]
        feature_health = self.X[disease_num:, :]

        # 使用留一法计算每个特征的p值和t值，将每次循环出现的特征存储在select_list中
        select_list = []
        for i in range(sample_num):
            if i < disease_num:
                disease = np.delete(feature_disease, i, axis=0)
                health = feature_health
            else:
                disease = feature_disease
                health = np.delete(feature_health, i - disease_num, axis=0)
            u, p = ttest_ind(disease, health)

            for col, name in zip(range(feature_num), self.feature_name):
                # 如果用p值来进行特征选择，则用p值进行筛选，否则为t值
                if p[col] < self.threshold:
                    # 每次循环p值小于阈值的特征名字都存到selectList，等下一步统计每个特征的出现频率
                    select_list.append(name)

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

        # 如果用百分比选择特征数量，那么用percent，如果用个数选择特征数量，那么用Kbest，得到实际的特征数量
        if choice == 'percent':
            select_num = int((K_value / 100) * feature_num)
            # 如果类要求的特征数量大于筛选得到的特征数量，会出错，所以将要求的特征数量设置为实际筛选得到的特征数量
            if select_num > len(select_feature_name):
                print('输入特征数量K_best大于过滤后特征数量，已将自动将特征最大值设置为过滤后特征数量% s' % len(select_feature_name))
                select_num = len(select_feature_name)
        elif choice == 'Kbest':
            select_num = K_value
            if select_num > len(select_feature_name):
                print('输入特征数量K_best大于过滤后特征数量，已将自动将特征最大值设置为过滤后特征数量' % len(select_feature_name))
                select_num = len(select_feature_name)

        # 只选择预先定好数量的前x个特征
        select_feature_name = select_feature_name[: select_num]
        select_feature_name_freq = select_feature_name_freq[: select_num]

        select_feature_index = []
        name_list = list(select_feature_name)
        all_name_list = list(self.feature_name)
        for i in range(len(select_feature_name)):
            index = all_name_list.index(name_list[i])
            select_feature_index.append(index)

        # 通过索引将选择后的特征矩阵赋值给select_feature
        select_feature = self.X[:, select_feature_index]

        # 将所有特征和筛选后的特征p值选出来
        u, p = ttest_ind(feature_disease, feature_health)
        select_pvalue = p[select_feature_index]

        # 将返回的值存入txt文件中
        stat_txt = open(os.path.join(self.path, 'u_test_loo.txt'), 'w')
        stat_txt.write('U test loocv parameters set:\n')
        stat_txt.write('\n---------------------------------------------\n')
        stat_txt.write('Thresholf set: % s' % self.threshold)
        stat_txt.write('\nChoice: % s' % choice)
        stat_txt.write('\nKvalue: % s' % K_value)
        stat_txt.write('\n---------------------------------------------\n')

        if self.all_value is True:
            # 将返回值存入文件中
            stat_txt.write('\nSelected features index:\n')
            stat_txt.write(str(select_feature_index))
            stat_txt.write('\n---------------------------------------------\n')
            stat_txt.write('\nSelected features name:\n')
            stat_txt.write(str(select_feature_name))
            stat_txt.write('\n---------------------------------------------\n')
            stat_txt.write('\nSelected features appearance frequency:\n')
            stat_txt.write(str(select_feature_name_freq))
            stat_txt.write('\n---------------------------------------------\n')
            stat_txt.write('\nSelected features p value:\n')
            stat_txt.write(str(select_pvalue))
            stat_txt.write('\n---------------------------------------------\n')
            stat_txt.write('\nValues of all features: \n')
            stat_txt.write(str(p))
            stat_txt.write('\n---------------------------------------------\n')

            return select_feature, select_feature_name, select_feature_name_freq,\
                   select_pvalue, p, select_feature_index
        else:
            # 将返回值存入文件中
            stat_txt.write('\nSelected features index:\n')
            stat_txt.write(str(select_feature_index))
            stat_txt.write('\n---------------------------------------------\n')
            stat_txt.write('\nSelected features name:\n')
            stat_txt.write(str(select_feature_name))
            stat_txt.write('\n---------------------------------------------\n')
            stat_txt.write('\nSelected features appearance frequency:\n')
            stat_txt.write(str(select_feature_name_freq))
            stat_txt.write('\n---------------------------------------------\n')
            stat_txt.write('\nSelected features p value:\n')
            stat_txt.write(str(select_pvalue))
            stat_txt.write('\n---------------------------------------------\n')

            return select_feature, select_feature_name, select_feature_name_freq, \
                   select_pvalue, select_feature_index

class result_test():
    '''
    用于对分类结果进行统计检验，这里包括two sample t-test和permutation test
    输入：
        estimator：使用的分类器
        X: 特征矩阵
        y: 类别标签
        cv: 交叉验证次数
        n_permutations: permutition检验的置乱次数
    输出：
        true_score: 未经过置乱的真实分类结果得分
        permutation_scores: 每一次置乱后的结果得分
        pvalue: p值

    '''

    def __init__(self, estimator, X, y, path, cv=10, n_permutation=1000):
        self.estimator = estimator
        self.X = X
        self.y = y
        self.cv = cv
        self.n_permutation = n_permutation
        self.path = path

    def permutation(self):
        score, permutation_scores, pvalue = permutation_test_score(
            self.estimator, self.X, self.y, scoring="accuracy", cv=self.cv, n_permutations=self.n_permutation)

        print("Classification score %s (pvalue : %s)" % (score, pvalue))
        n_classes = np.unique(self.y).size
        # View histogram of permutation scores
        plt.hist(permutation_scores, 20, label='Permutation scores',
                 edgecolor='black')
        ylim = plt.ylim()
        plt.plot(2 * [score], ylim, '--g', linewidth=3,
                 label='Classification Score'
                       ' (pvalue %s)' % pvalue)
        plt.plot(2 * [1. / n_classes], ylim, '--k', linewidth=3, label='Luck')

        plt.ylim(ylim)
        plt.legend()
        plt.xlabel('Score')
        plt.show()