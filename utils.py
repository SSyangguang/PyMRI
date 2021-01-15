import numpy as np
from sklearn.preprocessing import StandardScaler


class normalization():
    '''对数据进行预处理


    '''
    def __init__(self, x):
        self.X = x

    def z_score(self):
        '''
        对数据进行z标准化

        '''
        # 对数据进行z标准化
        ss = StandardScaler().fit(self.X)
        X = ss.fit_transform(self.X)

        return X


class kernel():
    def __init__(self, kernel_type, kernel_para):
        self.kernel_type = kernel_type
        self.kernel_para = kernel_para

    def calckernel(self, X_train, X_test=None):

        # 将输入矩阵转为mat格式，便于使用.T用于转置
        mat1 = np.mat(X_train)

        # 判断X_test是否存在
        if X_test is not None:
            mat2 = np.mat(X_test)

        # 核函数类型
        if self.kernel_type == 'linear':
            if X_test is not None:
                kernel_mat = mat2 * mat1.T
            else:
                kernel_mat = mat1 * mat1.T

        elif self.kernel_type == 'poly':
            if X_test is not None:
                kernel_mat = np.power((mat2 * mat1.T), self.kernel_para)
            else:
                kernel_mat = np.power((mat1 * mat1.T), self.kernel_para)

        elif self.kernel_type == 'rbf':
            if X_test is not None:
                trnorms1 = np.mat([(v * v.T)[0, 0] for v in mat1]).T
                trnorms2 = np.mat([(v * v.T)[0, 0] for v in mat2]).T
                sigma = self.kernel_para
                k1 = trnorms1 * np.mat(np.ones((mat2.shape[0], 1), dtype=np.float64)).T
                k2 = np.mat(np.ones((mat1.shape[0], 1), dtype=np.float64)) * trnorms2.T
                k = k1 + k2
                k -= 2 * np.mat(mat1 * mat2.T)
                k *= - 1. / (2 * np.power(sigma, 2))
                kernel_mat = np.exp(k)
                kernel_mat = kernel_mat.T

            else:
                trnorms1 = np.mat([(v * v.T)[0, 0] for v in mat1]).T
                trnorms2 = np.mat([(v * v.T)[0, 0] for v in mat1]).T
                sigma = self.kernel_para
                k1 = trnorms1 * np.mat(np.ones((mat1.shape[0], 1), dtype=np.float64)).T
                k2 = np.mat(np.ones((mat1.shape[0], 1), dtype=np.float64)) * trnorms2.T
                k = k1 + k2
                k -= 2 * np.mat(mat1 * mat1.T)
                k *= - 1. / (2 * np.power(sigma, 2))
                kernel_mat = np.exp(k)
                kernel_mat = kernel_mat.T

        return kernel_mat


class index_split():
    '''用于将多模态的特征拆分为每个模态单独的索引。根据每个模态特征的数量以及在混合索引中的值，将不同模态的索引进行划分。
        例如混合索引为index=[1, 3, 6, 8, 11]，其中模态1的特征数量为6，模态2的数量为10，那么拆分后模态1的index=[1, 3]，模态2的index=[6, 8, 11]
        index：需要进行拆分的索引
    '''

    def __init__(self, index):
        self.index = index

    def two_split(self, mod1_num, mod2_num):
        '''直接拆分两种模态的索引
        输入：
            mod1_num：模态1的数量
            mod2_num：模态2的数量

        输出：
            index_mod1：拆分后模态1的索引
            index_mod2：拆分后模态2的索引

        '''
        # 根据fmri和smri的特征数量，将他们重新分成两个矩阵，以便进行mkl
        index_mod1 = [i for i in self.index if i < mod1_num]
        index_mod2 = [i for i in self.index if (i >= mod1_num) and (i < mod2_num + mod1_num)]

        return index_mod1, index_mod2

    def two_reset(self, mod1_num, mod2_num):
        '''拆分只有两种模态的索引，每种模态的索引重新按照0为起始开始计算
        输入：
            mod1_num：模态1的数量
            mod2_num：模态2的数量

        输出：
            index_mod1：拆分后重新置0模态1的索引
            index_mod2：拆分后重新置0模态2的索引

        '''
        # 根据fmri和smri的特征数量，将他们重新分成两个矩阵，以便进行mkl
        index_mod1 = [i for i in self.index if i < mod1_num]
        index_mod2 = [i for i in self.index if (i >= mod1_num) and (i < mod2_num + mod1_num)]

        for i, index in zip(range(len(index_mod2)), index_mod2):
            index_mod2[i] = index - mod1_num
        # index_mod2 = index_mod2 - mod1_num

        return index_mod1, index_mod2

    def three_split(self, mod1_num, mod2_num, mod3_num):
        '''直接拆分两种模态的索引
        输入：
            mod1_num：模态1的数量
            mod2_num：模态2的数量
            mod3_num：模态3的数量

        输出：
            index_mod1：拆分后模态1的索引
            index_mod2：拆分后模态2的索引
            index_mod3：拆分后模态3的索引

        '''
        # 根据三个模态的特征数量，将他们重新分成两个矩阵，以便进行mkl
        index_mod1 = [i for i in self.index if i < mod1_num]
        index_mod2 = [i for i in self.index if (i >= mod1_num) and (i < mod2_num + mod1_num)]
        index_mod3 = [i for i in self.index if (i >= (mod2_num + mod1_num)) and (i < (mod3_num + mod2_num + mod1_num))]

        return index_mod1, index_mod2, index_mod3

    def three_reset(self, mod1_num, mod2_num, mod3_num):
        '''拆分只有三种模态的索引，每种模态的索引重新按照0为起始开始计算
        输入：
            mod1_num：模态1的数量
            mod2_num：模态2的数量
            mod3_num：模态3的数量

        输出：
            index_mod1：拆分后重新置0模态1的索引
            index_mod2：拆分后重新置0模态2的索引
            index_mod3：拆分后重新置0模态3的索引

        '''
        # 根据fmri和smri的特征数量，将他们重新分成两个矩阵，以便进行mkl
        index_mod1 = [i for i in self.index if i < mod1_num]
        index_mod2 = [i for i in self.index if (i >= mod1_num) and (i < mod2_num + mod1_num)]
        index_mod3 = [i for i in self.index if (i >= mod2_num + mod1_num) and (i < mod3_num + mod2_num + mod1_num)]

        for i, index in zip(range(len(index_mod2)), index_mod2):
            index_mod2[i] = index - mod1_num
        # index_mod2 = index_mod2 - mod1_num
        for i, index in zip(range(len(index_mod3)), index_mod3):
            index_mod3[i] = index - mod1_num - mod2_num

        return index_mod1, index_mod2, index_mod3