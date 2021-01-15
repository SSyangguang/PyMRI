import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix


class classifier_mics():
    '''计算准确率等指标
    输入:
        y_train: 训练集特征矩阵对应的标签
        pred_train: 训练集的预测结果，可以计算accuracy, precision, recall, f1
        y_score_train: 训练集预测结果的概率，可以算fpr和tpr，继而绘制ROC
        y_test: 测试集特征矩阵对应的标签
        pred_test: 测试的预测结果，可以计算accuracy, precision, recall, f1
        y_score_test: 测试集预测结果的概率，可以算fpr和tpr，继而绘制ROC
        path: 结果的存储路径

    '''

    def __init__(self, y_train, pred_train, y_score_train, y_test, pred_test, y_score_test, path):
        self.y_train = y_train
        self.y_test = y_test
        self.pred_train = pred_train
        self.pred_test = pred_test
        self.y_score_train = y_score_train
        self.y_score_test = y_score_test
        self.path = path

    def draw_roc(self, pos_index):
        '''绘制ROC，同时将训练集和测试集的ROC绘制在同一图中，可以观察过拟合程度
        输入:
            pos_index: 正类的类别，绘制ROC时候需要进行定义，如果定义反了会导致ROC反向

        输出(均为存储在path路径中的png图片):
            train_roc.png: 训练集的ROC图
            test_roc.png: 测试集的ROC图
            roc.png: 训练集和测试集的ROC绘制在同一张图中

        '''
        os.makedirs(self.path, exist_ok=True)
        # 绘制训练集ROC曲线
        fpr_train, tpr_train, threshold = roc_curve(self.y_train, self.y_score_train, pos_label=pos_index)

        train_img_path = os.path.join(self.path, 'train_roc.png')
        fig_train, axe_train = plt.subplots(ncols=1, nrows=1)
        fig_train.set_size_inches(8, 8)
        axe_train.plot([0, 1], [0, 1], 'r--')
        axe_train.plot(fpr_train, tpr_train)
        axe_train.set_title(label='Training ROC Curve', size=25, weight='bold')
        axe_train.set_ylabel('True Positive Rate')
        axe_train.set_xlabel('False Positive Rate')
        fig_train.savefig(train_img_path)

        # 绘制测试集ROC曲线
        fpr_test, tpr_test, threshold = roc_curve(self.y_test, self.y_score_test, pos_label=pos_index)
        AUC_test = auc(fpr_test, tpr_test)

        test_img_path = os.path.join(self.path, 'test_roc.png')
        fig_test, axe_test = plt.subplots(ncols=1, nrows=1)
        fig_test.set_size_inches(8, 8)
        axe_test.plot([0, 1], [0, 1], 'r--')
        axe_test.plot(fpr_test, tpr_test)
        axe_test.set_title(label='Testing ROC Curve', size=25, weight='bold')
        axe_test.set_ylabel('True Positive Rate')
        axe_test.set_xlabel('False Positive Rate')
        fig_test.savefig(test_img_path)

        # 将训练集和测试集ROC绘制在一起
        test_img_path = os.path.join(self.path, 'roc.png')
        fig_roc, axe_roc = plt.subplots()
        fig_roc.set_size_inches(8, 8)
        axe_roc.plot([0, 1], [0, 1], 'r--')
        axe_roc.plot(fpr_train, tpr_train)
        axe_roc.plot(fpr_test, tpr_test)
        axe_roc.set_title(label='ROC Curve', size=25, weight='bold')
        axe_roc.set_ylabel('True Positive Rate')
        axe_roc.set_xlabel('False Positive Rate')
        axe_roc.legend(["Train", "Test"], loc=0)
        fig_roc.savefig(test_img_path)

        return AUC_test

    def confusion_matrix(self):
        '''就散混淆矩阵
        输出: 混淆矩阵
        '''
        conf_matrix = pd.DataFrame(confusion_matrix(self.y_test, self.pred_test))
        return conf_matrix

    def mics_sum_train(self):
        '''计算训练集的accuracy, precision, recall, f1
        输出:
            acc: float
            precision: float
            recall: float
            f1: float
        '''
        # 计算准确率
        acc = accuracy_score(self.y_train, self.pred_train)
        # 计算精准率
        precision = precision_score(self.y_train, self.pred_train)
        # 计算召回率
        recall = recall_score(self.y_train, self.pred_train)
        # 计算F1值
        f1 = f1_score(self.y_train, self.pred_train)

        return acc, precision, recall, f1

    def mics_sum_test(self):
        '''计算测试集的accuracy, precision, recall, f1
        输出:
            acc: float
            precision: float
            recall: float
            f1: float
        '''
        # 计算准确率
        acc = accuracy_score(self.y_test, self.pred_test)
        # 计算精准率
        precision = precision_score(self.y_test, self.pred_test)
        # 计算召回率
        recall = recall_score(self.y_test, self.pred_test)
        # 计算F1值
        f1 = f1_score(self.y_test, self.pred_test)

        return acc, precision, recall, f1

    def sensitivity(self):
        '''计算指标sensitivity
        输出：
            sensitivity

        '''
        tn_tra, fp_tra, fn_tra, tp_tra = confusion_matrix(self.y_train, self.pred_train).ravel()
        tn_tes, fp_tes, fn_tes, tp_tes = confusion_matrix(self.y_test, self.pred_test).ravel()
        sens_tra = tp_tra / (tp_tra + fn_tra)
        sens_tes = tp_tes / (tp_tes + fn_tes)

        return sens_tra, sens_tes

    def specificity(self):
        '''计算指标sensitivity
        输出：
            specificity

        '''
        tn_tra, fp_tra, fn_tra, tp_tra = confusion_matrix(self.y_train, self.pred_train).ravel()
        tn_tes, fp_tes, fn_tes, tp_tes = confusion_matrix(self.y_test, self.pred_test).ravel()
        spec_tra = tn_tra / (fp_tra + tn_tra)
        spec_tes = tn_tes / (fp_tes + tn_tes)

        return spec_tra, spec_tes

class corr():
    '''用于计算特征之间的相关性
    输入:
        X: 输入的特征矩阵
        col_name: 矩阵对应的特征名称

    '''

    def __init__(self, X, col_name, corr_path):
        self.X = X
        self.name = col_name
        self.path = corr_path

    def corr_heatmap(self, method='pearson'):
        '''绘制特征之间相关性的heatmap，不在heatmap上加值
        输入:
            method: 用于计算特征之间相关性的方法，包括pearson, kendall, spearman
        输出:
            存储一张特征之间相关性的heatmap

        '''
        X_corr = pd.DataFrame(self.X, columns=self.name)
        corr_mat = X_corr.corr(method=method)
        fig, axe = plt.subplots(figsize=(12, 9))
        sns.heatmap(corr_mat, vmax=1, square=True, ax=axe)
        fig.savefig(os.path.join(self.path, 'feature_corr_heatmap.png'))


if __name__ == '__main__':
    pass