import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import classification_report
from ml_model import RandomForest
from ml_model import SVM
from mics import classifier_mics

if __name__ == '__main__':
    # 读取数据
    data_alff = pd.read_csv('./pe_feature/DATA_Recog/data_ALFF.csv', sep=',', engine='python')
    data_falff = pd.read_csv('./pe_feature/DATA_Recog/data_fALFF.csv', sep=',', engine='python')
    data_dc = pd.read_csv('./pe_feature/DATA_Recog/data_DC.csv', sep=',', engine='python')
    data_vmhc_global = pd.read_csv('./pe_feature/DATA_Recog/data_global_VMHC.csv', sep=',', engine='python')
    data_vmhc = pd.read_csv('./pe_feature/DATA_Recog/data_NonGlobal_VMHC.csv', sep=',', engine='python')
    data_reho = pd.read_csv('./pe_feature/DATA_Recog/data_ReHo.csv', sep=',', engine='python')
    data_vbm = pd.read_csv('./pe_feature/DATA_Recog/data_VBM.csv', sep=',', engine='python')
    label = pd.read_csv('./pe_feature/DATA_Recog/label.csv', sep=',', engine='python', header=None)
    X = np.array(data.iloc[:, 0:-1])
    y = np.array(label.iloc[:, -1])
    labelName = ['0', '1']
    posLabel = 0

    # print('data column')
    # print(data.columns)

    # 对数据进行标准化
    ss = StandardScaler().fit(data)
    X = ss.fit_transform(data)
    print('X的均值：% s' % X.mean())
    print('X的标准差： % s' % X.std())

    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y)

    '''
    随机森林模型构建及结果，包括baseline和随机参数搜索法结果
    '''
    # 使用随机森林模型分析数据
    path = os.path.join('Result', 'Random Forest')
    rf = RandomForest(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                      labelName=labelName, path=path)
    baselinePath = os.path.join(path, 'baseline')
    pred_train, y_score_train, pred_test, y_score_test = rf.rf_baseline(
        columeName=data.columns,
        n_estimators=100,
        max_features='auto',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True
    )
    y_score_train = y_score_train[:, 1]
    y_score_test = y_score_test[:, 1]

    # 输出RF baseline的ROC和其余几大指标
    mics = classifier_mics(y_train, pred_train, y_score_train, y_test, pred_test, y_score_test, baselinePath)
    auc = mics.draw_roc(pos_index=posLabel)
    acc, precision, recall, f1 = mics.mics_sum()
    print('---------------------------------------------------')
    print('Random Forest Baseline Model-classification report:')
    print(classification_report(y_test, pred_test, target_names=labelName))
    print('auc:%s, acc:%s, precision:%s, recall:%s, f1:%s' % (auc, acc, precision, recall, f1))
    print('---------------------------------------------------')

    # 使用网格搜索法确定最佳参数集合的RF模型
    cross_val = RepeatedKFold(n_splits=3, n_repeats=1, random_state=10)
    # 使用RandomSearchCV来详细搜索参数范围
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
    max_features = ['log2', 'sqrt']
    max_depth = [int(x) for x in np.linspace(start=1, stop=15, num=15)]
    min_samples_split = [int(x) for x in np.linspace(start=2, stop=50, num=10)]
    min_samples_leaf = [int(x) for x in np.linspace(start=2, stop=50, num=10)]
    bootstrap = [True, False]

    # 使用GridSearchCV来详细搜索参数范围
    # n_estimators = [300, 500, 700]
    # max_features = ['sqrt']
    # max_depth = [7, 11, 15]
    # min_samples_split = [7, 23, 44]
    # min_samples_leaf = [18, 28, 34]
    # bootstrap = [False]
    # 构建RF参数搜索模型
    RandomSearchCVPath = os.path.join(path, 'RandomSearchCV')
    pred_train, y_score_train, pred_test, y_score_test = rf.rf_paraSearch(n_estimators, max_features, max_depth,
                                                                          min_samples_split,
                                                                          min_samples_leaf,
                                                                          bootstrap,
                                                                          cross_val=5,
                                                                          search_method='RandomizedSearch')

    # 输出RF参数搜索模型的ROC和其余几大指标
    y_score_train = y_score_train[:, 1]
    y_score_test = y_score_test[:, 1]
    mics = classifier_mics(y_train, pred_train, y_score_train, y_test, pred_test, y_score_test, RandomSearchCVPath)
    auc = mics.draw_roc(pos_index=posLabel)
    acc, precision, recall, f1 = mics.mics_sum()
    print('---------------------------------------------------')
    print('Random Forest RandomSearch Model-classification report:')
    print(classification_report(y_test, pred_test, target_names=labelName))
    print('auc:%s, acc:%s, precision:%s, recall:%s, f1:%s' % (auc, acc, precision, recall, f1))
    print('---------------------------------------------------')
