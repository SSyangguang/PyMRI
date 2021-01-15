import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind

from utils import normalization, index_split
from sklearn.model_selection import train_test_split

from ml_model import elastic_net, lasso, SVM
from ml_model import mkl_svm
from stat_model import Utest, Ttest
from mics import corr

# plt.style.use('seaborn')
# concatenate独立进行特征选择
'''
if __name__ == '__main__':
    # 设定结果存储路径并创建文件夹
    timeDir = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    crohnPath = os.path.join('crohn', 'dti+fmri', timeDir)
    os.makedirs(crohnPath, exist_ok=True)

    # 读取数据
    dataFalff = pd.read_csv('./mkl_feature_1007/feature_fMRI/mALFF_classic_global.csv', sep=',', engine='python')
    dataFc = pd.read_csv('./mkl_feature_1007/feature_fMRI/FC_feature_90_R_G.csv', sep=',', engine='python')
    dataReho = pd.read_csv('./mkl_feature_1007/feature_fMRI/smReHo_G_90.csv', sep=',', engine='python')

    dataFA = pd.read_csv('./mkl_feature_1007/feature_Native_space/FA_90.csv', sep=',', engine='python')
    dataMD = pd.read_csv('./mkl_feature_1007/feature_Native_space/MD_90.csv', sep=',', engine='python')
    dataRD = pd.read_csv('./mkl_feature_1007/feature_Native_space/RD_90.csv', sep=',', engine='python')
    dataAD = pd.read_csv('./mkl_feature_1007/feature_Native_space/L1_90.csv', sep=',', engine='python')
    # dataL2 = pd.read_csv('./mkl_feature_1007/feature_Native_space/FA_90.csv', sep=',', engine='python')
    # dataL3 = pd.read_csv('./mkl_feature_1007/feature_Native_space/FA_90.csv', sep=',', engine='python')

    dataFaMatrix = pd.read_csv('./mkl_feature_1007/feature_Deterministi/FA_Matrix_90.csv', sep=',', engine='python')
    dataFN = pd.read_csv('./mkl_feature_1007/feature_Deterministi/FN_Matrix_90.csv', sep=',', engine='python')
    dataLength = pd.read_csv('./mkl_feature_1007/feature_Deterministi/Length_Matrix_Mapped_90.csv', sep=',', engine='python')

    dataVbm = pd.read_csv('./mkl_feature_1007/vbm/VBM_90.csv', sep=',', engine='python')

    # 读取label
    label = pd.read_csv('./mkl_feature_1007/label.csv', sep=',', engine='python', header=None)

    # 数据统一转换为numpy格式
    falffNumpy = np.array(dataFalff.iloc[:, :])
    fcNumpy = np.array(dataFc.iloc[:, :])
    rehoNumpy = np.array(dataReho.iloc[:, :])
    faNumpy = np.array(dataFA.iloc[:, :])
    mdNumpy = np.array(dataMD.iloc[:, :])
    rdNumpy = np.array(dataRD.iloc[:, :])
    adNumpy = np.array(dataAD.iloc[:, :])
    # l2Numpy = np.array(dataL2.iloc[:, :])
    # l3Numpy = np.array(dataL3.iloc[:, :])
    faMatrixNumpy = np.array(dataFaMatrix.iloc[:, :])
    fnNumpy = np.array(dataFN.iloc[:, :])
    lengthNumpy = np.array(dataLength.iloc[:, :])
    vbmNumpy = np.array(dataVbm.iloc[:, :])
    # 标签转换为numpy格式
    y = np.array(label.iloc[:, :]).ravel()
    # 规定病人为0，健康人为1
    labelName = ['0', '1']
    posLabel = 1

    # 存储特征名称
    falffName = dataFalff.columns._values
    fcName = dataFc.columns._values
    rehoName = dataReho.columns._values
    faName = dataFA.columns._values
    mdName = dataMD.columns._values
    rdName = dataRD.columns._values
    adName = dataAD.columns._values
    # l2Name = dataL2.columns._values
    # l3Name = dataL3.columns._values
    faMatrixName = dataFaMatrix.columns._values
    fnName = dataFN.columns._values
    lengthName = dataLength.columns._values
    vbmName = dataVbm.columns._values

    # 单独对FC, FA Matrix, FN, Length使用留一法U检验进行特征选择
    fcUPath = os.path.join(crohnPath, 'fc_utest')
    os.makedirs(fcUPath, exist_ok=True)
    utestFC = Utest(fcNumpy, fcName,
                    dis_num=63, hc_num=39, thres=0.01, all_value=False, stat_path=fcUPath)
    selectFC, selectFCName, _, _, fcUIndex = utestFC.utest_loo_freq(K_value=40, choice='percent')

    faMatUPath = os.path.join(crohnPath, 'fa_matrix_utest')
    os.makedirs(faMatUPath, exist_ok=True)
    utestFAMatrix = Utest(faMatrixNumpy, faMatrixName,
                          dis_num=63, hc_num=39, thres=0.1, all_value=False, stat_path=faMatUPath)
    selectFaMat, selectFaMatName, _, _, faMatUIndex = utestFAMatrix.utest_loo_freq(K_value=40, choice='percent')

    fnUPath = os.path.join(crohnPath, 'fn_utest')
    os.makedirs(fnUPath, exist_ok=True)
    utestFN = Utest(fnNumpy, fnName,
                    dis_num=63, hc_num=39, thres=0.1, all_value=False, stat_path=fnUPath)
    selectFN, selectFNName, _, _, fnUIndex = utestFN.utest_loo_freq(K_value=40, choice='percent')

    lengthUPath = os.path.join(crohnPath, 'length_utest')
    os.makedirs(lengthUPath, exist_ok=True)
    utestLength = Utest(lengthNumpy, lengthName,
                        dis_num=63, hc_num=39, thres=0.1, all_value=False, stat_path=lengthUPath)
    selectLength, selectLengthName, _, _, lengthUIndex = utestLength.utest_loo_freq(K_value=40, choice='percent')

    # 对数据进行标准化并拼接
    # 拼接功能数据
    fMRI = np.concatenate((falffNumpy, selectFC, rehoNumpy), axis=1)
    fMRIName = np.concatenate((falffName, selectFCName, rehoName), axis=0)
    # 对功能数据进行z标准化
    nor_fMRI = normalization(fMRI)
    fMRI = nor_fMRI.z_score()
    # 功能数据的特征数量
    fMRI_num = len(fMRIName)

    # 对DTI进行拼接
    DTI = np.concatenate((selectFaMat, selectFN, selectLength, faNumpy, mdNumpy, rdNumpy, adNumpy), axis=1)
    DTIName = np.concatenate((selectFaMatName, selectFNName, selectLengthName, faName, mdName, rdName, adName), axis=0)

    # 对DTI数据进行z标准化
    nor_DTI = normalization(DTI)
    DTI = nor_DTI.z_score()
    # DTI数据的特征数量
    DTI_num = len(DTIName)

    # 对结构数据进行拼接
    sMRI = vbmNumpy
    sMRIName = vbmName
    # 对结构数据进行z标准化
    nor_sMRI = normalization(sMRI)
    sMRI = nor_sMRI.z_score()
    # 结构数据的特征数量
    sMRI_num = len(sMRIName)

    # 直接拼接多模态数据
    multimodal = np.concatenate((fMRI, DTI, sMRI), axis=1)
    multimodalName = np.concatenate((fMRIName, DTIName, sMRIName), axis=0)
    # X为最终的输入
    X = multimodal

    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

    # lasso = lasso(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
    #               feature_name=fMRIName, path=crohnPath, cv_val=False)
    #
    # X_train_new, X_test_new, \
    # featureName, featureFreq, feature_coef, index = lasso.lasso_shuffle(shuffle_time=100,
    #                                                                     alpha_range=0.05)

    # 使用elastic net进行特征选择
    # elas = elastic_net(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
    #                    feature_name=multimodalName, path=crohnPath, cv_val=False)
    #
    # X_train_new, X_test_new, featureName, _, _, index = elas.elasticnet_shuffle(shuffle_time=100,
    #                                                                             l1_range=0.5,
    #                                                                             alphas_range=0.13)

    index = [620, 625, 0, 210, 215, 217, 221, 223, 229, 239, 240, 244, 86, 90, 91, 93, 96, 97, 98, 99,
                  103, 104, 107, 115, 117, 120, 121, 251, 256, 260, 269, 271, 290, 291, 296, 297, 300, 321,
                  323, 480, 489, 564, 565, 142, 181, 200, 730, 735]
    featureName = multimodalName[index]

    # 根据fmri DTI sMRI的特征数量，将他们重新分成两个矩阵，以便进行mkl
    index_mod = index_split(index)
    index_fMRI, index_DTI, index_sMRI = index_mod.three_split(fMRI_num, DTI_num, sMRI_num)
    X_fMRI = X[:, index_fMRI]
    X_DTI = X[:, index_DTI]
    X_sMRI = X[:, index_sMRI]
    # 不进行模态区分，为了对比试验和特征相关性矩阵
    X_new = X[:, index]

    # 使用svm_shuffle返回结果并且绘制ROC曲线的示意图
    svmShufflePath = os.path.join(crohnPath, 'svm_shuffle')
    os.makedirs(svmShufflePath, exist_ok=True)
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_new, y, path=crohnPath)
    train_means, train_std, test_means, test_std, roc_dict = svm.svm_shuffle(outer=10, shuffle_time=svmTime,
                                                                             para={'kernel': 'linear', 'C': 10},
                                                                             svm_metrics=['accuracy',
                                                                                          'sensitivity',
                                                                                          'specificity'])

    # 绘制accuracy随着shuffle次数变化的曲线
    x_axis = np.arange(1, svmTime+1)
    fig_svm, axe_svm = plt.subplots()
    axe_svm.fill_between(x_axis,
                         np.array(test_means['accuracy']) - np.array(test_std['accuracy']),
                         np.array(test_means['accuracy']) + np.array(test_std['accuracy']),
                         alpha=0.2)
    axe_svm.plot(x_axis, np.array(test_means['accuracy']), '--', color='g', alpha=1,
                 label='Train accuracy: % 0.3f' % np.mean(test_means['accuracy']))
    axe_svm.set_ylim([-0.01, 1.01])
    axe_svm.set_xlabel('Shuffle Time')
    axe_svm.set_ylabel('Accuracy')
    axe_svm.set_title('Accuracy for Crohn vs. HC classification')
    axe_svm.legend(loc="lower right", prop={'size': 8})
    fig_svm.savefig(os.path.join(svmShufflePath, 'shuffle_result.png'))

    # 绘制ROC曲线
    std_auc_train = np.std(roc_dict['auc_train'])
    std_auc_test = np.std(roc_dict['auc_test'])

    fig_svm_roc, axe_svm_roc = plt.subplots()
    axe_svm_roc.plot(roc_dict['fpr_train'], roc_dict['tpr_train'],
                     label=r'Train ROC (AUC = %0.2f $\pm$ %0.2f)' % (roc_dict['auc_train'], std_auc_train),
                     lw=2, alpha=.8)
    axe_svm_roc.plot(roc_dict['fpr_test'], roc_dict['tpr_test'],
                     label=r'Test ROC (AUC = %0.2f $\pm$ %0.2f)' % (roc_dict['auc_test'], std_auc_test),
                     lw=2, alpha=.8)

    std_tpr_train = np.std(roc_dict['tpr_list_train'], axis=0)
    std_tpr_test = np.std(roc_dict['tpr_list_test'], axis=0)

    tprs_upper = np.minimum(roc_dict['tpr_test'] + std_tpr_test, 1)
    tprs_lower = np.maximum(roc_dict['tpr_test'] - std_tpr_test, 0)

    axe_svm_roc.fill_between(roc_dict['fpr_test'], tprs_lower, tprs_upper, color='grey', alpha=.2,
                             label=r'$\pm$ 1 std. dev.')

    tprs_upper = np.minimum(roc_dict['tpr_train'] + std_tpr_train, 1)
    tprs_lower = np.maximum(roc_dict['tpr_train'] - std_tpr_train, 0)

    axe_svm_roc.fill_between(roc_dict['fpr_train'], tprs_lower, tprs_upper, color='grey', alpha=.2,
                             label=r'$\pm$ 1 std. dev.')

    axe_svm_roc.grid(True)
    axe_svm_roc.plot([0, 1], [0, 1], 'r--')
    axe_svm_roc.set_xlim([-0.01, 1.01])
    axe_svm_roc.set_ylim([-0.01, 1.01])
    axe_svm_roc.set_xlabel('False Positive Rate', fontsize=15)
    axe_svm_roc.set_ylabel('True Positive Rate', fontsize=15)
    axe_svm_roc.set_title('ROC for CD vs. HC classification', fontsize=18)
    axe_svm_roc.legend(loc="lower right", prop={'size': 8})
    fig_svm_roc.savefig(os.path.join(svmShufflePath, 'shuffle_ROC.png'))

    # 使用mkl返回结果并且绘制ROC曲线的示意图
    svmMklPath = os.path.join(crohnPath, 'mkl_svm')
    os.makedirs(svmMklPath, exist_ok=True)

    mkl_kernel = {'kernel_type1': 'rbf',
                  'kernel_type2': 'rbf',
                  'kernel_type3': 'rbf',
                  'kernel_weight1': 0.3,
                  'kernel_weight2': 0.6,
                  'kernel_weight3': 0.1}
    mkl_para = {'kernel1': 15,
                'kernel2': 50,
                'kernel3': 10,
                'C': 80}
    mkl = mkl_svm(X_fMRI, X_DTI, X_sMRI, y, mkl_path=crohnPath)
    train_means, train_std, test_means, test_std, roc_dict = mkl.mksvm3(kernel_dict=mkl_kernel,
                                                                        para=mkl_para,
                                                                        svm_metrics=['accuracy',
                                                                                     'sensitivity',
                                                                                     'specificity'])

    # mkl-svm绘制accuracy随着shuffle次数变化的曲线
    x_axis = np.arange(1, svmTime+1)
    fig_mkl, axe_mkl = plt.subplots()
    axe_mkl.fill_between(x_axis,
                         np.array(test_means['accuracy']) - np.array(test_std['accuracy']),
                         np.array(test_means['accuracy']) + np.array(test_std['accuracy']),
                         alpha=0.2)
    axe_mkl.plot(x_axis, np.array(test_means['accuracy']), '--', color='g', alpha=1,
                 label='Mean accuracy: % 0.3f' % np.mean(test_means['accuracy']))
    axe_mkl.set_ylim([-0.01, 1.01])
    axe_mkl.set_xlabel('Shuffle Time')
    axe_mkl.set_ylabel('Accuracy')
    axe_mkl.set_title('Accuracy for Crohn vs. HC classification')
    axe_mkl.legend(loc="lower right", prop={'size': 8})
    fig_mkl.savefig(os.path.join(svmMklPath, 'mkl_result.png'))

    # 绘制ROC曲线
    std_auc_train = np.std(roc_dict['auc_train'])
    std_auc_test = np.std(roc_dict['auc_test'])

    fig_mkl_roc, axe_mkl_roc = plt.subplots()
    axe_mkl_roc.plot(roc_dict['fpr_train'], roc_dict['tpr_train'],
                     label=r'Train ROC (AUC = %0.2f $\pm$ %0.2f)' % (roc_dict['auc_train'], std_auc_train),
                     lw=2, alpha=.8)
    axe_mkl_roc.plot(roc_dict['fpr_test'], roc_dict['tpr_test'],
                     label=r'Train ROC (AUC = %0.2f $\pm$ %0.2f)' % (roc_dict['auc_test'], std_auc_test),
                     lw=2, alpha=.8)

    std_tpr_train = np.std(roc_dict['tpr_list_train'], axis=0)
    std_tpr_test = np.std(roc_dict['tpr_list_test'], axis=0)

    tprs_upper = np.minimum(roc_dict['tpr_test'] + std_tpr_test, 1)
    tprs_lower = np.maximum(roc_dict['tpr_test'] - std_tpr_test, 0)

    axe_mkl_roc.fill_between(roc_dict['fpr_test'], tprs_lower, tprs_upper, color='grey', alpha=.2,
                             label=r'$\pm$ 1 std. dev.')

    tprs_upper = np.minimum(roc_dict['tpr_test'] + std_tpr_train, 1)
    tprs_lower = np.maximum(roc_dict['tpr_test'] - std_tpr_train, 0)

    axe_mkl_roc.fill_between(roc_dict['fpr_train'], tprs_lower, tprs_upper, color='grey', alpha=.2,
                             label=r'$\pm$ 1 std. dev.')

    axe_mkl_roc.grid(True)
    axe_mkl_roc.plot([0, 1], [0, 1], 'r--')
    axe_mkl_roc.set_xlim([-0.01, 1.01])
    axe_mkl_roc.set_ylim([-0.01, 1.01])
    axe_mkl_roc.set_xlabel('False Positive Rate', fontsize=15)
    axe_mkl_roc.set_ylabel('True Positive Rate', fontsize=15)
    axe_mkl_roc.set_title('ROC for Crohn vs. HC classification', fontsize=18)
    axe_mkl_roc.legend(loc="lower right", prop={'size': 8})
    fig_mkl_roc.savefig(os.path.join(svmMklPath, 'MKL_ROC.png'))

    # mkl-svm的权重网格搜索法
    train_means, test_means, weight_name = mkl.mksvm3_grid(kernel_dict=mkl_kernel,
                                                          para=mkl_para,
                                                          grid_num=10,
                                                          svm_metrics=['accuracy',
                                                                       'sensitivity',
                                                                       'specificity'])

    # 绘制特征之间的相关性并存储
    corr_mat = corr(X_new, featureName, crohnPath)
    corr_mat.corr_heatmap()

    ###############################
    #####用于每个模态单独索引的分类#####
    ###############################
    mod1_path = os.path.join(crohnPath, 'fMRI')
    mod2_path = os.path.join(crohnPath, 'DTI')
    mod3_path = os.path.join(crohnPath, 'sMRI')
    os.makedirs(mod1_path, exist_ok=True)
    os.makedirs(mod2_path, exist_ok=True)
    os.makedirs(mod3_path, exist_ok=True)

    index_fMRI_only, index_DTI_only, index_sMRI_only = index_mod.three_reset(fMRI_num, DTI_num, sMRI_num)

    X_new1 = fMRI[:, index_fMRI_only]
    X_new2 = DTI[:, index_DTI_only]
    X_new3 = sMRI[:, index_sMRI_only]
    featureName1 = fMRIName[index_fMRI_only]
    featureName2 = DTIName[index_DTI_only]
    featureName3 = sMRIName[index_sMRI_only]

    # 使用svm_shuffle返回fMRI结果并且绘制ROC曲线的示意图
    svmShufflePath_1 = os.path.join(mod1_path, 'svm_shuffle')
    os.makedirs(svmShufflePath_1, exist_ok=True)
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_new1, y, path=svmShufflePath_1)
    train_means, train_std, test_means, test_std, roc_dict = svm.svm_shuffle(outer=10, shuffle_time=svmTime,
                                                                             para={'kernel': 'linear', 'C': 10},
                                                                             svm_metrics=['accuracy',
                                                                                          'sensitivity',
                                                                                          'specificity'])

    # 绘制accuracy随着shuffle次数变化的曲线
    x_axis = np.arange(1, svmTime + 1)
    fig_svm, axe_svm = plt.subplots()
    axe_svm.fill_between(x_axis,
                         np.array(test_means['accuracy']) - np.array(test_std['accuracy']),
                         np.array(test_means['accuracy']) + np.array(test_std['accuracy']),
                         alpha=0.2)
    axe_svm.plot(x_axis, np.array(test_means['accuracy']), '--', color='g', alpha=1,
                 label='Mean accuracy: % 0.3f' % np.mean(test_means['accuracy']))
    axe_svm.set_ylim([-0.01, 1.01])
    axe_svm.set_xlabel('Shuffle Time')
    axe_svm.set_ylabel('Accuracy')
    axe_svm.set_title('Accuracy for Crohn vs. HC classification')
    axe_svm.legend(loc="lower right", prop={'size': 8})
    fig_svm.savefig(os.path.join(svmShufflePath_1, 'shuffle_result.png'))

    # 绘制ROC曲线
    std_auc_train = np.std(roc_dict['auc_train'])
    std_auc_test = np.std(roc_dict['auc_test'])

    fig_svm_roc, axe_svm_roc = plt.subplots()
    axe_svm_roc.plot(roc_dict['fpr_train'], roc_dict['tpr_train'],
                     label=r'Trian ROC (AUC = %0.2f $\pm$ %0.2f)' % (roc_dict['auc_train'], std_auc_train),
                     lw=2, alpha=.8)
    axe_svm_roc.plot(roc_dict['fpr_test'], roc_dict['tpr_test'],
                     label=r'Test ROC (AUC = %0.2f $\pm$ %0.2f)' % (roc_dict['auc_test'], std_auc_test),
                     lw=2, alpha=.8)

    std_tpr_train = np.std(roc_dict['tpr_list_train'], axis=0)
    std_tpr_test = np.std(roc_dict['tpr_list_test'], axis=0)

    tprs_upper = np.minimum(roc_dict['tpr_test'] + std_tpr_test, 1)
    tprs_lower = np.maximum(roc_dict['tpr_test'] - std_tpr_test, 0)

    axe_svm_roc.fill_between(roc_dict['fpr_test'], tprs_lower, tprs_upper, color='grey', alpha=.2,
                             label=r'$\pm$ 1 std. dev.')

    tprs_upper = np.minimum(roc_dict['tpr_train'] + std_tpr_train, 1)
    tprs_lower = np.maximum(roc_dict['tpr_train'] - std_tpr_train, 0)

    axe_svm_roc.fill_between(roc_dict['fpr_train'], tprs_lower, tprs_upper, color='grey', alpha=.2,
                             label=r'$\pm$ 1 std. dev.')

    axe_svm_roc.grid(True)
    axe_svm_roc.plot([0, 1], [0, 1], 'r--')
    axe_svm_roc.set_xlim([-0.01, 1.01])
    axe_svm_roc.set_ylim([-0.01, 1.01])
    axe_svm_roc.set_xlabel('False Positive Rate', fontsize=15)
    axe_svm_roc.set_ylabel('True Positive Rate', fontsize=15)
    axe_svm_roc.set_title('ROC for Crohn vs. HC classification', fontsize=18)
    axe_svm_roc.legend(loc="lower right", prop={'size': 8})
    fig_svm_roc.savefig(os.path.join(svmShufflePath_1, 'shuffle_ROC.png'))

    # 使用svm_shuffle返回DTI结果并且绘制ROC曲线的示意图
    svmShufflePath_2 = os.path.join(mod2_path, 'svm_shuffle')
    os.makedirs(svmShufflePath_2, exist_ok=True)
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_new2, y, path=svmShufflePath_2)
    train_means, train_std, test_means, test_std, roc_dict = svm.svm_shuffle(outer=10, shuffle_time=svmTime,
                                                                             para={'kernel': 'linear', 'C': 10},
                                                                             svm_metrics=['accuracy',
                                                                                          'sensitivity',
                                                                                          'specificity'])

    # 绘制accuracy随着shuffle次数变化的曲线
    x_axis = np.arange(1, svmTime + 1)
    fig_svm, axe_svm = plt.subplots()
    axe_svm.fill_between(x_axis,
                         np.array(test_means['accuracy']) - np.array(test_std['accuracy']),
                         np.array(test_means['accuracy']) + np.array(test_std['accuracy']),
                         alpha=0.2)
    axe_svm.plot(x_axis, np.array(test_means['accuracy']), '--', color='g', alpha=1,
                 label='Mean accuracy: % 0.3f' % np.mean(test_means['accuracy']))
    axe_svm.set_ylim([-0.01, 1.01])
    axe_svm.set_xlabel('Shuffle Time')
    axe_svm.set_ylabel('Accuracy')
    axe_svm.set_title('Accuracy for Crohn vs. HC classification')
    axe_svm.legend(loc="lower right", prop={'size': 8})
    fig_svm.savefig(os.path.join(svmShufflePath_2, 'shuffle_result.png'))

    # 绘制ROC曲线
    std_auc_train = np.std(roc_dict['auc_train'])
    std_auc_test = np.std(roc_dict['auc_test'])

    fig_svm_roc, axe_svm_roc = plt.subplots()
    axe_svm_roc.plot(roc_dict['fpr_train'], roc_dict['tpr_train'],
                     label=r'Train ROC (AUC = %0.2f $\pm$ %0.2f)' % (roc_dict['auc_train'], std_auc_train),
                     lw=2, alpha=.8)
    axe_svm_roc.plot(roc_dict['fpr_test'], roc_dict['tpr_test'],
                     label=r'Test ROC (AUC = %0.2f $\pm$ %0.2f)' % (roc_dict['auc_test'], std_auc_test),
                     lw=2, alpha=.8)

    std_tpr_train = np.std(roc_dict['tpr_list_train'], axis=0)
    std_tpr_test = np.std(roc_dict['tpr_list_test'], axis=0)

    tprs_upper = np.minimum(roc_dict['tpr_test'] + std_tpr_test, 1)
    tprs_lower = np.maximum(roc_dict['tpr_test'] - std_tpr_test, 0)

    axe_svm_roc.fill_between(roc_dict['fpr_test'], tprs_lower, tprs_upper, color='grey', alpha=.2,
                             label=r'$\pm$ 1 std. dev.')

    tprs_upper = np.minimum(roc_dict['tpr_train'] + std_tpr_train, 1)
    tprs_lower = np.maximum(roc_dict['tpr_train'] - std_tpr_train, 0)

    axe_svm_roc.fill_between(roc_dict['fpr_train'], tprs_lower, tprs_upper, color='grey', alpha=.2,
                             label=r'$\pm$ 1 std. dev.')

    axe_svm_roc.grid(True)
    axe_svm_roc.plot([0, 1], [0, 1], 'r--')
    axe_svm_roc.set_xlim([-0.01, 1.01])
    axe_svm_roc.set_ylim([-0.01, 1.01])
    axe_svm_roc.set_xlabel('False Positive Rate', fontsize=15)
    axe_svm_roc.set_ylabel('True Positive Rate', fontsize=15)
    axe_svm_roc.set_title('ROC for Crohn vs. HC classification', fontsize=18)
    axe_svm_roc.legend(loc="lower right", prop={'size': 8})
    fig_svm_roc.savefig(os.path.join(svmShufflePath_2, 'shuffle_ROC.png'))

    # 使用svm_shuffle返回sMRI结果并且绘制ROC曲线的示意图
    svmShufflePath_3 = os.path.join(mod3_path, 'svm_shuffle')
    os.makedirs(svmShufflePath_3, exist_ok=True)
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_new3, y, path=svmShufflePath_3)
    train_means, train_std, test_means, test_std, roc_dict = svm.svm_shuffle(outer=10, shuffle_time=svmTime,
                                                                             para={'kernel': 'linear', 'C': 10},
                                                                             svm_metrics=['accuracy',
                                                                                          'sensitivity',
                                                                                          'specificity'])

    # 绘制accuracy随着shuffle次数变化的曲线
    x_axis = np.arange(1, svmTime + 1)
    fig_svm, axe_svm = plt.subplots()
    axe_svm.fill_between(x_axis,
                         np.array(test_means['accuracy']) - np.array(test_std['accuracy']),
                         np.array(test_means['accuracy']) + np.array(test_std['accuracy']),
                         alpha=0.2)
    axe_svm.plot(x_axis, np.array(test_means['accuracy']), '--', color='g', alpha=1,
                 label='Mean accuracy: % 0.3f' % np.mean(test_means['accuracy']))
    axe_svm.set_ylim([-0.01, 1.01])
    axe_svm.set_xlabel('Shuffle Time')
    axe_svm.set_ylabel('Accuracy')
    axe_svm.set_title('Accuracy for Crohn vs. HC classification')
    axe_svm.legend(loc="lower right", prop={'size': 8})
    fig_svm.savefig(os.path.join(svmShufflePath_3, 'shuffle_result.png'))

    # 绘制ROC曲线
    std_auc_train = np.std(roc_dict['auc_train'])
    std_auc_test = np.std(roc_dict['auc_test'])

    fig_svm_roc, axe_svm_roc = plt.subplots()
    axe_svm_roc.plot(roc_dict['fpr_train'], roc_dict['tpr_train'],
                     label=r'Train ROC (AUC = %0.2f $\pm$ %0.2f)' % (roc_dict['auc_train'], std_auc_train),
                     lw=2, alpha=.8)
    axe_svm_roc.plot(roc_dict['fpr_test'], roc_dict['tpr_test'],
                     label=r'Test ROC (AUC = %0.2f $\pm$ %0.2f)' % (roc_dict['auc_test'], std_auc_test),
                     lw=2, alpha=.8)

    std_tpr_train = np.std(roc_dict['tpr_list_train'], axis=0)
    std_tpr_test = np.std(roc_dict['tpr_list_test'], axis=0)

    tprs_upper = np.minimum(roc_dict['tpr_test'] + std_tpr_test, 1)
    tprs_lower = np.maximum(roc_dict['tpr_test'] - std_tpr_test, 0)

    axe_svm_roc.fill_between(roc_dict['fpr_test'], tprs_lower, tprs_upper, color='grey', alpha=.2,
                             label=r'$\pm$ 1 std. dev.')

    tprs_upper = np.minimum(roc_dict['tpr_train'] + std_tpr_train, 1)
    tprs_lower = np.maximum(roc_dict['tpr_train'] - std_tpr_train, 0)

    axe_svm_roc.fill_between(roc_dict['fpr_train'], tprs_lower, tprs_upper, color='grey', alpha=.2,
                             label=r'$\pm$ 1 std. dev.')

    axe_svm_roc.grid(True)
    axe_svm_roc.plot([0, 1], [0, 1], 'r--')
    axe_svm_roc.set_xlim([-0.01, 1.01])
    axe_svm_roc.set_ylim([-0.01, 1.01])
    axe_svm_roc.set_xlabel('False Positive Rate', fontsize=15)
    axe_svm_roc.set_ylabel('True Positive Rate', fontsize=15)
    axe_svm_roc.set_title('ROC for Crohn vs. HC classification', fontsize=18)
    axe_svm_roc.legend(loc="lower right", prop={'size': 8})
    fig_svm_roc.savefig(os.path.join(svmShufflePath_3, 'shuffle_ROC.png'))

    corr_mat = corr(X_new1, featureName1, mod1_path)
    corr_mat.corr_heatmap()
    corr_mat = corr(X_new2, featureName2, mod2_path)
    corr_mat.corr_heatmap()
    corr_mat = corr(X_new3, featureName3, mod3_path)
    corr_mat.corr_heatmap()
'''

# 多模态的索引直接进行分类
'''
if __name__ == '__main__':
    # 设定结果存储路径并创建文件夹
    timeDir = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    crohnPath = os.path.join('crohn', 'dti+fmri', timeDir)
    os.makedirs(crohnPath, exist_ok=True)

    # 读取数据
    dataFalff = pd.read_csv('./mkl_feature_1007/feature_fMRI/mALFF_classic_global.csv', sep=',', engine='python')
    dataFc = pd.read_csv('./mkl_feature_1007/feature_fMRI/FC_feature_90_R_G.csv', sep=',', engine='python')
    dataReho = pd.read_csv('./mkl_feature_1007/feature_fMRI/smReHo_G_90.csv', sep=',', engine='python')

    dataFA = pd.read_csv('./mkl_feature_1007/feature_Native_space/FA_90.csv', sep=',', engine='python')
    dataMD = pd.read_csv('./mkl_feature_1007/feature_Native_space/MD_90.csv', sep=',', engine='python')
    dataRD = pd.read_csv('./mkl_feature_1007/feature_Native_space/RD_90.csv', sep=',', engine='python')
    dataAD = pd.read_csv('./mkl_feature_1007/feature_Native_space/L1_90.csv', sep=',', engine='python')
    # dataL2 = pd.read_csv('./mkl_feature_1007/feature_Native_space/FA_90.csv', sep=',', engine='python')
    # dataL3 = pd.read_csv('./mkl_feature_1007/feature_Native_space/FA_90.csv', sep=',', engine='python')

    dataFaMatrix = pd.read_csv('./mkl_feature_1007/feature_Deterministi/FA_Matrix_90.csv', sep=',', engine='python')
    dataFN = pd.read_csv('./mkl_feature_1007/feature_Deterministi/FN_Matrix_90.csv', sep=',', engine='python')
    dataLength = pd.read_csv('./mkl_feature_1007/feature_Deterministi/Length_Matrix_Mapped_90.csv', sep=',', engine='python')

    # 读取label
    label = pd.read_csv('./mkl_feature_1007/label.csv', sep=',', engine='python', header=None)

    # 数据统一转换为numpy格式
    falffNumpy = np.array(dataFalff.iloc[:, :])
    fcNumpy = np.array(dataFc.iloc[:, :])
    rehoNumpy = np.array(dataReho.iloc[:, :])
    faNumpy = np.array(dataFA.iloc[:, :])
    mdNumpy = np.array(dataMD.iloc[:, :])
    rdNumpy = np.array(dataRD.iloc[:, :])
    adNumpy = np.array(dataAD.iloc[:, :])
    # l2Numpy = np.array(dataL2.iloc[:, :])
    # l3Numpy = np.array(dataL3.iloc[:, :])
    faMatrixNumpy = np.array(dataFaMatrix.iloc[:, :])
    fnNumpy = np.array(dataFN.iloc[:, :])
    lengthNumpy = np.array(dataLength.iloc[:, :])
    # 标签转换为numpy格式
    y = np.array(label.iloc[:, :]).ravel()
    # 规定病人为0，健康人为1
    labelName = ['0', '1']
    posLabel = 1

    # 存储特征名称
    falffName = dataFalff.columns._values
    fcName = dataFc.columns._values
    rehoName = dataReho.columns._values
    faName = dataFA.columns._values
    mdName = dataMD.columns._values
    rdName = dataRD.columns._values
    adName = dataAD.columns._values
    # l2Name = dataL2.columns._values
    # l3Name = dataL3.columns._values
    faMatrixName = dataFaMatrix.columns._values
    fnName = dataFN.columns._values
    lengthName = dataLength.columns._values

    # 单独对FC, FA Matrix, FN, Length使用留一法U检验进行特征选择
    fcUPath = os.path.join(crohnPath, 'fc_utest')
    os.makedirs(fcUPath, exist_ok=True)
    utestFC = Utest(fcNumpy, fcName,
                    dis_num=63, hc_num=39, thres=0.01, all_value=False, stat_path=fcUPath)
    selectFC, selectFCName, _, _, fcUIndex = utestFC.utest_loo_freq(K_value=40, choice='percent')

    faMatUPath = os.path.join(crohnPath, 'fa_matrix_utest')
    os.makedirs(faMatUPath, exist_ok=True)
    utestFAMatrix = Utest(faMatrixNumpy, faMatrixName,
                          dis_num=63, hc_num=39, thres=0.1, all_value=False, stat_path=faMatUPath)
    selectFaMat, selectFaMatName, _, _, faMatUIndex = utestFAMatrix.utest_loo_freq(K_value=40, choice='percent')

    fnUPath = os.path.join(crohnPath, 'fn_utest')
    os.makedirs(fnUPath, exist_ok=True)
    utestFN = Utest(fnNumpy, fnName,
                    dis_num=63, hc_num=39, thres=0.1, all_value=False, stat_path=fnUPath)
    selectFN, selectFNName, _, _, fnUIndex = utestFN.utest_loo_freq(K_value=40, choice='percent')

    lengthUPath = os.path.join(crohnPath, 'length_utest')
    os.makedirs(lengthUPath, exist_ok=True)
    utestLength = Utest(lengthNumpy, lengthName,
                        dis_num=63, hc_num=39, thres=0.1, all_value=False, stat_path=lengthUPath)
    selectLength, selectLengthName, _, _, lengthUIndex = utestLength.utest_loo_freq(K_value=40, choice='percent')

    # 对数据进行标准化并拼接
    # 拼接功能数据
    fMRI = np.concatenate((falffNumpy, selectFC, rehoNumpy), axis=1)
    fMRIName = np.concatenate((falffName, selectFCName, rehoName), axis=0)
    # 对功能数据进行z标准化
    nor_fMRI = normalization(fMRI)
    fMRI = nor_fMRI.z_score()
    # 功能数据的特征数量
    fMRI_num = len(fMRIName)

    # 对结构进行拼接
    sMRI = np.concatenate((selectFaMat, selectFN, selectLength, faNumpy, mdNumpy, rdNumpy, adNumpy), axis=1)
    sMRIName = np.concatenate((selectFaMatName, selectFNName, selectLengthName, faName, mdName, rdName, adName), axis=0)

    # 对结构数据进行z标准化
    nor_sMRI = normalization(sMRI)
    sMRI = nor_sMRI.z_score()
    # 结构数据的特征数量
    sMRI_num = len(sMRIName)

    # 直接拼接多模态数据
    multimodal = np.concatenate((fMRI, sMRI), axis=1)
    multimodalName = np.concatenate((fMRIName, sMRIName), axis=0)
    # X为最终的输入
    X = multimodal

    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

    index = [620, 625, 0, 210, 215, 217, 221, 223, 229, 239, 240, 244, 86, 90, 91, 93, 96, 97, 98, 99,
             103, 104, 107, 115, 117, 120, 121, 251, 256, 260, 269, 271, 290, 291, 296, 297, 300, 321,
             323, 480, 489, 564, 565, 142, 173, 181, 200]
    featureName = multimodalName[index]

    # 根据fmri和smri的特征数量，将他们重新分成两个矩阵，以便进行mkl

    index_mod = index_split(index)
    index_fMRI, index_sMRI = index_mod.two_split(fMRI_num, sMRI_num)
    X_fMRI = X[:, index_fMRI]
    X_sMRI = X[:, index_sMRI]
    # 不进行模态区分，为了对比试验和特征相关性矩阵
    X_new = X[:, index]

    ###############################
    #####用于每个模态单独索引的分类#####
    ###############################
    mod1_path = os.path.join(crohnPath, 'fMRI')
    mod2_path = os.path.join(crohnPath, 'sMRI')
    os.makedirs(mod1_path, exist_ok=True)
    os.makedirs(mod2_path, exist_ok=True)

    index_fMRI_only, index_sMRI_only = index_mod.two_reset(fMRI_num, sMRI_num)

    X_new1 = fMRI[:, index_fMRI_only]
    X_new2 = sMRI[:, index_sMRI_only]
    featureName1 = fMRIName[index_fMRI_only]
    featureName2 = sMRIName[index_sMRI_only]

    # 使用svm_shuffle返回fMRI结果并且绘制ROC曲线的示意图
    svmShufflePath_1 = os.path.join(mod1_path, 'svm_shuffle')
    os.makedirs(svmShufflePath_1, exist_ok=True)
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_new1, y, path=svmShufflePath_1)
    train_means, train_std, test_means, test_std, roc_dict = svm.svm_shuffle(outer=10, shuffle_time=svmTime,
                                                                             para={'kernel': 'linear',
                                                                                   'gamma': 50,
                                                                                   'C': 80},
                                                                             svm_metrics=['accuracy',
                                                                                          'sensitivity',
                                                                                          'specificity'])

    # 绘制accuracy随着shuffle次数变化的曲线
    x_axis = np.arange(1, svmTime + 1)
    fig_svm, axe_svm = plt.subplots()
    axe_svm.fill_between(x_axis,
                         np.array(test_means['accuracy']) - np.array(test_std['accuracy']),
                         np.array(test_means['accuracy']) + np.array(test_std['accuracy']),
                         alpha=0.2)
    axe_svm.plot(x_axis, np.array(test_means['accuracy']), '--', color='g', alpha=1,
                 label='Mean accuracy: % 0.3f' % np.mean(test_means['accuracy']))
    axe_svm.set_ylim([-0.01, 1.01])
    axe_svm.set_xlabel('Shuffle Time')
    axe_svm.set_ylabel('Accuracy')
    axe_svm.set_title('Accuracy for Crohn vd. HC classification')
    axe_svm.legend(loc="lower right", prop={'size': 8})
    fig_svm.savefig(os.path.join(svmShufflePath_1, 'shuffle_result.png'))

    # 绘制ROC曲线
    std_auc_train = np.std(roc_dict['auc_train'])
    std_auc_test = np.std(roc_dict['auc_test'])

    fig_svm_roc, axe_svm_roc = plt.subplots()
    axe_svm_roc.plot(roc_dict['fpr_train'], roc_dict['tpr_train'],
                     label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (roc_dict['auc_train'], std_auc_train),
                     lw=2, alpha=.8)
    axe_svm_roc.plot(roc_dict['fpr_test'], roc_dict['tpr_test'],
                     label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (roc_dict['auc_test'], std_auc_test),
                     lw=2, alpha=.8)

    std_tpr_train = np.std(roc_dict['tpr_list_train'], axis=0)
    std_tpr_test = np.std(roc_dict['tpr_list_test'], axis=0)

    tprs_upper = np.minimum(roc_dict['tpr_test'] + std_tpr_test, 1)
    tprs_lower = np.maximum(roc_dict['tpr_test'] - std_tpr_test, 0)

    axe_svm_roc.fill_between(roc_dict['fpr_test'], tprs_lower, tprs_upper, color='grey', alpha=.2,
                             label=r'$\pm$ 1 std. dev.')

    tprs_upper = np.minimum(roc_dict['tpr_train'] + std_tpr_train, 1)
    tprs_lower = np.maximum(roc_dict['tpr_train'] - std_tpr_train, 0)

    axe_svm_roc.fill_between(roc_dict['fpr_train'], tprs_lower, tprs_upper, color='grey', alpha=.2,
                             label=r'$\pm$ 1 std. dev.')

    axe_svm_roc.grid(True)
    axe_svm_roc.plot([0, 1], [0, 1], 'r--')
    axe_svm_roc.set_xlim([-0.01, 1.01])
    axe_svm_roc.set_ylim([-0.01, 1.01])
    axe_svm_roc.set_xlabel('False Positive Rate', fontsize=15)
    axe_svm_roc.set_ylabel('True Positive Rate', fontsize=15)
    axe_svm_roc.set_title('ROC for Crohn vd. HC classification', fontsize=18)
    axe_svm_roc.legend(loc="lower right", prop={'size': 8})
    fig_svm_roc.savefig(os.path.join(svmShufflePath_1, 'shuffle_ROC.png'))

    # # 使用svm_shuffle返回sMRI结果并且绘制ROC曲线的示意图
    # svmShufflePath_2 = os.path.join(mod2_path, 'svm_shuffle')
    # os.makedirs(svmShufflePath_2, exist_ok=True)
    # # 设置参数
    # para = {'kernel': ['rbf'],
    #         'gamma': np.arange(0.1, 1, 0.1),
    #         'C': np.arange(1, 100, 1)}
    # svmTime = 100
    # svm = SVM(X_new2, y, path=svmShufflePath_2)
    # train_means, train_std, test_means, test_std, roc_dict = svm.svm_shuffle(outer=10, shuffle_time=svmTime,
    #                                                                          para={'kernel': 'linear', 'C': 10},
    #                                                                          svm_metrics=['accuracy',
    #                                                                                       'sensitivity',
    #                                                                                       'specificity'])
    #
    # # 绘制accuracy随着shuffle次数变化的曲线
    # x_axis = np.arange(1, svmTime + 1)
    # fig_svm, axe_svm = plt.subplots()
    # axe_svm.fill_between(x_axis,
    #                      np.array(test_means['accuracy']) - np.array(test_std['accuracy']),
    #                      np.array(test_means['accuracy']) + np.array(test_std['accuracy']),
    #                      alpha=0.2)
    # axe_svm.plot(x_axis, np.array(test_means['accuracy']), '--', color='g', alpha=1,
    #              label='Mean accuracy: % 0.3f' % np.mean(test_means['accuracy']))
    # axe_svm.set_ylim([-0.01, 1.01])
    # axe_svm.set_xlabel('Shuffle Time')
    # axe_svm.set_ylabel('Accuracy')
    # axe_svm.set_title('Accuracy for Crohn vd. HC classification')
    # axe_svm.legend(loc="lower right", prop={'size': 8})
    # fig_svm.savefig(os.path.join(svmShufflePath_2, 'shuffle_result.png'))
    #
    # # 绘制ROC曲线
    # std_auc_train = np.std(roc_dict['auc_train'])
    # std_auc_test = np.std(roc_dict['auc_test'])
    #
    # fig_svm_roc, axe_svm_roc = plt.subplots()
    # axe_svm_roc.plot(roc_dict['fpr_train'], roc_dict['tpr_train'],
    #                  label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (roc_dict['auc_train'], std_auc_train),
    #                  lw=2, alpha=.8)
    # axe_svm_roc.plot(roc_dict['fpr_test'], roc_dict['tpr_test'],
    #                  label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (roc_dict['auc_test'], std_auc_test),
    #                  lw=2, alpha=.8)
    #
    # std_tpr_train = np.std(roc_dict['tpr_list_train'], axis=0)
    # std_tpr_test = np.std(roc_dict['tpr_list_test'], axis=0)
    #
    # tprs_upper = np.minimum(roc_dict['tpr_test'] + std_tpr_test, 1)
    # tprs_lower = np.maximum(roc_dict['tpr_test'] - std_tpr_test, 0)
    #
    # axe_svm_roc.fill_between(roc_dict['fpr_test'], tprs_lower, tprs_upper, color='grey', alpha=.2,
    #                          label=r'$\pm$ 1 std. dev.')
    #
    # tprs_upper = np.minimum(roc_dict['tpr_train'] + std_tpr_train, 1)
    # tprs_lower = np.maximum(roc_dict['tpr_train'] - std_tpr_train, 0)
    #
    # axe_svm_roc.fill_between(roc_dict['fpr_train'], tprs_lower, tprs_upper, color='grey', alpha=.2,
    #                          label=r'$\pm$ 1 std. dev.')
    #
    # axe_svm_roc.grid(True)
    # axe_svm_roc.plot([0, 1], [0, 1], 'r--')
    # axe_svm_roc.set_xlim([-0.01, 1.01])
    # axe_svm_roc.set_ylim([-0.01, 1.01])
    # axe_svm_roc.set_xlabel('False Positive Rate', fontsize=15)
    # axe_svm_roc.set_ylabel('True Positive Rate', fontsize=15)
    # axe_svm_roc.set_title('ROC for Crohn vd. HC classification', fontsize=18)
    # axe_svm_roc.legend(loc="lower right", prop={'size': 8})
    # fig_svm_roc.savefig(os.path.join(svmShufflePath_2, 'shuffle_ROC.png'))
    #
    # corr_mat = corr(X_new1, featureName1, mod1_path)
    # corr_mat.corr_heatmap()
    # corr_mat = corr(X_new2, featureName2, mod2_path)
    # corr_mat.corr_heatmap()
'''

# fMRI单模态
'''
if __name__ == '__main__':
    # 设定结果存储路径并创建文件夹
    timeDir = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    crohnPath = os.path.join('crohn', 'fmri', timeDir)
    os.makedirs(crohnPath, exist_ok=True)

    # 读取数据
    dataFalff = pd.read_csv('./mkl_feature_1007/feature_fMRI/mALFF_classic_global.csv', sep=',', engine='python')
    dataFc = pd.read_csv('./mkl_feature_1007/feature_fMRI/FC_feature_90_R_G.csv', sep=',', engine='python')
    dataReho = pd.read_csv('./mkl_feature_1007/feature_fMRI/smReHo_G_90.csv', sep=',', engine='python')

    # 读取label
    label = pd.read_csv('./mkl_feature_1007/label.csv', sep=',', engine='python', header=None)

    # 数据统一转换为numpy格式
    falffNumpy = np.array(dataFalff.iloc[:, :])
    fcNumpy = np.array(dataFc.iloc[:, :])
    rehoNumpy = np.array(dataReho.iloc[:, :])
    # 标签转换为numpy格式
    y = np.array(label.iloc[:, :]).ravel()
    # 规定病人为0，健康人为1
    labelName = ['0', '1']
    posLabel = 1

    # 存储特征名称
    falffName = dataFalff.columns._values
    fcName = dataFc.columns._values
    rehoName = dataReho.columns._values

    # 单独对FC使用留一法U检验进行特征选择
    fcUPath = os.path.join(crohnPath, 'fc_utest')
    os.makedirs(fcUPath, exist_ok=True)
    utestFC = Utest(fcNumpy, fcName,
                    dis_num=63, hc_num=39, thres=0.01, all_value=False, stat_path=fcUPath)
    selectFC, selectFCName, _, _, fcUIndex = utestFC.utest_loo_freq(K_value=40, choice='percent')

    # 对数据进行标准化并拼接
    # 拼接功能数据
    fMRI = np.concatenate((falffNumpy, selectFC, rehoNumpy), axis=1)
    fMRIName = np.concatenate((falffName, selectFCName, rehoName), axis=0)
    # 对功能数据进行z标准化
    nor_fMRI = normalization(fMRI)
    fMRI = nor_fMRI.z_score()
    # 功能数据的特征数量
    fMRI_num = len(fMRIName)

    # X为最终的输入
    X = fMRI

    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

    # lasso = lasso(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
    #               feature_name=fMRIName, path=crohnPath, cv_val=True)
    #
    # X_train_new, X_test_new, \
    # featureName, feature_freq, feature_coef, index = lasso.lasso_shuffle(shuffle_time=100,
    #                                                                      alpha_range=np.arange(0.1, 0.4, 0.01))

    # 使用elastic net进行特征选择
    elas = elastic_net(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                       feature_name=fMRIName, path=crohnPath, cv_val=False)

    X_train_new, X_test_new, featureName, _, _, index = elas.elasticnet_shuffle(shuffle_time=100,
                                                                                l1_range=1.0,
                                                                                alphas_range=0.05)

    # index = [0, 10, 17, 77, 85, 86, 91, 93, 94, 97, 98, 107, 110, 112, 115, 117, 120, 123, 143, 145, 147, 179]

    X_new = X[:, index]
    featureName = fMRIName[index]

    # 使用svm_shuffle返回结果并且绘制ROC曲线的示意图
    svmShufflePath = os.path.join(crohnPath, 'svm_shuffle')
    os.makedirs(svmShufflePath, exist_ok=True)
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_new, y, path=crohnPath)
    train_means, train_std, test_means, test_std, roc_dict = svm.svm_shuffle(outer=10, shuffle_time=svmTime,
                                                                             para={'kernel': 'rbf',
                                                                                   'gamma': 15,
                                                                                   'C': 80},
                                                                             svm_metrics=['accuracy',
                                                                                          'sensitivity',
                                                                                          'specificity'])

    # 绘制accuracy随着shuffle次数变化的曲线
    x_axis = np.arange(1, svmTime+1)
    fig_svm, axe_svm = plt.subplots()
    axe_svm.fill_between(x_axis,
                         np.array(test_means['accuracy']) - np.array(test_std['accuracy']),
                         np.array(test_means['accuracy']) + np.array(test_std['accuracy']),
                         alpha=0.2)
    axe_svm.plot(x_axis, np.array(test_means['accuracy']), '--', color='g', alpha=1,
                 label='Mean accuracy: % 0.3f' % np.mean(test_means['accuracy']))
    axe_svm.set_ylim([-0.01, 1.01])
    axe_svm.set_xlabel('Shuffle Time')
    axe_svm.set_ylabel('Accuracy')
    axe_svm.set_title('Accuracy for Crohn vd. HC classification')
    axe_svm.legend(loc="lower right", prop={'size': 8})
    fig_svm.savefig(os.path.join(svmShufflePath, 'shuffle_result.png'))

    # 绘制ROC曲线
    std_auc_train = np.std(roc_dict['auc_train'])
    std_auc_test = np.std(roc_dict['auc_test'])

    fig_svm_roc, axe_svm_roc = plt.subplots()
    axe_svm_roc.plot(roc_dict['fpr_train'], roc_dict['tpr_train'],
                     label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (roc_dict['auc_train'], std_auc_train),
                     lw=2, alpha=.8)
    axe_svm_roc.plot(roc_dict['fpr_test'], roc_dict['tpr_test'],
                     label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (roc_dict['auc_test'], std_auc_test),
                     lw=2, alpha=.8)

    std_tpr_train = np.std(roc_dict['tpr_list_train'], axis=0)
    std_tpr_test = np.std(roc_dict['tpr_list_test'], axis=0)

    tprs_upper = np.minimum(roc_dict['tpr_test'] + std_tpr_test, 1)
    tprs_lower = np.maximum(roc_dict['tpr_test'] - std_tpr_test, 0)

    axe_svm_roc.fill_between(roc_dict['fpr_test'], tprs_lower, tprs_upper, color='grey', alpha=.2,
                             label=r'$\pm$ 1 std. dev.')

    tprs_upper = np.minimum(roc_dict['tpr_train'] + std_tpr_train, 1)
    tprs_lower = np.maximum(roc_dict['tpr_train'] - std_tpr_train, 0)

    axe_svm_roc.fill_between(roc_dict['fpr_train'], tprs_lower, tprs_upper, color='grey', alpha=.2,
                             label=r'$\pm$ 1 std. dev.')

    axe_svm_roc.grid(True)
    axe_svm_roc.plot([0, 1], [0, 1], 'r--')
    axe_svm_roc.set_xlim([-0.01, 1.01])
    axe_svm_roc.set_ylim([-0.01, 1.01])
    axe_svm_roc.set_xlabel('False Positive Rate', fontsize=15)
    axe_svm_roc.set_ylabel('True Positive Rate', fontsize=15)
    axe_svm_roc.set_title('ROC for Crohn vd. HC classification', fontsize=18)
    axe_svm_roc.legend(loc="lower right", prop={'size': 8})
    fig_svm_roc.savefig(os.path.join(svmShufflePath, 'shuffle_ROC.png'))

    # 绘制特征之间的相关性并存储
    corr_mat = corr(X_new, featureName, crohnPath)
    corr_mat.corr_heatmap()
'''


# DTI单模态
'''
if __name__ == '__main__':
    # 设定结果存储路径并创建文件夹
    timeDir = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    crohnPath = os.path.join('crohn', 'dti', timeDir)
    os.makedirs(crohnPath, exist_ok=True)

    # 读取数据
    dataFA = pd.read_csv('./mkl_feature_1007/feature_Native_space/FA_90.csv', sep=',', engine='python')
    dataMD = pd.read_csv('./mkl_feature_1007/feature_Native_space/MD_90.csv', sep=',', engine='python')
    dataRD = pd.read_csv('./mkl_feature_1007/feature_Native_space/RD_90.csv', sep=',', engine='python')
    dataAD = pd.read_csv('./mkl_feature_1007/feature_Native_space/L1_90.csv', sep=',', engine='python')
    # dataL2 = pd.read_csv('./mkl_feature_1007/feature_Native_space/FA_90.csv', sep=',', engine='python')
    # dataL3 = pd.read_csv('./mkl_feature_1007/feature_Native_space/FA_90.csv', sep=',', engine='python')

    dataFaMatrix = pd.read_csv('./mkl_feature_1007/feature_Deterministi/FA_Matrix_90.csv', sep=',', engine='python')
    dataFN = pd.read_csv('./mkl_feature_1007/feature_Deterministi/FN_Matrix_90.csv', sep=',', engine='python')
    dataLength = pd.read_csv('./mkl_feature_1007/feature_Deterministi/Length_Matrix_Mapped_90.csv', sep=',',
                             engine='python')

    # 读取label
    label = pd.read_csv('./mkl_feature_1007/label.csv', sep=',', engine='python', header=None)

    # 数据统一转换为numpy格式
    faNumpy = np.array(dataFA.iloc[:, :])
    mdNumpy = np.array(dataMD.iloc[:, :])
    rdNumpy = np.array(dataRD.iloc[:, :])
    adNumpy = np.array(dataAD.iloc[:, :])
    # l2Numpy = np.array(dataL2.iloc[:, :])
    # l3Numpy = np.array(dataL3.iloc[:, :])
    faMatrixNumpy = np.array(dataFaMatrix.iloc[:, :])
    fnNumpy = np.array(dataFN.iloc[:, :])
    lengthNumpy = np.array(dataLength.iloc[:, :])
    # 标签转换为numpy格式
    y = np.array(label.iloc[:, :]).ravel()
    # 规定病人为0，健康人为1
    labelName = ['0', '1']
    posLabel = 1

    # 存储特征名称
    faName = dataFA.columns._values
    mdName = dataMD.columns._values
    rdName = dataRD.columns._values
    adName = dataAD.columns._values
    # l2Name = dataL2.columns._values
    # l3Name = dataL3.columns._values
    faMatrixName = dataFaMatrix.columns._values
    fnName = dataFN.columns._values
    lengthName = dataLength.columns._values

    # 单独对FA Matrix, FN, Length使用留一法U检验进行特征选择
    faMatUPath = os.path.join(crohnPath, 'fa_matrix_utest')
    os.makedirs(faMatUPath, exist_ok=True)
    utestFAMatrix = Utest(faMatrixNumpy, faMatrixName,
                          dis_num=63, hc_num=39, thres=0.1, all_value=False, stat_path=faMatUPath)
    selectFaMat, selectFaMatName, _, _, faMatUIndex = utestFAMatrix.utest_loo_freq(K_value=40, choice='percent')

    fnUPath = os.path.join(crohnPath, 'fn_utest')
    os.makedirs(fnUPath, exist_ok=True)
    utestFN = Utest(fnNumpy, fnName,
                    dis_num=63, hc_num=39, thres=0.1, all_value=False, stat_path=fnUPath)
    selectFN, selectFNName, _, _, fnUIndex = utestFN.utest_loo_freq(K_value=40, choice='percent')

    lengthUPath = os.path.join(crohnPath, 'length_utest')
    os.makedirs(lengthUPath, exist_ok=True)
    utestLength = Utest(lengthNumpy, lengthName,
                        dis_num=63, hc_num=39, thres=0.1, all_value=False, stat_path=lengthUPath)
    selectLength, selectLengthName, _, _, lengthUIndex = utestLength.utest_loo_freq(K_value=40, choice='percent')

    # 对数据进行标准化并拼接

    # 对结构进行拼接
    sMRI = np.concatenate((selectFaMat, selectFN, selectLength, faNumpy, mdNumpy, rdNumpy, adNumpy), axis=1)
    sMRIName = np.concatenate((selectFaMatName, selectFNName, selectLengthName, faName, mdName, rdName, adName), axis=0)

    # 对结构数据进行z标准化
    nor_sMRI = normalization(sMRI)
    sMRI = nor_sMRI.z_score()
    # 结构数据的特征数量
    sMRI_num = len(sMRIName)

    # X为最终的输入
    X = sMRI

    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

    # lasso = lasso(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
    #               feature_name=sMRIName, path=crohnPath, cv_val=True)
    #
    # X_train_new, X_test_new, \
    # featureName, feature_freq, feature_coef, index = lasso.lasso_shuffle(shuffle_time=100,
    #                                                                      alpha_range=np.arange(0.01, 0.1, 0.01))

    # 使用elastic net进行特征选择
    elas = elastic_net(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                       feature_name=sMRIName, path=crohnPath, cv_val=False)

    X_train_new, X_test_new, featureName, _, _, index = elas.elasticnet_shuffle(shuffle_time=100,
                                                                                l1_range=1.0,
                                                                                alphas_range=0.14)

    # index = [106, 21, 22, 23, 419, 597, 614, 654, 266]
    featureName = sMRIName[index]

    # 不进行模态区分，为了对比试验和特征相关性矩阵
    X_new = X[:, index]

    # 使用svm_shuffle返回结果并且绘制ROC曲线的示意图
    svmShufflePath = os.path.join(crohnPath, 'svm_shuffle')
    os.makedirs(svmShufflePath, exist_ok=True)
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_new, y, path=crohnPath)
    train_means, train_std, test_means, test_std, roc_dict = svm.svm_shuffle(outer=10, shuffle_time=svmTime,
                                                                             para={'kernel': 'linear',
                                                                                   'gamma': 50,
                                                                                   'C': 60},
                                                                             svm_metrics=['accuracy',
                                                                                          'sensitivity',
                                                                                          'specificity'])

    # 绘制accuracy随着shuffle次数变化的曲线
    x_axis = np.arange(1, svmTime+1)
    fig_svm, axe_svm = plt.subplots()
    axe_svm.fill_between(x_axis,
                         np.array(test_means['accuracy']) - np.array(test_std['accuracy']),
                         np.array(test_means['accuracy']) + np.array(test_std['accuracy']),
                         alpha=0.2)
    axe_svm.plot(x_axis, np.array(test_means['accuracy']), '--', color='g', alpha=1,
                 label='Mean accuracy: % 0.3f' % np.mean(test_means['accuracy']))
    axe_svm.set_ylim([-0.01, 1.01])
    axe_svm.set_xlabel('Shuffle Time')
    axe_svm.set_ylabel('Accuracy')
    axe_svm.set_title('Accuracy for Crohn vd. HC classification')
    axe_svm.legend(loc="lower right", prop={'size': 8})
    fig_svm.savefig(os.path.join(svmShufflePath, 'shuffle_result.png'))

    # 绘制ROC曲线
    std_auc_train = np.std(roc_dict['auc_train'])
    std_auc_test = np.std(roc_dict['auc_test'])

    fig_svm_roc, axe_svm_roc = plt.subplots()
    axe_svm_roc.plot(roc_dict['fpr_train'], roc_dict['tpr_train'],
                     label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (roc_dict['auc_train'], std_auc_train),
                     lw=2, alpha=.8)
    axe_svm_roc.plot(roc_dict['fpr_test'], roc_dict['tpr_test'],
                     label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (roc_dict['auc_test'], std_auc_test),
                     lw=2, alpha=.8)

    std_tpr_train = np.std(roc_dict['tpr_list_train'], axis=0)
    std_tpr_test = np.std(roc_dict['tpr_list_test'], axis=0)

    tprs_upper = np.minimum(roc_dict['tpr_test'] + std_tpr_test, 1)
    tprs_lower = np.maximum(roc_dict['tpr_test'] - std_tpr_test, 0)

    axe_svm_roc.fill_between(roc_dict['fpr_test'], tprs_lower, tprs_upper, color='grey', alpha=.2,
                             label=r'$\pm$ 1 std. dev.')

    tprs_upper = np.minimum(roc_dict['tpr_train'] + std_tpr_train, 1)
    tprs_lower = np.maximum(roc_dict['tpr_train'] - std_tpr_train, 0)

    axe_svm_roc.fill_between(roc_dict['fpr_train'], tprs_lower, tprs_upper, color='grey', alpha=.2,
                             label=r'$\pm$ 1 std. dev.')

    axe_svm_roc.grid(True)
    axe_svm_roc.plot([0, 1], [0, 1], 'r--')
    axe_svm_roc.set_xlim([-0.01, 1.01])
    axe_svm_roc.set_ylim([-0.01, 1.01])
    axe_svm_roc.set_xlabel('False Positive Rate', fontsize=15)
    axe_svm_roc.set_ylabel('True Positive Rate', fontsize=15)
    axe_svm_roc.set_title('ROC for Crohn vd. HC classification', fontsize=18)
    axe_svm_roc.legend(loc="lower right", prop={'size': 8})
    fig_svm_roc.savefig(os.path.join(svmShufflePath, 'shuffle_ROC.png'))

    corr_mat = corr(X_new, featureName, crohnPath)
    corr_mat.corr_heatmap()
'''


# 每个特征单独进行分类
'''
if __name__ == '__main__':
    # 设定结果存储路径并创建文件夹
    timeDir = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    crohnPath = os.path.join('crohn', 'reho', timeDir)
    os.makedirs(crohnPath, exist_ok=True)

    # 读取数据

    ###############
    #####DTI数据####
    ###############
    # 读取数据
    dataFA = pd.read_csv('./mkl_feature_1007/feature_Native_space/FA_90.csv', sep=',', engine='python')
    dataMD = pd.read_csv('./mkl_feature_1007/feature_Native_space/MD_90.csv', sep=',', engine='python')
    dataRD = pd.read_csv('./mkl_feature_1007/feature_Native_space/RD_90.csv', sep=',', engine='python')
    dataAD = pd.read_csv('./mkl_feature_1007/feature_Native_space/L1_90.csv', sep=',', engine='python')
    # dataL2 = pd.read_csv('./mkl_feature_1007/feature_Native_space/FA_90.csv', sep=',', engine='python')
    # dataL3 = pd.read_csv('./mkl_feature_1007/feature_Native_space/FA_90.csv', sep=',', engine='python')

    dataFaMatrix = pd.read_csv('./mkl_feature_1007/feature_Deterministi/FA_Matrix_90.csv', sep=',', engine='python')
    dataFN = pd.read_csv('./mkl_feature_1007/feature_Deterministi/FN_Matrix_90.csv', sep=',', engine='python')
    dataLength = pd.read_csv('./mkl_feature_1007/feature_Deterministi/Length_Matrix_Mapped_90.csv', sep=',',
                             engine='python')

    # 读取label
    label = pd.read_csv('./mkl_feature_1007/label.csv', sep=',', engine='python', header=None)

    # 数据统一转换为numpy格式
    faNumpy = np.array(dataFA.iloc[:, :])
    mdNumpy = np.array(dataMD.iloc[:, :])
    rdNumpy = np.array(dataRD.iloc[:, :])
    adNumpy = np.array(dataAD.iloc[:, :])
    # l2Numpy = np.array(dataL2.iloc[:, :])
    # l3Numpy = np.array(dataL3.iloc[:, :])
    faMatrixNumpy = np.array(dataFaMatrix.iloc[:, :])
    fnNumpy = np.array(dataFN.iloc[:, :])
    lengthNumpy = np.array(dataLength.iloc[:, :])
    # 标签转换为numpy格式
    y = np.array(label.iloc[:, :]).ravel()
    # 规定病人为0，健康人为1
    labelName = ['0', '1']
    posLabel = 1

    # 存储特征名称
    faName = dataFA.columns._values
    mdName = dataMD.columns._values
    rdName = dataRD.columns._values
    adName = dataAD.columns._values
    # l2Name = dataL2.columns._values
    # l3Name = dataL3.columns._values
    faMatrixName = dataFaMatrix.columns._values
    fnName = dataFN.columns._values
    lengthName = dataLength.columns._values

    # 单独对FA Matrix, FN, Length使用留一法U检验进行特征选择
    faMatUPath = os.path.join(crohnPath, 'fa_matrix_utest')
    os.makedirs(faMatUPath, exist_ok=True)
    utestFAMatrix = Utest(faMatrixNumpy, faMatrixName,
                          dis_num=63, hc_num=39, thres=0.1, all_value=False, stat_path=faMatUPath)
    selectFaMat, selectFaMatName, _, _, faMatUIndex = utestFAMatrix.utest_loo_freq(K_value=40, choice='percent')

    fnUPath = os.path.join(crohnPath, 'fn_utest')
    os.makedirs(fnUPath, exist_ok=True)
    utestFN = Utest(fnNumpy, fnName,
                    dis_num=63, hc_num=39, thres=0.1, all_value=False, stat_path=fnUPath)
    selectFN, selectFNName, _, _, fnUIndex = utestFN.utest_loo_freq(K_value=40, choice='percent')

    lengthUPath = os.path.join(crohnPath, 'length_utest')
    os.makedirs(lengthUPath, exist_ok=True)
    utestLength = Utest(lengthNumpy, lengthName,
                        dis_num=63, hc_num=39, thres=0.1, all_value=False, stat_path=lengthUPath)
    selectLength, selectLengthName, _, _, lengthUIndex = utestLength.utest_loo_freq(K_value=40, choice='percent')

    ###############
    ####功能数据####
    ###############

    # 读取数据
    dataFalff = pd.read_csv('./mkl_feature_1007/feature_fMRI/mALFF_classic_global.csv', sep=',', engine='python')
    dataFc = pd.read_csv('./mkl_feature_1007/feature_fMRI/FC_feature_90_R_G.csv', sep=',', engine='python')
    dataReho = pd.read_csv('./mkl_feature_1007/feature_fMRI/smReHo_G_90.csv', sep=',', engine='python')

    # 数据统一转换为numpy格式
    falffNumpy = np.array(dataFalff.iloc[:, :])
    fcNumpy = np.array(dataFc.iloc[:, :])
    rehoNumpy = np.array(dataReho.iloc[:, :])

    # 存储特征名称
    falffName = dataFalff.columns._values
    fcName = dataFc.columns._values
    rehoName = dataReho.columns._values

    # 单独对FC使用留一法U检验进行特征选择
    fcUPath = os.path.join(crohnPath, 'fc_utest')
    os.makedirs(fcUPath, exist_ok=True)
    utestFC = Utest(fcNumpy, fcName,
                    dis_num=63, hc_num=39, thres=0.01, all_value=False, stat_path=fcUPath)
    selectFC, selectFCName, _, _, fcUIndex = utestFC.utest_loo_freq(K_value=40, choice='percent')

    # X为最终的输入
    X = rehoNumpy
    MRIName = rehoName
    MRIName = np.array(MRIName)

    nor_X = normalization(X)
    X = nor_X.z_score()

    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

    # lasso = lasso(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
    #               feature_name=sMRIName, path=crohnPath, cv_val=True)
    #
    # X_train_new, X_test_new, \
    # featureName, feature_freq, feature_coef, index = lasso.lasso_shuffle(shuffle_time=100,
    #                                                                      alpha_range=np.arange(0.01, 0.1, 0.01))

    # 使用elastic net进行特征选择
    elas = elastic_net(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                       feature_name=MRIName, path=crohnPath, cv_val=False)

    X_train_new, X_test_new, featureName, _, _, index = elas.elasticnet_shuffle(shuffle_time=100,
                                                                                l1_range=0.5,
                                                                                alphas_range=0.1)

    # index = [106, 21, 22, 23, 419, 597, 614, 654, 266]
    featureName = MRIName[index]

    # 不进行模态区分，为了对比试验和特征相关性矩阵
    X_new = X[:, index]

    # 使用svm_shuffle返回结果并且绘制ROC曲线的示意图
    svmShufflePath = os.path.join(crohnPath, 'svm_shuffle')
    os.makedirs(svmShufflePath, exist_ok=True)
    # 设置参数
    para_rbf = {'kernel': ['rbf'],
                'gamma': np.arange(0.1, 1, 0.1),
                'C': np.arange(1, 100, 1)}
    para_linear = {'kernel': ['linear'],
                   'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_new, y, path=crohnPath)
    train_means, train_std, test_means, test_std, roc_dict = svm.svm_shuffle(outer=10, shuffle_time=svmTime,
                                                                             para={'kernel': 'linear',
                                                                                   'gamma': 15,
                                                                                   'C': 60},
                                                                             svm_metrics=['accuracy',
                                                                                          'sensitivity',
                                                                                          'specificity'])

    # 绘制accuracy随着shuffle次数变化的曲线
    x_axis = np.arange(1, svmTime+1)
    fig_svm, axe_svm = plt.subplots()
    axe_svm.fill_between(x_axis,
                         np.array(test_means['accuracy']) - np.array(test_std['accuracy']),
                         np.array(test_means['accuracy']) + np.array(test_std['accuracy']),
                         alpha=0.2)
    axe_svm.plot(x_axis, np.array(test_means['accuracy']), '--', color='g', alpha=1,
                 label='Mean accuracy: % 0.3f' % np.mean(test_means['accuracy']))
    axe_svm.set_ylim([-0.01, 1.01])
    axe_svm.set_xlabel('Shuffle Time')
    axe_svm.set_ylabel('Accuracy')
    axe_svm.set_title('Accuracy for Crohn vs. HC classification')
    axe_svm.legend(loc="lower right", prop={'size': 8})
    fig_svm.savefig(os.path.join(svmShufflePath, 'shuffle_result.png'))

    # 绘制ROC曲线
    std_auc_train = np.std(roc_dict['auc_train'])
    std_auc_test = np.std(roc_dict['auc_test'])

    fig_svm_roc, axe_svm_roc = plt.subplots()
    axe_svm_roc.plot(roc_dict['fpr_train'], roc_dict['tpr_train'],
                     label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (roc_dict['auc_train'], std_auc_train),
                     lw=2, alpha=.8)
    axe_svm_roc.plot(roc_dict['fpr_test'], roc_dict['tpr_test'],
                     label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (roc_dict['auc_test'], std_auc_test),
                     lw=2, alpha=.8)

    std_tpr_train = np.std(roc_dict['tpr_list_train'], axis=0)
    std_tpr_test = np.std(roc_dict['tpr_list_test'], axis=0)

    tprs_upper = np.minimum(roc_dict['tpr_test'] + std_tpr_test, 1)
    tprs_lower = np.maximum(roc_dict['tpr_test'] - std_tpr_test, 0)

    axe_svm_roc.fill_between(roc_dict['fpr_test'], tprs_lower, tprs_upper, color='grey', alpha=.2,
                             label=r'$\pm$ 1 std. dev.')

    tprs_upper = np.minimum(roc_dict['tpr_train'] + std_tpr_train, 1)
    tprs_lower = np.maximum(roc_dict['tpr_train'] - std_tpr_train, 0)

    axe_svm_roc.fill_between(roc_dict['fpr_train'], tprs_lower, tprs_upper, color='grey', alpha=.2,
                             label=r'$\pm$ 1 std. dev.')

    axe_svm_roc.grid(True)
    axe_svm_roc.plot([0, 1], [0, 1], 'r--')
    axe_svm_roc.set_xlim([-0.01, 1.01])
    axe_svm_roc.set_ylim([-0.01, 1.01])
    axe_svm_roc.set_xlabel('False Positive Rate', fontsize=15)
    axe_svm_roc.set_ylabel('True Positive Rate', fontsize=15)
    axe_svm_roc.set_title('ROC for Crohn vs. HC classification', fontsize=18)
    axe_svm_roc.legend(loc="lower right", prop={'size': 8})
    fig_svm_roc.savefig(os.path.join(svmShufflePath, 'shuffle_ROC.png'))
'''

# 所有模型集成（单measurements重新用elastic net选特征）
'''
if __name__ == '__main__':
    timeDir = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    crohnPath = os.path.join('crohn', 'all_results', timeDir)
    os.makedirs(crohnPath, exist_ok=True)

    # 读取数据
    dataFalff = pd.read_csv('./mkl_feature_1007/feature_fMRI/mALFF_classic_global.csv', sep=',', engine='python')
    dataFc = pd.read_csv('./mkl_feature_1007/feature_fMRI/FC_feature_90_R_G.csv', sep=',', engine='python')
    dataReho = pd.read_csv('./mkl_feature_1007/feature_fMRI/smReHo_G_90.csv', sep=',', engine='python')

    dataFA = pd.read_csv('./mkl_feature_1007/feature_Native_space/FA_90.csv', sep=',', engine='python')
    dataMD = pd.read_csv('./mkl_feature_1007/feature_Native_space/MD_90.csv', sep=',', engine='python')
    dataRD = pd.read_csv('./mkl_feature_1007/feature_Native_space/RD_90.csv', sep=',', engine='python')
    dataAD = pd.read_csv('./mkl_feature_1007/feature_Native_space/L1_90.csv', sep=',', engine='python')
    # dataL2 = pd.read_csv('./mkl_feature_1007/feature_Native_space/FA_90.csv', sep=',', engine='python')
    # dataL3 = pd.read_csv('./mkl_feature_1007/feature_Native_space/FA_90.csv', sep=',', engine='python')

    dataFaMatrix = pd.read_csv('./mkl_feature_1007/feature_Deterministi/FA_Matrix_90.csv', sep=',', engine='python')
    dataFN = pd.read_csv('./mkl_feature_1007/feature_Deterministi/FN_Matrix_90.csv', sep=',', engine='python')
    dataLength = pd.read_csv('./mkl_feature_1007/feature_Deterministi/Length_Matrix_Mapped_90.csv', sep=',', engine='python')

    dataVbm = pd.read_csv('./mkl_feature_1007/vbm/VBM_90.csv', sep=',', engine='python')

    # 读取label
    label = pd.read_csv('./mkl_feature_1007/label.csv', sep=',', engine='python', header=None)

    # 数据统一转换为numpy格式
    falffNumpy = np.array(dataFalff.iloc[:, :])
    fcNumpy = np.array(dataFc.iloc[:, :])
    rehoNumpy = np.array(dataReho.iloc[:, :])
    faNumpy = np.array(dataFA.iloc[:, :])
    mdNumpy = np.array(dataMD.iloc[:, :])
    rdNumpy = np.array(dataRD.iloc[:, :])
    adNumpy = np.array(dataAD.iloc[:, :])
    # l2Numpy = np.array(dataL2.iloc[:, :])
    # l3Numpy = np.array(dataL3.iloc[:, :])
    faMatrixNumpy = np.array(dataFaMatrix.iloc[:, :])
    fnNumpy = np.array(dataFN.iloc[:, :])
    lengthNumpy = np.array(dataLength.iloc[:, :])
    vbmNumpy = np.array(dataVbm.iloc[:, :])
    # 标签转换为numpy格式
    y = np.array(label.iloc[:, :]).ravel()
    # 规定病人为0，健康人为1
    labelName = ['0', '1']
    posLabel = 1

    # 存储特征名称
    falffName = dataFalff.columns._values
    fcName = dataFc.columns._values
    rehoName = dataReho.columns._values
    faName = dataFA.columns._values
    mdName = dataMD.columns._values
    rdName = dataRD.columns._values
    adName = dataAD.columns._values
    # l2Name = dataL2.columns._values
    # l3Name = dataL3.columns._values
    faMatrixName = dataFaMatrix.columns._values
    fnName = dataFN.columns._values
    lengthName = dataLength.columns._values
    vbmName = dataVbm.columns._values

    # 单独对FC, FA Matrix, FN, Length使用留一法U检验进行特征选择
    fcUPath = os.path.join(crohnPath, 'fc_utest')
    os.makedirs(fcUPath, exist_ok=True)
    utestFC = Utest(fcNumpy, fcName,
                    dis_num=63, hc_num=39, thres=0.01, all_value=False, stat_path=fcUPath)
    selectFC, selectFCName, _, _, fcUIndex = utestFC.utest_loo_freq(K_value=40, choice='percent')

    faMatUPath = os.path.join(crohnPath, 'fa_matrix_utest')
    os.makedirs(faMatUPath, exist_ok=True)
    utestFAMatrix = Utest(faMatrixNumpy, faMatrixName,
                          dis_num=63, hc_num=39, thres=0.1, all_value=False, stat_path=faMatUPath)
    selectFaMat, selectFaMatName, _, _, faMatUIndex = utestFAMatrix.utest_loo_freq(K_value=40, choice='percent')

    fnUPath = os.path.join(crohnPath, 'fn_utest')
    os.makedirs(fnUPath, exist_ok=True)
    utestFN = Utest(fnNumpy, fnName,
                    dis_num=63, hc_num=39, thres=0.1, all_value=False, stat_path=fnUPath)
    selectFN, selectFNName, _, _, fnUIndex = utestFN.utest_loo_freq(K_value=40, choice='percent')

    lengthUPath = os.path.join(crohnPath, 'length_utest')
    os.makedirs(lengthUPath, exist_ok=True)
    utestLength = Utest(lengthNumpy, lengthName,
                        dis_num=63, hc_num=39, thres=0.1, all_value=False, stat_path=lengthUPath)
    selectLength, selectLengthName, _, _, lengthUIndex = utestLength.utest_loo_freq(K_value=40, choice='percent')

    # 对数据进行标准化并拼接
    # 拼接功能数据
    fMRI = np.concatenate((falffNumpy, selectFC, rehoNumpy), axis=1)
    fMRIName = np.concatenate((falffName, selectFCName, rehoName), axis=0)
    # fMRIName = np.concatenate((falffName, fcName, rehoName), axis=0)
    # 对功能数据进行z标准化
    nor_fMRI = normalization(fMRI)
    fMRI = nor_fMRI.z_score()
    # 功能数据的特征数量
    fMRI_num = len(fMRIName)

    # 对DTI进行拼接
    DTI = np.concatenate((selectFaMat, selectFN, selectLength, faNumpy, mdNumpy, rdNumpy, adNumpy), axis=1)
    DTIName = np.concatenate((selectFaMatName, selectFNName, selectLengthName, faName, mdName, rdName, adName), axis=0)

    # 对DTI数据进行z标准化
    nor_DTI = normalization(DTI)
    DTI = nor_DTI.z_score()
    # DTI数据的特征数量
    DTI_num = len(DTIName)

    # 对结构数据进行拼接
    sMRI = vbmNumpy
    sMRIName = vbmName
    # 对结构数据进行z标准化
    nor_sMRI = normalization(sMRI)
    sMRI = nor_sMRI.z_score()
    # 结构数据的特征数量
    sMRI_num = len(sMRIName)

    # 直接拼接多模态数据
    multimodal = np.concatenate((fMRI, DTI, sMRI), axis=1)
    multimodalName = np.concatenate((fMRIName, DTIName, sMRIName), axis=0)
    # X为最终的输入
    X = multimodal

    # ii = ['fc_1094', 'fc_1119', 'fc_1442', 'fc_1444', 'fc_145',
    #       'fc_1504', 'fc_1670', 'fc_1796', 'fc_1827', 'fc_1929', 'fc_1950']
    # jj = []
    # for i in ii:
    #     name1 = list(multimodalName).index(i)
    #     jj.append(name1)

    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

    # 多模态的索引
    indexMulti = [620, 625, 0, 210, 215, 217, 221, 223, 229, 239, 240, 244, 86, 90, 91, 93, 96, 97, 98, 99,
                  103, 104, 107, 115, 117, 120, 121, 251, 256, 260, 269, 271, 290, 291, 296, 297, 300, 321,
                  323, 480, 489, 564, 565, 142, 181, 200, 730, 735]
    featureName = multimodalName[indexMulti]

    # 根据fmri DTI sMRI的特征数量，将他们重新分成两个矩阵，以便进行mkl
    index_mod = index_split(indexMulti)
    index_fMRI, index_DTI, index_sMRI = index_mod.three_split(fMRI_num, DTI_num, sMRI_num)
    X_fMRI = X[:, index_fMRI]
    X_DTI = X[:, index_DTI]
    X_sMRI = X[:, index_sMRI]
    # 不进行模态区分，为了对比试验和特征相关性矩阵
    X_new = X[:, indexMulti]

    fig_roc, axe_roc = plt.subplots()
    fig_roc.set_size_inches(15, 10)

    # pipeline1: 使用mkl返回结果

    mkl_kernel = {'kernel_type1': 'rbf',
                  'kernel_type2': 'rbf',
                  'kernel_type3': 'rbf',
                  'kernel_weight1': 0.2,
                  'kernel_weight2': 0.5,
                  'kernel_weight3': 0.3}
    mkl_para = {'kernel1': 15,
                'kernel2': 50,
                'kernel3': 10,
                'C': 80}
    mkl = mkl_svm(X_fMRI, X_DTI, X_sMRI, y, mkl_path=crohnPath)
    train_means, train_std, mklMeans, mklStd, mklROC = mkl.mksvm3(kernel_dict=mkl_kernel,
                                                                        para=mkl_para,
                                                                        svm_metrics=['accuracy',
                                                                                     'sensitivity',
                                                                                     'specificity'])
    # 绘制ROC曲线
    mklAUCStd = np.std(mklROC['auc_test'])
    axe_roc.plot(mklROC['fpr_test'], mklROC['tpr_test'],
                 label=r'Multi-kernel AUC = %0.2f' % mklROC['auc_test'],
                 lw=2, alpha=.8)

    # pipeline2: 使用svm_shuffle返回结果并且绘制ROC曲线的示意图

    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 10, 1)}
    svmTime = 100
    svm = SVM(X_new, y, path=crohnPath)
    train_means, train_std, concatMeans, concatStd, concatROC = svm.svm_shuffle(outer=5, shuffle_time=svmTime,
                                                                                para={'kernel': 'linear',
                                                                                      'gamma': 0.08,
                                                                                      'C': 1},
                                                                                svm_metrics=['accuracy',
                                                                                             'sensitivity',
                                                                                             'specificity'])

    # 绘制ROC曲线
    axe_roc.plot(concatROC['fpr_test'], concatROC['tpr_test'],
                 label=r'Concatenate AUC = %0.2f' % concatROC['auc_test'],
                 lw=2, alpha=.8)

    # pipeline3: fMRI独立结果
    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(fMRI, y, test_size=0.25, stratify=y)
    # 使用elastic net进行特征选择
    elas = elastic_net(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                       feature_name=multimodalName, path=crohnPath, cv_val=False)

    X_train_new, X_test_new, _, _, _, indexfMRI = elas.elasticnet_shuffle(shuffle_time=100,
                                                                                l1_range=1.0,
                                                                                alphas_range=0.05)
    indexfMRI = [0, 52, 77, 82, 86, 93, 94, 96, 97, 98, 103, 107, 110, 115, 117, 119]
    X_fMRI_only = fMRI[:, indexfMRI]
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_fMRI_only, y, path=crohnPath)
    train_means, train_std, fmriMeans, fmriStd, fmriROC = svm.svm_shuffle(outer=5, shuffle_time=svmTime,
                                                                             para={'kernel': 'linear',
                                                                                   'gamma': 15,
                                                                                   'C': 60},
                                                                             svm_metrics=['accuracy',
                                                                                          'sensitivity',
                                                                                          'specificity'])

    # 绘制ROC曲线

    axe_roc.plot(fmriROC['fpr_test'], fmriROC['tpr_test'],
                     label=r'rs-fMRI only AUC = %0.2f' % fmriROC['auc_test'],
                     lw=2, alpha=.8)

    # pipeline4； DTI独立结果
    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(DTI, y, test_size=0.25, stratify=y)
    # 使用elastic net进行特征选择
    elas = elastic_net(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                       feature_name=multimodalName, path=crohnPath, cv_val=False)

    # X_train_new, X_test_new, _, _, _, indexsMRI = elas.elasticnet_shuffle(shuffle_time=100,
    #                                                                       l1_range=1.0,
    #                                                                       alphas_range=0.1)
    indexDTI = [412, 414, 417, 448, 121, 137, 148, 189, 1, 5, 13, 14, 31, 37, 41, 43, 79,
                 51, 54, 55, 59, 74, 75, 76, 77, 64, 65, 81, 82, 104, 113, 115]
    X_DTI_only = DTI[:, indexDTI]
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_DTI_only, y, path=crohnPath)
    train_means, train_std, DTIMeans, DTIStd, DTIROC = svm.svm_shuffle(outer=5, shuffle_time=svmTime,
                                                                          para={'kernel': 'linear',
                                                                                'gamma': 15,
                                                                                'C': 100},
                                                                          svm_metrics=['accuracy',
                                                                                       'sensitivity',
                                                                                       'specificity'])
    axe_roc.plot(DTIROC['fpr_test'], DTIROC['tpr_test'],
                 label=r'DTI only AUC = %0.2f' % DTIROC['auc_test'],
                 lw=2, alpha=.8)

    # pipeline5； sMRI独立结果
    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(sMRI, y, test_size=0.25, stratify=y)
    # 使用elastic net进行特征选择
    elas = elastic_net(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                       feature_name=multimodalName, path=crohnPath, cv_val=False)

    X_train_new, X_test_new, _, _, _, indexsMRI = elas.elasticnet_shuffle(shuffle_time=100,
                                                                          l1_range=1.0,
                                                                          alphas_range=0.1)
    # indexsMRI = []
    X_sMRI_only = sMRI[:, indexsMRI]
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_sMRI_only, y, path=crohnPath)
    train_means, train_std, smriMeans, smriStd, smriROC = svm.svm_shuffle(outer=5, shuffle_time=svmTime,
                                                                          para={'kernel': 'linear',
                                                                                'gamma': 15,
                                                                                'C': 100},
                                                                          svm_metrics=['accuracy',
                                                                                       'sensitivity',
                                                                                       'specificity'])
    axe_roc.plot(smriROC['fpr_test'], smriROC['tpr_test'],
                 label=r'sMRI only AUC = %0.2f' % smriROC['auc_test'],
                 lw=2, alpha=.8)

    # pipeline5: falff only
    X_train, X_test, y_train, y_test = train_test_split(falffNumpy, y, test_size=0.25, stratify=y)
    # 使用elastic net进行特征选择
    elas = elastic_net(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                       feature_name=multimodalName, path=crohnPath, cv_val=False)

    X_train_new, X_test_new, _, _, _, indexFalff = elas.elasticnet_shuffle(shuffle_time=100,
                                                                          l1_range=1.0,
                                                                          alphas_range=0.001)
    # indexFalff = [17, 29, 52, 55, 77, 82, 83]
    X_falff_only = falffNumpy[:, indexFalff]
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_falff_only, y, path=crohnPath)
    train_means, train_std, falffMeans, falffStd, falffROC = svm.svm_shuffle(outer=5, shuffle_time=svmTime,
                                                                          para={'kernel': 'linear',
                                                                                'gamma': 15,
                                                                                'C': 60},
                                                                          svm_metrics=['accuracy',
                                                                                       'sensitivity',
                                                                                       'specificity'])
    axe_roc.plot(falffROC['fpr_test'], falffROC['tpr_test'],
                 label=r'fALFF only AUC = %0.2f' % falffROC['auc_test'],
                 lw=2, alpha=.8)

    # pipeline6: fc only
    X_train, X_test, y_train, y_test = train_test_split(fcNumpy, y, test_size=0.25, stratify=y)
    # 使用elastic net进行特征选择
    elas = elastic_net(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                       feature_name=multimodalName, path=crohnPath, cv_val=False)

    # X_train_new, X_test_new, _, _, _, indexFc = elas.elasticnet_shuffle(shuffle_time=100,
    #                                                                        l1_range=1.0,
    #                                                                        alphas_range=0.004)
    # indexFc = [1047, 1073, 1093, 1188, 1303, 1342, 1352, 1447, 1457, 1535, 196, 1769, 1870, 1905,
    #            1946, 2116, 2132, 2473, 2553, 2701, 2702, 2813, 2823, 2912, 2966, 3085, 3216, 3236,
    #            3246, 3254, 377, 3311, 384, 39, 3387, 524, 61, 858, 887, 899]
    # indexFc = [1047, 1073, 1093, 1188, 1303, 1342, 1352, 1447, 1457, 1535, 196, 1769, 1870, 1905,
    #            1946, 2116, 2132, 2473, 2553, 2701, 2702, 2813, 2823, 2912, 2966, 3085, 3216, 3236,
    #            3246, 3254, 377, 3311, 384, 39, 3387]
    indexFc = [86, 90, 91, 93, 96, 97, 98, 99, 103, 104, 107, 115, 117, 120, 121]
    X_fc_only = multimodal[:, indexFc]
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_fc_only, y, path=crohnPath)
    train_means, train_std, fcMeans, fcStd, fcROC = svm.svm_shuffle(outer=5, shuffle_time=svmTime,
                                                                    para={'kernel': 'linear',
                                                                                   'gamma': 15,
                                                                                   'C': 1},
                                                                    svm_metrics=['accuracy',
                                                                                'sensitivity',
                                                                                'specificity'])
    axe_roc.plot(fcROC['fpr_test'], fcROC['tpr_test'],
                 label=r'FC only AUC = %0.2f' % fcROC['auc_test'],
                 lw=2, alpha=.8)

    # pipeline7: reho only
    X_train, X_test, y_train, y_test = train_test_split(rehoNumpy, y, test_size=0.25, stratify=y)
    # 使用elastic net进行特征选择
    elas = elastic_net(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                       feature_name=multimodalName, path=crohnPath, cv_val=False)

    X_train_new, X_test_new, _, _, _, indexReho = elas.elasticnet_shuffle(shuffle_time=100,
                                                                        l1_range=1.0,
                                                                        alphas_range=0.004)
    # indexReho = [10, 19, 23, 48, 49, 57, 58, 65, 69, 72, 76, 77, 79, 81, 82, 83]
    X_reho_only = rehoNumpy[:, indexReho]
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_reho_only, y, path=crohnPath)
    train_means, train_std, rehoMeans, rehoStd, rehoROC = svm.svm_shuffle(outer=5, shuffle_time=svmTime,
                                                                             para={'kernel': 'linear',
                                                                                   'gamma': 15,
                                                                                   'C': 60},
                                                                             svm_metrics=['accuracy',
                                                                                          'sensitivity',
                                                                                          'specificity'])
    axe_roc.plot(rehoROC['fpr_test'], rehoROC['tpr_test'],
                 label=r'ReHo only AUC = %0.2f' % rehoROC['auc_test'],
                 lw=2, alpha=.8)

    # pipeline8: fa only
    X_train, X_test, y_train, y_test = train_test_split(faNumpy, y, test_size=0.25, stratify=y)
    # 使用elastic net进行特征选择
    elas = elastic_net(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                       feature_name=multimodalName, path=crohnPath, cv_val=False)

    X_train_new, X_test_new, _, _, _, indexFa = elas.elasticnet_shuffle(shuffle_time=100,
                                                                          l1_range=1.0,
                                                                          alphas_range=0.0007)
    # indexFa = [10, 12, 23, 24, 28, 32, 33, 41, 44, 46, 6, 56, 58, 64, 69, 77]
    X_fa_only = faNumpy[:, indexFa]
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_fa_only, y, path=crohnPath)
    train_means, train_std, faMeans, faStd, faROC = svm.svm_shuffle(outer=5, shuffle_time=svmTime,
                                                                          para={'kernel': 'linear',
                                                                                'gamma': 15,
                                                                                'C': 60},
                                                                          svm_metrics=['accuracy',
                                                                                       'sensitivity',
                                                                                       'specificity'])
    axe_roc.plot(faROC['fpr_test'], faROC['tpr_test'],
                 label=r'FA only AUC = %0.2f' % faROC['auc_test'],
                 lw=2, alpha=.8)

    # pipeline9: md only
    X_train, X_test, y_train, y_test = train_test_split(mdNumpy, y, test_size=0.25, stratify=y)
    # 使用elastic net进行特征选择
    elas = elastic_net(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                       feature_name=multimodalName, path=crohnPath, cv_val=False)

    X_train_new, X_test_new, _, _, _, indexMd = elas.elasticnet_shuffle(shuffle_time=100,
                                                                        l1_range=1.0,
                                                                        alphas_range=0.000005)
    # indexMd = [10, 15, 2, 24, 3, 40, 41, 42, 58, 59, 64, 69, 72, 76, 80]
    X_md_only = mdNumpy[:, indexMd]
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_md_only, y, path=crohnPath)
    train_means, train_std, mdMeans, mdStd, mdROC = svm.svm_shuffle(outer=5, shuffle_time=svmTime,
                                                                    para={'kernel': 'linear',
                                                                          'gamma': 15,
                                                                          'C': 60},
                                                                    svm_metrics=['accuracy',
                                                                                 'sensitivity',
                                                                                 'specificity'])
    axe_roc.plot(mdROC['fpr_test'], mdROC['tpr_test'],
                 label=r'MD only AUC = %0.2f' % mdROC['auc_test'],
                 lw=2, alpha=.8)

    # pipeline10: rd only
    X_train, X_test, y_train, y_test = train_test_split(rdNumpy, y, test_size=0.25, stratify=y)
    # 使用elastic net进行特征选择
    elas = elastic_net(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                       feature_name=multimodalName, path=crohnPath, cv_val=False)

    X_train_new, X_test_new, _, _, _, indexRd = elas.elasticnet_shuffle(shuffle_time=100,
                                                                        l1_range=1.0,
                                                                        alphas_range=0.0000000009)
    # indexRd = [10, 21, 27, 4, 40, 45, 50, 59, 68, 8, 76, 80]
    X_rd_only = rdNumpy[:, indexRd]
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_rd_only, y, path=crohnPath)
    train_means, train_std, rdMeans, rdStd, rdROC = svm.svm_shuffle(outer=5, shuffle_time=svmTime,
                                                                    para={'kernel': 'linear',
                                                                          'gamma': 15,
                                                                          'C': 60},
                                                                    svm_metrics=['accuracy',
                                                                                 'sensitivity',
                                                                                 'specificity'])
    axe_roc.plot(rdROC['fpr_test'], rdROC['tpr_test'],
                 label=r'RD only AUC = %0.2f' % rdROC['auc_test'],
                 lw=2, alpha=.8)

    # pipeline11: ad only
    X_train, X_test, y_train, y_test = train_test_split(adNumpy, y, test_size=0.25, stratify=y)
    # 使用elastic net进行特征选择
    elas = elastic_net(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                       feature_name=multimodalName, path=crohnPath, cv_val=False)

    X_train_new, X_test_new, _, _, _, indexAd = elas.elasticnet_shuffle(shuffle_time=100,
                                                                        l1_range=1.0,
                                                                        alphas_range=0.00001)
    # indexAd = [2, 40, 41, 45, 80]
    X_ad_only = adNumpy[:, indexAd]
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_ad_only, y, path=crohnPath)
    train_means, train_std, adMeans, adStd, adROC = svm.svm_shuffle(outer=5, shuffle_time=svmTime,
                                                                    para={'kernel': 'linear',
                                                                          'gamma': 15,
                                                                          'C': 60},
                                                                    svm_metrics=['accuracy',
                                                                                 'sensitivity',
                                                                                 'specificity'])
    axe_roc.plot(adROC['fpr_test'], adROC['tpr_test'],
                 label=r'AD only AUC = %0.2f' % adROC['auc_test'],
                 lw=2, alpha=.8)

    # pipeline12: fa_matrix only
    X_train, X_test, y_train, y_test = train_test_split(selectFaMat, y, test_size=0.25, stratify=y)
    # 使用elastic net进行特征选择
    elas = elastic_net(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                       feature_name=multimodalName, path=crohnPath, cv_val=False)

    X_train_new, X_test_new, _, _, _, indexFaMatrix = elas.elasticnet_shuffle(shuffle_time=100,
                                                                        l1_range=0.5,
                                                                        alphas_range=0.13)
    # indexFaMatrix = [1, 3, 5, 7, 10, 13, 14, 24, 25, 27, 33, 38]
    X_famat_only = selectFaMat[:, indexFaMatrix]
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_famat_only, y, path=crohnPath)
    train_means, train_std, faMatMeans, faMatStd, faMatROC = svm.svm_shuffle(outer=5, shuffle_time=svmTime,
                                                                    para={'kernel': 'linear',
                                                                          'gamma': 15,
                                                                          'C': 60},
                                                                    svm_metrics=['accuracy',
                                                                                 'sensitivity',
                                                                                 'specificity'])
    axe_roc.plot(faMatROC['fpr_test'], faMatROC['tpr_test'],
                 label=r'FA Matrix only AUC = %0.2f' % faMatROC['auc_test'],
                 lw=2, alpha=.8)

    # pipeline13: fn only
    X_train, X_test, y_train, y_test = train_test_split(selectFN, y, test_size=0.25, stratify=y)
    # 使用elastic net进行特征选择
    elas = elastic_net(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                       feature_name=multimodalName, path=crohnPath, cv_val=False)

    X_train_new, X_test_new, _, _, _, indexFn = elas.elasticnet_shuffle(shuffle_time=100,
                                                                              l1_range=0.5,
                                                                              alphas_range=0.2)
    # indexFn = [1, 38, 39, 5, 7, 8, 15, 18, 19, 20, 21, 22, 37, 24, 26, 31]
    indexFn = [251, 256, 260, 269, 271]
    X_fn_only = multimodal[:, indexFn]
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_fn_only, y, path=crohnPath)
    train_means, train_std, fnMeans, fnStd, fnROC = svm.svm_shuffle(outer=5, shuffle_time=svmTime,
                                                                             para={'kernel': 'linear',
                                                                                   'gamma': 15,
                                                                                   'C': 60},
                                                                             svm_metrics=['accuracy',
                                                                                          'sensitivity',
                                                                                          'specificity'])
    axe_roc.plot(fnROC['fpr_test'], fnROC['tpr_test'],
                 label=r'Fiber Number only AUC = %0.2f' % fnROC['auc_test'],
                 lw=2, alpha=.8)

    # pipeline14: length only
    X_train, X_test, y_train, y_test = train_test_split(selectLength, y, test_size=0.25, stratify=y)
    # 使用elastic net进行特征选择
    elas = elastic_net(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                       feature_name=multimodalName, path=crohnPath, cv_val=False)

    # X_train_new, X_test_new, _, _, _, indexLength = elas.elasticnet_shuffle(shuffle_time=100,
    #                                                                     l1_range=0.5,
    #                                                                     alphas_range=0.05)
    # indexLength = [2, 4, 5, 7, 8, 9, 13, 15, 18, 23, 31, 32, 35]
    indexLength = [290, 291, 296, 297, 300, 321, 323]
    X_length_only = multimodal[:, indexLength]
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_length_only, y, path=crohnPath)
    train_means, train_std, lengthMeans, lengthStd, lengthROC = svm.svm_shuffle(outer=5, shuffle_time=svmTime,
                                                                    para={'kernel': 'linear',
                                                                          'gamma': 15,
                                                                          'C': 60},
                                                                    svm_metrics=['accuracy',
                                                                                 'sensitivity',
                                                                                 'specificity'])
    axe_roc.plot(lengthROC['fpr_test'], lengthROC['tpr_test'],
                 label=r'Fiber Length only AUC = %0.2f' % lengthROC['auc_test'],
                 lw=2, alpha=.8)

    # 最终的ROC设置
    axe_roc.grid(False)
    axe_roc.plot([0, 1], [0, 1], 'r--')
    axe_roc.set_xlim([-0.01, 1.01])
    axe_roc.set_ylim([-0.01, 1.01])
    axe_roc.set_xlabel('False Positive Rate', fontsize=11)
    axe_roc.set_ylabel('True Positive Rate', fontsize=11)
    axe_roc.set_title('ROC for classification of CD vs. HC', fontsize=15)
    axe_roc.legend(loc="lower right", prop={'size': 10})
    fig_roc.savefig(os.path.join(crohnPath, 'ROC_seperate.png'), dpi=300)

    # 绘制bar图
    meansList = (np.mean(mklMeans['accuracy']), np.mean(concatMeans['accuracy']),
                 np.mean(fmriMeans['accuracy']), np.mean(DTIMeans['accuracy']), np.mean(smriMeans['accuracy']),
                 np.mean(falffMeans['accuracy']), np.mean(fcMeans['accuracy']), np.mean(rehoMeans['accuracy']),
                 np.mean(faMeans['accuracy']), np.mean(mdMeans['accuracy']), np.mean(rdMeans['accuracy']),
                 np.mean(adMeans['accuracy']),
                 np.mean(faMatMeans['accuracy']), np.mean(fnMeans['accuracy']), np.mean(lengthMeans['accuracy']))
    stdList = (np.mean(mklStd['accuracy']), np.mean(concatStd['accuracy']), np.mean(fmriStd['accuracy']),
               np.mean(DTIStd['accuracy']), np.mean(smriStd['accuracy']), np.mean(falffStd['accuracy']),
               np.mean(fcStd['accuracy']), np.mean(rehoStd['accuracy']),
               np.mean(faStd['accuracy']), np.mean(mdStd['accuracy']), np.mean(rdStd['accuracy']),
               np.mean(adStd['accuracy']), np.mean(faMatStd['accuracy']), np.mean(fnStd['accuracy']),
               np.mean(lengthStd['accuracy']))

    totalNum = np.arange(fMRI_num + DTI_num + sMRI_num)
    fig_bar, axe_bar = plt.subplots()
    fig_bar.set_size_inches(15, 10)

    xName = ('Multi-kernel', 'Concatenate', 'rs-fMRI', 'DTI', 'sMRI', 'fALFF', 'FC', 'ReHo',
             'FA', 'MD', 'RD', 'AD', 'FA Matrix', 'Fiber Number', 'Fiber Length')
    axe_bar.bar(xName, meansList, width=0.7, yerr=stdList, error_kw={'elinewidth': 3,
                                                                     'ecolor': 'orangered',
                                                                     'capsize': 4})

    # axe_bar.set_ylim([-0.01, 1.01])
    axe_bar.grid(True)
    axe_bar.set_xticklabels(xName, rotation=45, ha='right', rotation_mode='anchor')

    axe_bar.set_xlabel('Measurements')
    axe_bar.set_ylabel('Accuracy')
    axe_bar.set_title('Accuracy for classification of CD vs. HC')
    fig_bar.savefig(os.path.join(crohnPath, 'bar_fig.png'), dpi=300)
'''

# 绘制除了connectivity之外的特征hist和分布图
'''
if __name__ == '__main__':
    # 设定结果存储路径并创建文件夹
    timeDir = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    crohnPath = os.path.join('crohn', 'dti+fmri', timeDir)
    os.makedirs(crohnPath, exist_ok=True)

    # 读取数据
    dataFalff = pd.read_csv('./mkl_feature_1007/feature_fMRI/mALFF_classic_global.csv', sep=',', engine='python')
    dataFc = pd.read_csv('./mkl_feature_1007/feature_fMRI/FC_feature_90_R_G.csv', sep=',', engine='python')
    dataReho = pd.read_csv('./mkl_feature_1007/feature_fMRI/smReHo_G_90.csv', sep=',', engine='python')

    dataFA = pd.read_csv('./mkl_feature_1007/feature_Native_space/FA_90.csv', sep=',', engine='python')
    dataMD = pd.read_csv('./mkl_feature_1007/feature_Native_space/MD_90.csv', sep=',', engine='python')
    dataRD = pd.read_csv('./mkl_feature_1007/feature_Native_space/RD_90.csv', sep=',', engine='python')
    dataAD = pd.read_csv('./mkl_feature_1007/feature_Native_space/L1_90.csv', sep=',', engine='python')
    # dataL2 = pd.read_csv('./mkl_feature_1007/feature_Native_space/FA_90.csv', sep=',', engine='python')
    # dataL3 = pd.read_csv('./mkl_feature_1007/feature_Native_space/FA_90.csv', sep=',', engine='python')

    dataFaMatrix = pd.read_csv('./mkl_feature_1007/feature_Deterministi/FA_Matrix_90.csv', sep=',', engine='python')
    dataFN = pd.read_csv('./mkl_feature_1007/feature_Deterministi/FN_Matrix_90.csv', sep=',', engine='python')
    dataLength = pd.read_csv('./mkl_feature_1007/feature_Deterministi/Length_Matrix_Mapped_90.csv', sep=',',
                             engine='python')

    dataVbm = pd.read_csv('./mkl_feature_1007/vbm/VBM_90.csv', sep=',', engine='python')

    # 读取label
    label = pd.read_csv('./mkl_feature_1007/label.csv', sep=',', engine='python', header=None)

    # 数据统一转换为numpy格式
    falffNumpy = np.array(dataFalff.iloc[:, :])
    fcNumpy = np.array(dataFc.iloc[:, :])
    rehoNumpy = np.array(dataReho.iloc[:, :])
    faNumpy = np.array(dataFA.iloc[:, :])
    mdNumpy = np.array(dataMD.iloc[:, :])
    rdNumpy = np.array(dataRD.iloc[:, :])
    adNumpy = np.array(dataAD.iloc[:, :])
    # l2Numpy = np.array(dataL2.iloc[:, :])
    # l3Numpy = np.array(dataL3.iloc[:, :])
    faMatrixNumpy = np.array(dataFaMatrix.iloc[:, :])
    fnNumpy = np.array(dataFN.iloc[:, :])
    lengthNumpy = np.array(dataLength.iloc[:, :])
    vbmNumpy = np.array(dataVbm.iloc[:, :])
    # 标签转换为numpy格式
    y = np.array(label.iloc[:, :]).ravel()
    # 规定病人为0，健康人为1
    labelName = ['0', '1']
    posLabel = 1

    # 存储特征名称
    falffName = dataFalff.columns._values
    fcName = dataFc.columns._values
    rehoName = dataReho.columns._values
    faName = dataFA.columns._values
    mdName = dataMD.columns._values
    rdName = dataRD.columns._values
    adName = dataAD.columns._values
    # l2Name = dataL2.columns._values
    # l3Name = dataL3.columns._values
    faMatrixName = dataFaMatrix.columns._values
    fnName = dataFN.columns._values
    lengthName = dataLength.columns._values
    vbmName = dataVbm.columns._values

    # 单独对FC, FA Matrix, FN, Length使用留一法U检验进行特征选择
    fcUPath = os.path.join(crohnPath, 'fc_utest')
    os.makedirs(fcUPath, exist_ok=True)
    utestFC = Utest(fcNumpy, fcName,
                    dis_num=63, hc_num=39, thres=0.01, all_value=False, stat_path=fcUPath)
    selectFC, selectFCName, _, _, fcUIndex = utestFC.utest_loo_freq(K_value=40, choice='percent')

    faMatUPath = os.path.join(crohnPath, 'fa_matrix_utest')
    os.makedirs(faMatUPath, exist_ok=True)
    utestFAMatrix = Utest(faMatrixNumpy, faMatrixName,
                          dis_num=63, hc_num=39, thres=0.1, all_value=False, stat_path=faMatUPath)
    selectFaMat, selectFaMatName, _, _, faMatUIndex = utestFAMatrix.utest_loo_freq(K_value=40, choice='percent')

    fnUPath = os.path.join(crohnPath, 'fn_utest')
    os.makedirs(fnUPath, exist_ok=True)
    utestFN = Utest(fnNumpy, fnName,
                    dis_num=63, hc_num=39, thres=0.1, all_value=False, stat_path=fnUPath)
    selectFN, selectFNName, _, _, fnUIndex = utestFN.utest_loo_freq(K_value=40, choice='percent')

    lengthUPath = os.path.join(crohnPath, 'length_utest')
    os.makedirs(lengthUPath, exist_ok=True)
    utestLength = Utest(lengthNumpy, lengthName,
                        dis_num=63, hc_num=39, thres=0.1, all_value=False, stat_path=lengthUPath)
    selectLength, selectLengthName, _, _, lengthUIndex = utestLength.utest_loo_freq(K_value=40, choice='percent')

    # 对数据进行标准化并拼接
    # 拼接功能数据
    fMRI = np.concatenate((falffNumpy, selectFC, rehoNumpy), axis=1)
    fMRIName = np.concatenate((falffName, selectFCName, rehoName), axis=0)
    # 对功能数据进行z标准化
    nor_fMRI = normalization(fMRI)
    fMRI = nor_fMRI.z_score()
    # 功能数据的特征数量
    fMRI_num = len(fMRIName)

    # 对DTI进行拼接
    DTI = np.concatenate((selectFaMat, selectFN, selectLength, faNumpy, mdNumpy, rdNumpy, adNumpy), axis=1)
    DTIName = np.concatenate((selectFaMatName, selectFNName, selectLengthName, faName, mdName, rdName, adName), axis=0)

    # 对DTI数据进行z标准化
    nor_DTI = normalization(DTI)
    DTI = nor_DTI.z_score()
    # DTI数据的特征数量
    DTI_num = len(DTIName)

    # 对结构数据进行拼接
    sMRI = vbmNumpy
    sMRIName = vbmName
    # 对结构数据进行z标准化
    nor_sMRI = normalization(sMRI)
    sMRI = nor_sMRI.z_score()
    # 结构数据的特征数量
    sMRI_num = len(sMRIName)

    # 直接拼接多模态数据
    multimodal = np.concatenate((fMRI, DTI, sMRI), axis=1)
    multimodalName = np.concatenate((fMRIName, DTIName, sMRIName), axis=0)
    # X为最终的输入
    X = multimodal

    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

    index = [620, 625, 0, 480, 489, 564, 565, 142, 181, 200, 730, 735]
    name = ['AD: Amygdala_L', 'AD: Cuneus_R', 'fALFF: Precentral_L', 'MD: Pallidum_L', 'MD: Temporal_Pole_Sup_R',
            'RD: Pallidum_L', 'RD: Pallidum_R', 'ReHo: Supp_Motor_Area_L', 'ReHo: SupraMarginal_R',
            'ReHo: Temporal_Pole_Sup_L', 'GMV: Putamen_L', 'GMV: Thalamus_R']

    sns.set()
    fig, axe = plt.subplots(3, 4, figsize=(30, 12))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)

    for ax, i in zip(axe.flat, range(len(index))):
        cd = X[:63, index[i]]
        hc = X[63:, index[i]]
        sns.distplot(cd, kde_kws={'label': 'CD'}, ax=ax, kde=False)
        sns.distplot(hc, kde_kws={'label': 'HC'}, ax=ax, kde=False)

        ax2 = ax.twinx()
        sns.distplot(cd, kde_kws={'label': 'CD'}, ax=ax2, hist=False)
        sns.distplot(hc, kde_kws={'label': 'HC'}, ax=ax2, hist=False)
        ax.set_title(name[i], fontsize=20)
        # ax.legend(loc="upper left", prop={'size': 12})
        ax.set_xlabel('Value', fontsize=15)
        ax.set_ylabel('Count', fontsize=15)
        ax2.set_ylabel('Density', fontsize=15)

    fig.savefig('hist.png', dpi=300)
    # plt.show()
'''

# 每个特征单独进行分类绘制bar图
'''
if __name__ == '__main__':
    timeDir = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    crohnPath = os.path.join('crohn', 'all_results', timeDir)
    os.makedirs(crohnPath, exist_ok=True)

    # 读取数据
    dataFalff = pd.read_csv('./mkl_feature_1007/feature_fMRI/mALFF_classic_global.csv', sep=',', engine='python')
    dataFc = pd.read_csv('./mkl_feature_1007/feature_fMRI/FC_feature_90_R_G.csv', sep=',', engine='python')
    dataReho = pd.read_csv('./mkl_feature_1007/feature_fMRI/smReHo_G_90.csv', sep=',', engine='python')

    dataFA = pd.read_csv('./mkl_feature_1007/feature_Native_space/FA_90.csv', sep=',', engine='python')
    dataMD = pd.read_csv('./mkl_feature_1007/feature_Native_space/MD_90.csv', sep=',', engine='python')
    dataRD = pd.read_csv('./mkl_feature_1007/feature_Native_space/RD_90.csv', sep=',', engine='python')
    dataAD = pd.read_csv('./mkl_feature_1007/feature_Native_space/L1_90.csv', sep=',', engine='python')
    # dataL2 = pd.read_csv('./mkl_feature_1007/feature_Native_space/FA_90.csv', sep=',', engine='python')
    # dataL3 = pd.read_csv('./mkl_feature_1007/feature_Native_space/FA_90.csv', sep=',', engine='python')

    dataFaMatrix = pd.read_csv('./mkl_feature_1007/feature_Deterministi/FA_Matrix_90.csv', sep=',', engine='python')
    dataFN = pd.read_csv('./mkl_feature_1007/feature_Deterministi/FN_Matrix_90.csv', sep=',', engine='python')
    dataLength = pd.read_csv('./mkl_feature_1007/feature_Deterministi/Length_Matrix_Mapped_90.csv', sep=',', engine='python')

    dataVbm = pd.read_csv('./mkl_feature_1007/vbm/VBM_90.csv', sep=',', engine='python')

    # 读取label
    label = pd.read_csv('./mkl_feature_1007/label.csv', sep=',', engine='python', header=None)

    # 数据统一转换为numpy格式
    falffNumpy = np.array(dataFalff.iloc[:, :])
    fcNumpy = np.array(dataFc.iloc[:, :])
    rehoNumpy = np.array(dataReho.iloc[:, :])
    faNumpy = np.array(dataFA.iloc[:, :])
    mdNumpy = np.array(dataMD.iloc[:, :])
    rdNumpy = np.array(dataRD.iloc[:, :])
    adNumpy = np.array(dataAD.iloc[:, :])
    # l2Numpy = np.array(dataL2.iloc[:, :])
    # l3Numpy = np.array(dataL3.iloc[:, :])
    faMatrixNumpy = np.array(dataFaMatrix.iloc[:, :])
    fnNumpy = np.array(dataFN.iloc[:, :])
    lengthNumpy = np.array(dataLength.iloc[:, :])
    vbmNumpy = np.array(dataVbm.iloc[:, :])
    # 标签转换为numpy格式
    y = np.array(label.iloc[:, :]).ravel()
    # 规定病人为0，健康人为1
    labelName = ['0', '1']
    posLabel = 1

    # 存储特征名称
    falffName = dataFalff.columns._values
    fcName = dataFc.columns._values
    rehoName = dataReho.columns._values
    faName = dataFA.columns._values
    mdName = dataMD.columns._values
    rdName = dataRD.columns._values
    adName = dataAD.columns._values
    # l2Name = dataL2.columns._values
    # l3Name = dataL3.columns._values
    faMatrixName = dataFaMatrix.columns._values
    fnName = dataFN.columns._values
    lengthName = dataLength.columns._values
    vbmName = dataVbm.columns._values

    # 单独对FC, FA Matrix, FN, Length使用留一法U检验进行特征选择
    fcUPath = os.path.join(crohnPath, 'fc_utest')
    os.makedirs(fcUPath, exist_ok=True)
    utestFC = Utest(fcNumpy, fcName,
                    dis_num=63, hc_num=39, thres=0.01, all_value=False, stat_path=fcUPath)
    selectFC, selectFCName, _, _, fcUIndex = utestFC.utest_loo_freq(K_value=40, choice='percent')

    faMatUPath = os.path.join(crohnPath, 'fa_matrix_utest')
    os.makedirs(faMatUPath, exist_ok=True)
    utestFAMatrix = Utest(faMatrixNumpy, faMatrixName,
                          dis_num=63, hc_num=39, thres=0.1, all_value=False, stat_path=faMatUPath)
    selectFaMat, selectFaMatName, _, _, faMatUIndex = utestFAMatrix.utest_loo_freq(K_value=40, choice='percent')

    fnUPath = os.path.join(crohnPath, 'fn_utest')
    os.makedirs(fnUPath, exist_ok=True)
    utestFN = Utest(fnNumpy, fnName,
                    dis_num=63, hc_num=39, thres=0.1, all_value=False, stat_path=fnUPath)
    selectFN, selectFNName, _, _, fnUIndex = utestFN.utest_loo_freq(K_value=40, choice='percent')

    lengthUPath = os.path.join(crohnPath, 'length_utest')
    os.makedirs(lengthUPath, exist_ok=True)
    utestLength = Utest(lengthNumpy, lengthName,
                        dis_num=63, hc_num=39, thres=0.1, all_value=False, stat_path=lengthUPath)
    selectLength, selectLengthName, _, _, lengthUIndex = utestLength.utest_loo_freq(K_value=40, choice='percent')

    # 对数据进行标准化并拼接
    # 拼接功能数据
    fMRI = np.concatenate((falffNumpy, selectFC, rehoNumpy), axis=1)
    fMRIName = np.concatenate((falffName, selectFCName, rehoName), axis=0)
    # 对功能数据进行z标准化
    nor_fMRI = normalization(fMRI)
    fMRI = nor_fMRI.z_score()
    # 功能数据的特征数量
    fMRI_num = len(fMRIName)

    # 对DTI进行拼接
    DTI = np.concatenate((selectFaMat, selectFN, selectLength, faNumpy, mdNumpy, rdNumpy, adNumpy), axis=1)
    DTIName = np.concatenate((selectFaMatName, selectFNName, selectLengthName, faName, mdName, rdName, adName), axis=0)

    # 对DTI数据进行z标准化
    nor_DTI = normalization(DTI)
    DTI = nor_DTI.z_score()
    # DTI数据的特征数量
    DTI_num = len(DTIName)

    # 对结构数据进行拼接
    sMRI = vbmNumpy
    sMRIName = vbmName
    # 对结构数据进行z标准化
    nor_sMRI = normalization(sMRI)
    sMRI = nor_sMRI.z_score()
    # 结构数据的特征数量
    sMRI_num = len(sMRIName)

    # 直接拼接多模态数据
    multimodal = np.concatenate((fMRI, DTI, sMRI), axis=1)
    multimodalName = np.concatenate((fMRIName, DTIName, sMRIName), axis=0)
    # X为最终的输入
    X = multimodal

    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

    # 多模态的索引
    indexMulti = [620, 625, 0, 210, 215, 217, 221, 223, 229, 239, 240, 244, 86, 90, 91, 93, 96, 97, 98, 99,
                  103, 104, 107, 115, 117, 120, 121, 251, 256, 260, 269, 271, 290, 291, 296, 297, 300, 321,
                  323, 480, 489, 564, 565, 142, 181, 200, 730, 735]
    featureName = multimodalName[indexMulti]

    fig_bar, axe_bar = plt.subplots()
    fig_bar.set_size_inches(15, 10)

    for i, name in zip(indexMulti, featureName):
        X_new = X[:, i].reshape(-1, 1)
        svmTime = 100
        svm = SVM(X_new, y, path=crohnPath)
        _, _, Means1, Std1, _ = svm.svm_shuffle(outer=10, shuffle_time=svmTime,
                                                para={'kernel': 'linear', 'gamma': 0.08, 'C': 10},
                                                svm_metrics=['accuracy', 'sensitivity', 'specificity'])

        axe_bar.bar(name, np.mean(Means1['accuracy']), width=0.7, yerr=np.mean(Std1['accuracy']),
                    error_kw={'elinewidth': 3, 'ecolor': 'orangered', 'capsize': 4})

    axe_bar.grid(False)
    axe_bar.set_xticklabels(featureName, rotation=45, ha='right', rotation_mode='anchor')

    axe_bar.set_xlabel('Measurements')
    axe_bar.set_ylabel('Accuracy')
    axe_bar.set_title('Accuracy for CD vs. HC classification')
    fig_bar.savefig(os.path.join(crohnPath, 'bar_sep.png'), dpi=300)
'''


# 绘制相关矩阵
'''
if __name__ == '__main__':
    timeDir = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    crohnPath = os.path.join('crohn', 'corr_mat', timeDir)
    os.makedirs(crohnPath, exist_ok=True)

    # 读取数据
    dataFalff = pd.read_csv('./mkl_feature_1007/feature_fMRI/mALFF_classic_global.csv', sep=',', engine='python')
    dataFc = pd.read_csv('./mkl_feature_1007/feature_fMRI/FC_feature_90_R_G.csv', sep=',', engine='python')
    dataReho = pd.read_csv('./mkl_feature_1007/feature_fMRI/smReHo_G_90.csv', sep=',', engine='python')

    dataFA = pd.read_csv('./mkl_feature_1007/feature_Native_space/FA_90.csv', sep=',', engine='python')
    dataMD = pd.read_csv('./mkl_feature_1007/feature_Native_space/MD_90.csv', sep=',', engine='python')
    dataRD = pd.read_csv('./mkl_feature_1007/feature_Native_space/RD_90.csv', sep=',', engine='python')
    dataAD = pd.read_csv('./mkl_feature_1007/feature_Native_space/L1_90.csv', sep=',', engine='python')
    # dataL2 = pd.read_csv('./mkl_feature_1007/feature_Native_space/FA_90.csv', sep=',', engine='python')
    # dataL3 = pd.read_csv('./mkl_feature_1007/feature_Native_space/FA_90.csv', sep=',', engine='python')

    dataFaMatrix = pd.read_csv('./mkl_feature_1007/feature_Deterministi/FA_Matrix_90.csv', sep=',', engine='python')
    dataFN = pd.read_csv('./mkl_feature_1007/feature_Deterministi/FN_Matrix_90.csv', sep=',', engine='python')
    dataLength = pd.read_csv('./mkl_feature_1007/feature_Deterministi/Length_Matrix_Mapped_90.csv', sep=',', engine='python')

    dataVbm = pd.read_csv('./mkl_feature_1007/vbm/VBM_90.csv', sep=',', engine='python')

    # 读取label
    label = pd.read_csv('./mkl_feature_1007/label.csv', sep=',', engine='python', header=None)

    # 数据统一转换为numpy格式
    falffNumpy = np.array(dataFalff.iloc[:, :])
    fcNumpy = np.array(dataFc.iloc[:, :])
    rehoNumpy = np.array(dataReho.iloc[:, :])
    faNumpy = np.array(dataFA.iloc[:, :])
    mdNumpy = np.array(dataMD.iloc[:, :])
    rdNumpy = np.array(dataRD.iloc[:, :])
    adNumpy = np.array(dataAD.iloc[:, :])
    # l2Numpy = np.array(dataL2.iloc[:, :])
    # l3Numpy = np.array(dataL3.iloc[:, :])
    faMatrixNumpy = np.array(dataFaMatrix.iloc[:, :])
    fnNumpy = np.array(dataFN.iloc[:, :])
    lengthNumpy = np.array(dataLength.iloc[:, :])
    vbmNumpy = np.array(dataVbm.iloc[:, :])
    # 标签转换为numpy格式
    y = np.array(label.iloc[:, :]).ravel()
    # 规定病人为0，健康人为1
    labelName = ['0', '1']
    posLabel = 1

    # 存储特征名称
    falffName = dataFalff.columns._values
    fcName = dataFc.columns._values
    rehoName = dataReho.columns._values
    faName = dataFA.columns._values
    mdName = dataMD.columns._values
    rdName = dataRD.columns._values
    adName = dataAD.columns._values
    # l2Name = dataL2.columns._values
    # l3Name = dataL3.columns._values
    faMatrixName = dataFaMatrix.columns._values
    fnName = dataFN.columns._values
    lengthName = dataLength.columns._values
    vbmName = dataVbm.columns._values

    # 单独对FC, FA Matrix, FN, Length使用留一法U检验进行特征选择
    fcUPath = os.path.join(crohnPath, 'fc_utest')
    os.makedirs(fcUPath, exist_ok=True)
    utestFC = Utest(fcNumpy, fcName,
                    dis_num=63, hc_num=39, thres=0.01, all_value=False, stat_path=fcUPath)
    selectFC, selectFCName, _, _, fcUIndex = utestFC.utest_loo_freq(K_value=40, choice='percent')

    faMatUPath = os.path.join(crohnPath, 'fa_matrix_utest')
    os.makedirs(faMatUPath, exist_ok=True)
    utestFAMatrix = Utest(faMatrixNumpy, faMatrixName,
                          dis_num=63, hc_num=39, thres=0.1, all_value=False, stat_path=faMatUPath)
    selectFaMat, selectFaMatName, _, _, faMatUIndex = utestFAMatrix.utest_loo_freq(K_value=40, choice='percent')

    fnUPath = os.path.join(crohnPath, 'fn_utest')
    os.makedirs(fnUPath, exist_ok=True)
    utestFN = Utest(fnNumpy, fnName,
                    dis_num=63, hc_num=39, thres=0.1, all_value=False, stat_path=fnUPath)
    selectFN, selectFNName, _, _, fnUIndex = utestFN.utest_loo_freq(K_value=40, choice='percent')

    lengthUPath = os.path.join(crohnPath, 'length_utest')
    os.makedirs(lengthUPath, exist_ok=True)
    utestLength = Utest(lengthNumpy, lengthName,
                        dis_num=63, hc_num=39, thres=0.1, all_value=False, stat_path=lengthUPath)
    selectLength, selectLengthName, _, _, lengthUIndex = utestLength.utest_loo_freq(K_value=40, choice='percent')

    # 对数据进行标准化并拼接
    # 拼接功能数据
    fMRI = np.concatenate((falffNumpy, selectFC, rehoNumpy), axis=1)
    fMRIName = np.concatenate((falffName, selectFCName, rehoName), axis=0)
    # 对功能数据进行z标准化
    nor_fMRI = normalization(fMRI)
    fMRI = nor_fMRI.z_score()
    # 功能数据的特征数量
    fMRI_num = len(fMRIName)

    # 对DTI进行拼接
    DTI = np.concatenate((selectFaMat, selectFN, selectLength, faNumpy, mdNumpy, rdNumpy, adNumpy), axis=1)
    DTIName = np.concatenate((selectFaMatName, selectFNName, selectLengthName, faName, mdName, rdName, adName), axis=0)

    # 对DTI数据进行z标准化
    nor_DTI = normalization(DTI)
    DTI = nor_DTI.z_score()
    # DTI数据的特征数量
    DTI_num = len(DTIName)

    # 对结构数据进行拼接
    sMRI = vbmNumpy
    sMRIName = vbmName
    # 对结构数据进行z标准化
    nor_sMRI = normalization(sMRI)
    sMRI = nor_sMRI.z_score()
    # 结构数据的特征数量
    sMRI_num = len(sMRIName)

    # 直接拼接多模态数据
    multimodal = np.concatenate((fMRI, DTI, sMRI), axis=1)
    multimodalName = np.concatenate((fMRIName, DTIName, sMRIName), axis=0)
    # X为最终的输入
    X = multimodal

    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

    # 多模态的索引
    indexMulti = [620, 625, 0, 210, 215, 217, 221, 223, 229, 239, 240, 244, 86, 90, 91, 93, 96, 97, 98, 99,
                  103, 104, 107, 115, 117, 120, 121, 251, 256, 260, 269, 271, 290, 291, 296, 297, 300, 321,
                  323, 480, 489, 564, 565, 142, 181, 200, 730, 735]
    featureName = ['Amygdala_L', 'Cuneus_R',
                   'Precentral_L',
                   'Lingual_L - Frontal_Inf_Tri_R', 'Postcentral_L - Frontal_Mid_Orb_R', 'Precuneus_L - Hippocampus_R',
                   'Putamen_L - Frontal_Inf_Tri_R', 'Frontal_Sup_Medial_R - Frontal_Inf_Tri_L',
                   'Pallidum_R - Fusiform_L', 'Temporal_Pole_Sup_L - Hippocampus_R',
                   'Temporal_Pole_Sup_R - ParaHippocampal_L', 'Rectus_R - Frontal_Mid_Orb_L',
                   'Lingual_L - Hippocampus_R', 'Lingual_R - Hippocampus_R', 'Rolandic_Oper_R - Frontal_Mid_R',
                   'Fusiform_L - Frontal_Inf_Tri_L', 'Fusiform_R - Frontal_Inf_Tri_L', 'Fusiform_R - Supp_Motor_Area_L',
                   'Parietal_Sup_L - Frontal_Inf_Orb_R', 'Parietal_Sup_L - Rolandic_Oper_L', 'Parietal_Inf_L - Frontal_Inf_Orb_R',
                   'Parietal_Inf_L - Rolandic_Oper_L', 'Parietal_Inf_L - Frontal_Mid_Orb_R', 'SupraMarginal_L - Hippocampus_R',
                   'SupraMarginal_L - Parietal_Sup_L', 'Precuneus_L - ParaHippocampal_R', 'Precuneus_L - Amygdala_L',
                   'Supp_Motor_Area_L - Precentral_L', 'Caudate_R - Supp_Motor_Area_L', 'Thalamus_R - Precentral_L',
                   'Temporal_Inf_L - Hippocampus_R', 'Frontal_Mid_Orb_R - Frontal_Mid_L',
                   'Lingual_L - Frontal_Inf_Tri_R', 'Rolandic_Oper_L - Frontal_Sup_Orb_R', 'Postcentral_L - Frontal_Mid_Orb_R',
                   'Supp_Motor_Area_L - Rolandic_Oper_R', 'Paracentral_Lobule_L - Cingulum_Ant_R',
                   'Temporal_Mid_L - Occipital_Inf_R', 'Rectus_R - Frontal_Mid_Orb_L',
                   'Pallidum_L', 'Temporal_Pole_Sup_R',
                   'Pallidum_L', 'Pallidum_R',
                   'Supp_Motor_Area_L', 'SupraMarginal_R', 'Temporal_Pole_Sup_L',
                   'Putamen_L', 'Thalamus_R']

    X_new = X[:, indexMulti]

    X_corr = pd.DataFrame(X_new, columns=featureName)
    corr_mat = X_corr.corr(method='pearson')
    fig, axe = plt.subplots(figsize=(22, 22))
    sns.heatmap(corr_mat, vmax=1, square=True, ax=axe)
    axe.set_xticklabels(featureName, rotation=45, ha='right', rotation_mode='anchor', fontsize=13)
    fig.savefig(os.path.join(crohnPath, 'feature_corr_heatmap.png'), dpi=300)
'''


# 计算选出来特征的统计值

if __name__ == '__main__':
    timeDir = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    crohnPath = os.path.join('crohn', 'statistic_data', timeDir)
    os.makedirs(crohnPath, exist_ok=True)

    # 读取数据
    dataFalff = pd.read_csv('./mkl_feature_1007/feature_fMRI/mALFF_classic_global.csv', sep=',', engine='python')
    dataFc = pd.read_csv('./mkl_feature_1007/feature_fMRI/FC_feature_90_R_G.csv', sep=',', engine='python')
    dataReho = pd.read_csv('./mkl_feature_1007/feature_fMRI/smReHo_G_90.csv', sep=',', engine='python')

    dataFA = pd.read_csv('./mkl_feature_1007/feature_Native_space/FA_90.csv', sep=',', engine='python')
    dataMD = pd.read_csv('./mkl_feature_1007/feature_Native_space/MD_90.csv', sep=',', engine='python')
    dataRD = pd.read_csv('./mkl_feature_1007/feature_Native_space/RD_90.csv', sep=',', engine='python')
    dataAD = pd.read_csv('./mkl_feature_1007/feature_Native_space/L1_90.csv', sep=',', engine='python')
    # dataL2 = pd.read_csv('./mkl_feature_1007/feature_Native_space/FA_90.csv', sep=',', engine='python')
    # dataL3 = pd.read_csv('./mkl_feature_1007/feature_Native_space/FA_90.csv', sep=',', engine='python')

    dataFaMatrix = pd.read_csv('./mkl_feature_1007/feature_Deterministi/FA_Matrix_90.csv', sep=',', engine='python')
    dataFN = pd.read_csv('./mkl_feature_1007/feature_Deterministi/FN_Matrix_90.csv', sep=',', engine='python')
    dataLength = pd.read_csv('./mkl_feature_1007/feature_Deterministi/Length_Matrix_Mapped_90.csv', sep=',', engine='python')

    dataVbm = pd.read_csv('./mkl_feature_1007/vbm/VBM_90.csv', sep=',', engine='python')

    # 读取label
    label = pd.read_csv('./mkl_feature_1007/label.csv', sep=',', engine='python', header=None)

    # 数据统一转换为numpy格式
    falffNumpy = np.array(dataFalff.iloc[:, :])
    fcNumpy = np.array(dataFc.iloc[:, :])
    rehoNumpy = np.array(dataReho.iloc[:, :])
    faNumpy = np.array(dataFA.iloc[:, :])
    mdNumpy = np.array(dataMD.iloc[:, :])
    rdNumpy = np.array(dataRD.iloc[:, :])
    adNumpy = np.array(dataAD.iloc[:, :])
    # l2Numpy = np.array(dataL2.iloc[:, :])
    # l3Numpy = np.array(dataL3.iloc[:, :])
    faMatrixNumpy = np.array(dataFaMatrix.iloc[:, :])
    fnNumpy = np.array(dataFN.iloc[:, :])
    lengthNumpy = np.array(dataLength.iloc[:, :])
    vbmNumpy = np.array(dataVbm.iloc[:, :])
    # 标签转换为numpy格式
    y = np.array(label.iloc[:, :]).ravel()
    # 规定病人为0，健康人为1
    labelName = ['0', '1']
    posLabel = 1

    # 存储特征名称
    falffName = dataFalff.columns._values
    fcName = dataFc.columns._values
    rehoName = dataReho.columns._values
    faName = dataFA.columns._values
    mdName = dataMD.columns._values
    rdName = dataRD.columns._values
    adName = dataAD.columns._values
    # l2Name = dataL2.columns._values
    # l3Name = dataL3.columns._values
    faMatrixName = dataFaMatrix.columns._values
    fnName = dataFN.columns._values
    lengthName = dataLength.columns._values
    vbmName = dataVbm.columns._values

    # 单独对FC, FA Matrix, FN, Length使用留一法U检验进行特征选择
    fcUPath = os.path.join(crohnPath, 'fc_utest')
    os.makedirs(fcUPath, exist_ok=True)
    utestFC = Utest(fcNumpy, fcName,
                    dis_num=63, hc_num=39, thres=0.01, all_value=False, stat_path=fcUPath)
    selectFC, selectFCName, _, _, fcUIndex = utestFC.utest_loo_freq(K_value=40, choice='percent')

    faMatUPath = os.path.join(crohnPath, 'fa_matrix_utest')
    os.makedirs(faMatUPath, exist_ok=True)
    utestFAMatrix = Utest(faMatrixNumpy, faMatrixName,
                          dis_num=63, hc_num=39, thres=0.1, all_value=False, stat_path=faMatUPath)
    selectFaMat, selectFaMatName, _, _, faMatUIndex = utestFAMatrix.utest_loo_freq(K_value=40, choice='percent')

    fnUPath = os.path.join(crohnPath, 'fn_utest')
    os.makedirs(fnUPath, exist_ok=True)
    utestFN = Utest(fnNumpy, fnName,
                    dis_num=63, hc_num=39, thres=0.1, all_value=False, stat_path=fnUPath)
    selectFN, selectFNName, _, _, fnUIndex = utestFN.utest_loo_freq(K_value=40, choice='percent')

    lengthUPath = os.path.join(crohnPath, 'length_utest')
    os.makedirs(lengthUPath, exist_ok=True)
    utestLength = Utest(lengthNumpy, lengthName,
                        dis_num=63, hc_num=39, thres=0.1, all_value=False, stat_path=lengthUPath)
    selectLength, selectLengthName, _, _, lengthUIndex = utestLength.utest_loo_freq(K_value=40, choice='percent')

    # 对数据进行标准化并拼接
    # 拼接功能数据
    fMRI = np.concatenate((falffNumpy, selectFC, rehoNumpy), axis=1)
    fMRIName = np.concatenate((falffName, selectFCName, rehoName), axis=0)
    # # 对功能数据进行z标准化
    # nor_fMRI = normalization(fMRI)
    # fMRI = nor_fMRI.z_score()
    # 功能数据的特征数量
    fMRI_num = len(fMRIName)

    # 对DTI进行拼接
    DTI = np.concatenate((selectFaMat, selectFN, selectLength, faNumpy, mdNumpy, rdNumpy, adNumpy), axis=1)
    DTIName = np.concatenate((selectFaMatName, selectFNName, selectLengthName, faName, mdName, rdName, adName), axis=0)

    # # 对DTI数据进行z标准化
    # nor_DTI = normalization(DTI)
    # DTI = nor_DTI.z_score()
    # DTI数据的特征数量
    DTI_num = len(DTIName)

    # 对结构数据进行拼接
    sMRI = vbmNumpy
    sMRIName = vbmName
    # # 对结构数据进行z标准化
    # nor_sMRI = normalization(sMRI)
    # sMRI = nor_sMRI.z_score()
    # 结构数据的特征数量
    sMRI_num = len(sMRIName)

    # 直接拼接多模态数据
    multimodal = np.concatenate((fMRI, DTI, sMRI), axis=1)
    multimodalName = np.concatenate((fMRIName, DTIName, sMRIName), axis=0)
    # X为最终的输入
    X = multimodal

    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

    # 多模态的索引
    indexMulti = [620, 625, 0, 210, 215, 217, 221, 223, 229, 239, 240, 244, 86, 90, 91, 93, 96, 97, 98, 99,
                  103, 104, 107, 115, 117, 120, 121, 251, 256, 260, 269, 271, 290, 291, 296, 297, 300, 321,
                  323, 480, 489, 564, 565, 142, 181, 200, 730, 735]
    featureName = multimodalName[indexMulti]

    # 计算均值
    cdMean = []
    hcMean = []
    for i in indexMulti:
        cd = X[0: 62, i]
        hc = X[62:, i]
        cd = np.mean(cd, axis=0)
        hc = np.mean(hc, axis=0)
        cdMean.append(cd)
        hcMean.append(hc)

    # 计算标准差
    cdStd = []
    hcStd = []
    for i in indexMulti:
        cd = X[0: 62, i]
        hc = X[62:, i]
        cd = np.std(cd, axis=0)
        hc = np.std(hc, axis=0)
        cdStd.append(cd)
        hcStd.append(hc)

    # 计算p值
    P = []
    for i in indexMulti:
        cd = X[0: 62, i]
        hc = X[62:, i]
        t, p_indiv = ttest_ind(cd, hc)
        P.append(p_indiv)

    stat_matrix = np.array([cdMean, cdStd, hcMean, hcStd, P]).T
    stat_matrix = pd.DataFrame(stat_matrix, columns=['cd_mean', 'cd_std', 'hc_mean', 'hc_std', 'p'])
    stat_matrix.to_csv(os.path.join(crohnPath, 'statistic_result.csv'))

