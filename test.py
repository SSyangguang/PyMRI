import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import normalization, index_split
from sklearn.model_selection import train_test_split

from ml_model import elastic_net, lasso, SVM
from ml_model import mkl_svm
from stat_model import Utest, Ttest
from mics import corr

# plt.style.use('seaborn')
#############################################################
##########crohn多模态数据处理程序（第二批数据结果）#################
#############################################################

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
    elas = elastic_net(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                       feature_name=multimodalName, path=crohnPath, cv_val=False)

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
    axe_mkl.set_title('Accuracy for Crohn vd. HC classification')
    axe_mkl.legend(loc="lower right", prop={'size': 8})
    fig_mkl.savefig(os.path.join(svmMklPath, 'mkl_result.png'))

    # 绘制ROC曲线
    std_auc_train = np.std(roc_dict['auc_train'])
    std_auc_test = np.std(roc_dict['auc_test'])

    fig_mkl_roc, axe_mkl_roc = plt.subplots()
    axe_mkl_roc.plot(roc_dict['fpr_train'], roc_dict['tpr_train'],
                     label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (roc_dict['auc_train'], std_auc_train),
                     lw=2, alpha=.8)
    axe_mkl_roc.plot(roc_dict['fpr_test'], roc_dict['tpr_test'],
                     label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (roc_dict['auc_test'], std_auc_test),
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
    # mod1_path = os.path.join(crohnPath, 'fMRI')
    # mod2_path = os.path.join(crohnPath, 'DTI')
    # os.makedirs(mod1_path, exist_ok=True)
    # os.makedirs(mod2_path, exist_ok=True)
    #
    # index_fMRI_only, index_DTI_only = index_mod.two_reset(fMRI_num, DTI_num)
    #
    # X_new1 = fMRI[:, index_fMRI_only]
    # X_new2 = DTI[:, index_DTI_only]
    # featureName1 = fMRIName[index_fMRI_only]
    # featureName2 = DTIName[index_DTI_only]
    #
    # # 使用svm_shuffle返回fMRI结果并且绘制ROC曲线的示意图
    # svmShufflePath_1 = os.path.join(mod1_path, 'svm_shuffle')
    # os.makedirs(svmShufflePath_1, exist_ok=True)
    # # 设置参数
    # para = {'kernel': ['rbf'],
    #         'gamma': np.arange(0.1, 1, 0.1),
    #         'C': np.arange(1, 100, 1)}
    # svmTime = 100
    # svm = SVM(X_new1, y, path=svmShufflePath_1)
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
    # fig_svm.savefig(os.path.join(svmShufflePath_1, 'shuffle_result.png'))
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
    # fig_svm_roc.savefig(os.path.join(svmShufflePath_1, 'shuffle_ROC.png'))
    #
    # # 使用svm_shuffle返回DTI结果并且绘制ROC曲线的示意图
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
    DTI = np.concatenate((selectFaMat, selectFN, selectLength, faNumpy, mdNumpy, rdNumpy, adNumpy), axis=1)
    DTIName = np.concatenate((selectFaMatName, selectFNName, selectLengthName, faName, mdName, rdName, adName), axis=0)

    # 对结构数据进行z标准化
    nor_DTI = normalization(DTI)
    DTI = nor_DTI.z_score()
    # 结构数据的特征数量
    DTI_num = len(DTIName)

    # 直接拼接多模态数据
    multimodal = np.concatenate((fMRI, DTI), axis=1)
    multimodalName = np.concatenate((fMRIName, DTIName), axis=0)
    # X为最终的输入
    X = multimodal

    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

    index = [620, 625, 0, 210, 215, 217, 221, 223, 229, 239, 240, 244, 86, 90, 91, 93, 96, 97, 98, 99,
             103, 104, 107, 115, 117, 120, 121, 251, 256, 260, 269, 271, 290, 291, 296, 297, 300, 321,
             323, 480, 489, 564, 565, 142, 173, 181, 200]
    featureName = multimodalName[index]

    # 根据fmri和DTI的特征数量，将他们重新分成两个矩阵，以便进行mkl

    index_mod = index_split(index)
    index_fMRI, index_DTI = index_mod.two_split(fMRI_num, DTI_num)
    X_fMRI = X[:, index_fMRI]
    X_DTI = X[:, index_DTI]
    # 不进行模态区分，为了对比试验和特征相关性矩阵
    X_new = X[:, index]

    ###############################
    #####用于每个模态单独索引的分类#####
    ###############################
    mod1_path = os.path.join(crohnPath, 'fMRI')
    mod2_path = os.path.join(crohnPath, 'DTI')
    os.makedirs(mod1_path, exist_ok=True)
    os.makedirs(mod2_path, exist_ok=True)

    index_fMRI_only, index_DTI_only = index_mod.two_reset(fMRI_num, DTI_num)

    X_new1 = fMRI[:, index_fMRI_only]
    X_new2 = DTI[:, index_DTI_only]
    featureName1 = fMRIName[index_fMRI_only]
    featureName2 = DTIName[index_DTI_only]

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

    # # 使用svm_shuffle返回DTI结果并且绘制ROC曲线的示意图
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
    DTI = np.concatenate((selectFaMat, selectFN, selectLength, faNumpy, mdNumpy, rdNumpy, adNumpy), axis=1)
    DTIName = np.concatenate((selectFaMatName, selectFNName, selectLengthName, faName, mdName, rdName, adName), axis=0)

    # 对结构数据进行z标准化
    nor_DTI = normalization(DTI)
    DTI = nor_DTI.z_score()
    # 结构数据的特征数量
    DTI_num = len(DTIName)

    # X为最终的输入
    X = DTI

    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

    # lasso = lasso(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
    #               feature_name=DTIName, path=crohnPath, cv_val=True)
    #
    # X_train_new, X_test_new, \
    # featureName, feature_freq, feature_coef, index = lasso.lasso_shuffle(shuffle_time=100,
    #                                                                      alpha_range=np.arange(0.01, 0.1, 0.01))

    # 使用elastic net进行特征选择
    elas = elastic_net(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                       feature_name=DTIName, path=crohnPath, cv_val=False)

    X_train_new, X_test_new, featureName, _, _, index = elas.elasticnet_shuffle(shuffle_time=100,
                                                                                l1_range=1.0,
                                                                                alphas_range=0.14)

    # index = [106, 21, 22, 23, 419, 597, 614, 654, 266]
    featureName = DTIName[index]

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
    #               feature_name=DTIName, path=crohnPath, cv_val=True)
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
'''

# 所有模型集成
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
    DTI = np.concatenate((selectFaMat, selectFN, selectLength, faNumpy, mdNumpy, rdNumpy, adNumpy), axis=1)
    DTIName = np.concatenate((selectFaMatName, selectFNName, selectLengthName, faName, mdName, rdName, adName), axis=0)

    # 对结构数据进行z标准化
    nor_DTI = normalization(DTI)
    DTI = nor_DTI.z_score()
    # 结构数据的特征数量
    DTI_num = len(DTIName)

    # 直接拼接多模态数据
    multimodal = np.concatenate((fMRI, DTI), axis=1)
    multimodalName = np.concatenate((fMRIName, DTIName), axis=0)
    # X为最终的输入
    X = multimodal

    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

    # 多模态的索引
    indexMulti = [620, 625, 0, 210, 215, 217, 221, 223, 229, 239, 240, 244, 86, 90, 91, 93, 96, 97, 98, 99,
                   103, 104, 107, 115, 117, 120, 121, 251, 256, 260, 269, 271, 290, 291, 296, 297, 300, 321,
                   323, 480, 489, 564, 565, 142, 173, 181, 200]
    featureName = multimodalName[indexMulti]

    # 根据fmri和DTI的特征数量，将他们重新分成两个矩阵，以便进行mkl
    index_mod = index_split(indexMulti)
    index_fMRI, index_DTI = index_mod.two_split(fMRI_num, DTI_num)
    X_fMRI = X[:, index_fMRI]
    X_DTI = X[:, index_DTI]
    # 不进行模态区分，为了对比试验和特征相关性矩阵
    X_new = X[:, indexMulti]

    fig_roc, axe_roc = plt.subplots()

    # pipeline1: 使用svm_shuffle返回结果并且绘制ROC曲线的示意图

    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_new, y, path=crohnPath)
    train_means, train_std, concatMeans, concatStd, concatROC = svm.svm_shuffle(outer=10, shuffle_time=svmTime,
                                                                              para={'kernel': 'linear', 'C': 1},
                                                                                svm_metrics=['accuracy',
                                                                                             'sensitivity',
                                                                                             'specificity'])

    # 绘制ROC曲线
    axe_roc.plot(concatROC['fpr_test'], concatROC['tpr_test'],
                 label=r'Concatenate AUC = %0.2f' % concatROC['auc_test'],
                 lw=2, alpha=.8)

    # pipeline2: 使用mkl返回结果

    mkl_kernel = {'kernel_type1': 'rbf',
                  'kernel_type2': 'rbf',
                  'kernel_weight1': 0.3,
                  'kernel_weight2': 0.7}
    mkl_para = {'kernel1': 15,
                'kernel2': 50,
                'C': 80}
    mkl = mkl_svm(X_fMRI, X_DTI, y, mkl_path=crohnPath)
    train_means, train_std, mklMeans, mklStd, mklROC = mkl.mksvm(kernel_dict=mkl_kernel,
                                                                       para=mkl_para,
                                                                       svm_metrics=['accuracy',
                                                                                    'sensitivity',
                                                                                    'specificity'])
    # 绘制ROC曲线
    mklAUCStd = np.std(mklROC['auc_test'])
    axe_roc.plot(mklROC['fpr_test'], mklROC['tpr_test'],
                     label=r'Multimodality AUC = %0.2f' % mklROC['auc_test'],
                     lw=2, alpha=.8)

    # pipeline3: fMRI独立结果
    indexfMRI = [0, 52, 77, 82, 86, 93, 94, 96, 97, 98, 103, 107, 110, 115, 117, 119]
    X_fMRI_only = fMRI[:, indexfMRI]
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_fMRI_only, y, path=crohnPath)
    train_means, train_std, fmriMeans, fmriStd, fmriROC = svm.svm_shuffle(outer=10, shuffle_time=svmTime,
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

    # pipeline4； DTId独立结果
    indexDTI = [412, 414, 417, 448, 121, 137, 148, 189, 1, 5, 13, 14, 31, 37, 41, 43, 79,
                 51, 54, 55, 59, 74, 75, 76, 77, 64, 65, 81, 82, 104, 113, 115]
    X_DTI_only = DTI[:, indexDTI]
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_DTI_only, y, path=crohnPath)
    train_means, train_std, DTIMeans, DTIStd, DTIROC = svm.svm_shuffle(outer=10, shuffle_time=svmTime,
                                                                          para={'kernel': 'linear',
                                                                                'gamma': 15,
                                                                                'C': 100},
                                                                          svm_metrics=['accuracy',
                                                                                       'sensitivity',
                                                                                       'specificity'])
    axe_roc.plot(DTIROC['fpr_test'], DTIROC['tpr_test'],
                 label=r'rs-fMRI only AUC = %0.2f' % fmriROC['auc_test'],
                 lw=2, alpha=.8)

    # pipeline5: falff only
    indexFalff = [17, 29, 52, 55, 77, 82, 83]
    X_falff_only = falffNumpy[:, indexFalff]
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_falff_only, y, path=crohnPath)
    train_means, train_std, falffMeans, falffStd, falffROC = svm.svm_shuffle(outer=10, shuffle_time=svmTime,
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
    indexFc = [1047, 1073, 1093, 1188, 1303, 1342, 1352, 1447, 1457, 1535, 196, 1769, 1870, 1905,
               1946, 2116, 2132, 2473, 2553, 2701, 2702, 2813, 2823, 2912, 2966, 3085, 3216, 3236,
               3246, 3254, 377, 3311, 384, 39, 3387, 524, 61, 858, 887, 899]
    X_fc_only = fcNumpy[:, indexFc]
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_fc_only, y, path=crohnPath)
    train_means, train_std, fcMeans, fcStd, fcROC = svm.svm_shuffle(outer=10, shuffle_time=svmTime,
                                                                    para={'kernel': 'linear',
                                                                                   'gamma': 15,
                                                                                   'C': 60},
                                                                    svm_metrics=['accuracy',
                                                                                'sensitivity',
                                                                                'specificity'])
    axe_roc.plot(fcROC['fpr_test'], fcROC['tpr_test'],
                 label=r'FC+U test AUC = %0.2f' % fcROC['auc_test'],
                 lw=2, alpha=.8)

    # pipeline7: reho only
    indexReho = [10, 19, 23, 48, 49, 57, 58, 65, 69, 72, 76, 77, 79, 81, 82, 83]
    X_reho_only = rehoNumpy[:, indexReho]
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_reho_only, y, path=crohnPath)
    train_means, train_std, rehoMeans, rehoStd, rehoROC = svm.svm_shuffle(outer=10, shuffle_time=svmTime,
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
    indexFa = [10, 12, 23, 24, 28, 32, 33, 41, 44, 46, 6, 56, 58, 64, 69, 77]
    X_fa_only = faNumpy[:, indexFa]
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_fa_only, y, path=crohnPath)
    train_means, train_std, faMeans, faStd, faROC = svm.svm_shuffle(outer=10, shuffle_time=svmTime,
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
    indexMd = [10, 15, 2, 24, 3, 40, 41, 42, 58, 59, 64, 69, 72, 76, 80]
    X_md_only = mdNumpy[:, indexMd]
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_md_only, y, path=crohnPath)
    train_means, train_std, mdMeans, mdStd, mdROC = svm.svm_shuffle(outer=10, shuffle_time=svmTime,
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
    indexRd = [10, 21, 27, 4, 40, 45, 50, 59, 68, 8, 76, 80]
    X_rd_only = rdNumpy[:, indexRd]
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_rd_only, y, path=crohnPath)
    train_means, train_std, rdMeans, rdStd, rdROC = svm.svm_shuffle(outer=10, shuffle_time=svmTime,
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
    indexAd = [2, 40, 41, 45, 80]
    X_ad_only = adNumpy[:, indexAd]
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_ad_only, y, path=crohnPath)
    train_means, train_std, adMeans, adStd, adROC = svm.svm_shuffle(outer=10, shuffle_time=svmTime,
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
    indexFaMatrix = [1, 3, 5, 7, 10, 13, 14, 24, 25, 27, 33, 38]
    X_famat_only = selectFaMat[:, indexFaMatrix]
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_famat_only, y, path=crohnPath)
    train_means, train_std, faMatMeans, faMatStd, faMatROC = svm.svm_shuffle(outer=10, shuffle_time=svmTime,
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
    indexFn = [1, 38, 39, 5, 7, 8, 15, 18, 19, 20, 21, 22, 37, 24, 26, 31]
    X_fn_only = selectFN[:, indexFn]
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_fn_only, y, path=crohnPath)
    train_means, train_std, fnMeans, fnStd, fnROC = svm.svm_shuffle(outer=10, shuffle_time=svmTime,
                                                                             para={'kernel': 'linear',
                                                                                   'gamma': 15,
                                                                                   'C': 60},
                                                                             svm_metrics=['accuracy',
                                                                                          'sensitivity',
                                                                                          'specificity'])
    axe_roc.plot(fnROC['fpr_test'], fnROC['tpr_test'],
                 label=r'FN only AUC = %0.2f' % fnROC['auc_test'],
                 lw=2, alpha=.8)

    # pipeline14: length only
    indexLength = [2, 4, 5, 7, 8, 9, 13, 15, 18, 23, 31, 32, 35]
    X_length_only = selectLength[:, indexLength]
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_length_only, y, path=crohnPath)
    train_means, train_std, lengthMeans, lengthStd, lengthROC = svm.svm_shuffle(outer=10, shuffle_time=svmTime,
                                                                    para={'kernel': 'linear',
                                                                          'gamma': 15,
                                                                          'C': 60},
                                                                    svm_metrics=['accuracy',
                                                                                 'sensitivity',
                                                                                 'specificity'])
    axe_roc.plot(lengthROC['fpr_test'], lengthROC['tpr_test'],
                 label=r'Length only AUC = %0.2f' % lengthROC['auc_test'],
                 lw=2, alpha=.8)

    # 绘制ROC曲线
    DTIAUCStd = np.std(DTIROC['auc_test'])

    axe_roc.plot(DTIROC['fpr_test'], DTIROC['tpr_test'],
                 label=r'DTI only AUC = %0.2f' % fmriROC['auc_test'],
                 lw=2, alpha=.8)

    # 最终的ROC设置
    axe_roc.grid(True)
    axe_roc.plot([0, 1], [0, 1], 'r--')
    axe_roc.set_xlim([-0.01, 1.01])
    axe_roc.set_ylim([-0.01, 1.01])
    axe_roc.set_xlabel('False Positive Rate', fontsize=15)
    axe_roc.set_ylabel('True Positive Rate', fontsize=15)
    axe_roc.set_title('ROC for Crohn vd. HC classification', fontsize=18)
    axe_roc.legend(loc="lower right", prop={'size': 8})
    fig_roc.savefig(os.path.join(crohnPath, 'ROC.png'))

    # 绘制bar图
    meansList = (np.mean(concatMeans['accuracy']), np.mean(mklMeans['accuracy']),
                 np.mean(fmriMeans['accuracy']), np.mean(DTIMeans['accuracy']),
                 np.mean(falffMeans['accuracy']), np.mean(fcMeans['accuracy']), np.mean(rehoMeans['accuracy']),
                 np.mean(faMeans['accuracy']), np.mean(mdMeans['accuracy']), np.mean(rdMeans['accuracy']),
                 np.mean(adMeans['accuracy']),
                 np.mean(faMatMeans['accuracy']), np.mean(fnMeans['accuracy']), np.mean(lengthMeans['accuracy']))
    stdList = (np.mean(concatStd['accuracy']), np.mean(mklStd['accuracy']), np.mean(fmriStd['accuracy']),
               np.mean(DTIStd['accuracy']), np.mean(falffStd['accuracy']),
               np.mean(fcStd['accuracy']), np.mean(rehoStd['accuracy']),
               np.mean(faStd['accuracy']), np.mean(mdStd['accuracy']), np.mean(rdStd['accuracy']),
               np.mean(adStd['accuracy']), np.mean(faMatStd['accuracy']), np.mean(fnStd['accuracy']),
               np.mean(lengthStd['accuracy']))

    totalNum = np.arange(fMRI_num + DTI_num)
    fig_bar, axe_bar = plt.subplots()
    fig_bar.set_size_inches(15, 10)

    xName = ('concatenate', 'multi-kernel', 'rs-fMRI', 'DTI', 'fALFF', 'FC+U', 'ReHo',
             'FA', 'MD', 'RD', 'AD', 'FA Matrix', 'FN', 'Length')
    axe_bar.bar(xName, meansList, width=0.35, yerr=stdList)

    # axe_bar.set_ylim([-0.01, 1.01])
    axe_bar.set_xticklabels(xName, rotation=45)
    axe_bar.set_xlabel('Measurements')
    axe_bar.set_ylabel('Accuracy')
    axe_bar.set_title('Accuracy for Crohn vs. HC classification')
    fig_bar.savefig(os.path.join(crohnPath, 'bar_fig.png'))
'''
#############################################################
##########crohn多模态数据处理程序（第一批数据结果）#################
#############################################################
'''
if __name__ == '__main__':
    # 设定结果存储路径并创建文件夹
    timeDir = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    crohnPath = os.path.join('crohn', 'fa+fmri', timeDir)
    os.makedirs(crohnPath, exist_ok=True)

    # 读取数据
    dataFalff = pd.read_csv('./DTI_RSFC_feature/RSFC/fALFF_feature.csv', sep=',', engine='python')
    dataFc = pd.read_csv('./DTI_RSFC_feature/RSFC/FC_feature.csv', sep=',', engine='python')
    dataReho = pd.read_csv('./DTI_RSFC_feature/RSFC/ReHo_feature.csv', sep=',', engine='python')

    dataFA = pd.read_csv('./DTI_RSFC_feature/DTI/FA_feature_3mm.csv', sep=',', engine='python')
    dataMD = pd.read_csv('./DTI_RSFC_feature/DTI/MD_feature_3mm.csv', sep=',', engine='python')
    dataRD = pd.read_csv('./DTI_RSFC_feature/DTI/RD_feature_3mm.csv', sep=',', engine='python')
    dataL1 = pd.read_csv('./DTI_RSFC_feature/DTI/L1_feature_3mm.csv', sep=',', engine='python')
    dataL2 = pd.read_csv('./DTI_RSFC_feature/DTI/L2_feature_3mm.csv', sep=',', engine='python')
    dataL3 = pd.read_csv('./DTI_RSFC_feature/DTI/L3_feature_3mm.csv', sep=',', engine='python')

    # 读取label
    label = pd.read_csv('./DTI_RSFC_feature/label.csv', sep=',', engine='python', header=None)

    # 数据统一转换为numpy格式
    falffNumpy = np.array(dataFalff.iloc[:, :])
    fcNumpy = np.array(dataFc.iloc[:, :])
    rehoNumpy = np.array(dataReho.iloc[:, :])
    faNumpy = np.array(dataFA.iloc[:, :])
    mdNumpy = np.array(dataMD.iloc[:, :])
    rdNumpy = np.array(dataRD.iloc[:, :])
    l1Numpy = np.array(dataL1.iloc[:, :])
    l2Numpy = np.array(dataL2.iloc[:, :])
    l3Numpy = np.array(dataL3.iloc[:, :])
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
    l1Name = dataL1.columns._values
    l2Name = dataL2.columns._values
    l3Name = dataL3.columns._values

    # 单独对FC使用留一法U检验进行特征选择

    # ttest = Ttest(fcNumpy, fcName,
    #               dis_num=65, hc_num=40, thres=0.05, all_value=False)
    utest = Utest(fcNumpy, fcName,
                  dis_num=65, hc_num=40, thres=0.01, all_value=False, stat_path=crohnPath)

    selectFeature, selectFeatureName, _, _, uTestIndex = utest.utest_loo_freq(K_value=40, choice='percent')

    # 对数据进行标准化并拼接
    # 拼接功能数据
    fMRI = np.concatenate((falffNumpy, selectFeature, rehoNumpy), axis=1)
    fMRIName = np.concatenate((falffName, selectFeatureName, rehoName), axis=0)
    # 对功能数据进行z标准化
    nor_fMRI = normalization(fMRI)
    fMRI = nor_fMRI.z_score()
    # 功能数据的特征数量
    fMRI_num = len(fMRIName)

    # 对结构进行拼接
    DTI = np.concatenate((faNumpy, mdNumpy, rdNumpy, l1Numpy, l2Numpy, l3Numpy), axis=1)
    DTIName = np.concatenate((faName, mdName, rdName, l1Name, l2Name, l3Name), axis=0)

    # 对结构数据进行z标准化
    nor_DTI = normalization(DTI)
    DTI = nor_DTI.z_score()
    # 结构数据的特征数量
    DTI_num = len(DTIName)

    # 直接拼接多模态数据
    multimodal = np.concatenate((fMRI, DTI), axis=1)
    multimodalName = np.concatenate((fMRIName, DTIName), axis=0)
    # X为最终的输入
    X = multimodal

    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y)

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
    #                                                                             l1_range=1.0,
    #                                                                             alphas_range=0.13)

    index = [382, 277, 278, 319, 329, 341, 357, 107, 11, 12, 42, 48, 52, 85, 98, 99, 118, 120, 123, 124, 127,
             130, 133, 138, 144, 148, 156, 157, 168, 264, 272, 177, 188, 206, 164, 219, 222, 240, 242, 244, 167]
    featureName = multimodalName[index]

    # 根据fmri和DTI的特征数量，将他们重新分成两个矩阵，以便进行mkl
    index_fMRI = [i for i in index if i < fMRI_num]
    index_DTI = [i for i in index if (i >= fMRI_num) and (i < DTI_num+fMRI_num)]
    X_fMRI = X[:, index_fMRI]
    X_DTI = X[:, index_DTI]
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

    # 使用mkl返回结果并且绘制ROC曲线的示意图
    svmMklPath = os.path.join(crohnPath, 'mkl_svm')
    os.makedirs(svmMklPath, exist_ok=True)

    mkl_kernel = {'kernel_type1': 'rbf',
                  'kernel_type2': 'linear',
                  'kernel_weight1': 0.3,
                  'kernel_weight2': 0.7}
    mkl_para = {'kernel1': 15,
                'kernel2': 0.1,
                'C': 130}
    mkl = mkl_svm(X_fMRI, X_DTI, y, mkl_path=crohnPath)
    train_means, train_std, test_means, test_std, roc_dict = mkl.mksvm(kernel_dict=mkl_kernel,
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
    axe_mkl.set_title('Accuracy for Crohn vd. HC classification')
    axe_mkl.legend(loc="lower right", prop={'size': 8})
    fig_mkl.savefig(os.path.join(svmMklPath, 'mkl_result.png'))

    # 绘制ROC曲线
    std_auc_train = np.std(roc_dict['auc_train'])
    std_auc_test = np.std(roc_dict['auc_test'])

    fig_mkl_roc, axe_mkl_roc = plt.subplots()
    axe_mkl_roc.plot(roc_dict['fpr_train'], roc_dict['tpr_train'],
                     label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (roc_dict['auc_train'], std_auc_train),
                     lw=2, alpha=.8)
    axe_mkl_roc.plot(roc_dict['fpr_test'], roc_dict['tpr_test'],
                     label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (roc_dict['auc_test'], std_auc_test),
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
    axe_mkl_roc.set_title('ROC for Crohn vd. HC classification', fontsize=18)
    axe_mkl_roc.legend(loc="lower right", prop={'size': 8})
    fig_mkl_roc.savefig(os.path.join(svmMklPath, 'MKL_ROC.png'))

    # mkl-svm的权重网格搜索法
    train_means, test_means, weight_name = mkl.mksvm_grid(kernel_dict=mkl_kernel,
                                                          para=mkl_para,
                                                          grid_num=10,
                                                          svm_metrics=['accuracy',
                                                                       'sensitivity',
                                                                       'specificity'])


    # 绘制特征之间的相关性并存储
    corr_mat = corr(X_new, featureName, crohnPath)
    corr_mat.corr_heatmap()
'''


# fMRI单模态
'''
if __name__ == '__main__':
    # 设定结果存储路径并创建文件夹
    timeDir = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    crohnPath = os.path.join('crohn', 'fmri', timeDir)
    os.makedirs(crohnPath, exist_ok=True)

    # 读取数据
    dataFalff = pd.read_csv('./DTI_RSFC_feature/RSFC/fALFF_feature.csv', sep=',', engine='python')
    dataFc = pd.read_csv('./DTI_RSFC_feature/RSFC/FC_feature.csv', sep=',', engine='python')
    dataReho = pd.read_csv('./DTI_RSFC_feature/RSFC/ReHo_feature.csv', sep=',', engine='python')

    # 读取label
    label = pd.read_csv('./DTI_RSFC_feature/label.csv', sep=',', engine='python', header=None)

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

    # ttest = Ttest(fcNumpy, fcName,
    #               dis_num=65, hc_num=40, thres=0.05, all_value=False)
    utest = Utest(fcNumpy, fcName,
                  dis_num=65, hc_num=40, thres=0.01, all_value=False, stat_path=crohnPath)

    selectFeature, selectFeatureName, _, _, uTestIndex = utest.utest_loo_freq(K_value=40, choice='percent')

    # 对数据进行标准化并拼接
    # 拼接功能数据
    fMRI = np.concatenate((falffNumpy, selectFeature, rehoNumpy), axis=1)
    fMRIName = np.concatenate((falffName, selectFeatureName, rehoName), axis=0)
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

    index = [0, 16, 18, 49, 58, 89, 118, 127, 129, 133, 145, 148, 155, 157, 271, 272, 219, 221]

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

    dataFA = pd.read_csv('./DTI_RSFC_feature/DTI/FA_feature_3mm.csv', sep=',', engine='python')
    dataMD = pd.read_csv('./DTI_RSFC_feature/DTI/MD_feature_3mm.csv', sep=',', engine='python')
    dataRD = pd.read_csv('./DTI_RSFC_feature/DTI/RD_feature_3mm.csv', sep=',', engine='python')
    dataL1 = pd.read_csv('./DTI_RSFC_feature/DTI/L1_feature_3mm.csv', sep=',', engine='python')
    dataL2 = pd.read_csv('./DTI_RSFC_feature/DTI/L2_feature_3mm.csv', sep=',', engine='python')
    dataL3 = pd.read_csv('./DTI_RSFC_feature/DTI/L3_feature_3mm.csv', sep=',', engine='python')

    # 读取label
    label = pd.read_csv('./DTI_RSFC_feature/label.csv', sep=',', engine='python', header=None)

    # 数据统一转换为numpy格式
    faNumpy = np.array(dataFA.iloc[:, :])
    mdNumpy = np.array(dataMD.iloc[:, :])
    rdNumpy = np.array(dataRD.iloc[:, :])
    l1Numpy = np.array(dataL1.iloc[:, :])
    l2Numpy = np.array(dataL2.iloc[:, :])
    l3Numpy = np.array(dataL3.iloc[:, :])
    # 标签转换为numpy格式
    y = np.array(label.iloc[:, :]).ravel()
    # 规定病人为0，健康人为1
    labelName = ['0', '1']
    posLabel = 1

    # 存储特征名称
    faName = dataFA.columns._values[:]
    mdName = dataMD.columns._values[:]
    rdName = dataRD.columns._values[:]
    l1Name = dataL1.columns._values[:]
    l2Name = dataL2.columns._values[:]
    l3Name = dataL3.columns._values[:]

    # 对结构进行拼接
    DTI = np.concatenate((faNumpy, mdNumpy, rdNumpy, l1Numpy, l2Numpy, l3Numpy), axis=1)
    DTIName = np.concatenate((faName, mdName, rdName, l1Name, l2Name, l3Name), axis=0)
    # 对结构数据进行z标准化
    nor_DTI = normalization(DTI)
    DTI = nor_DTI.z_score()
    # 结构数据的特征数量
    DTI_num = len(DTIName)

    # X为最终的输入
    X = DTI

    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

    # lasso = lasso(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
    #               feature_name=DTIName, path=crohnPath, cv_val=True)
    #
    # X_train_new, X_test_new, \
    # featureName, feature_freq, feature_coef, index = lasso.lasso_shuffle(shuffle_time=100,
    #                                                                      alpha_range=np.arange(0.01, 0.1, 0.01))

    # 使用elastic net进行特征选择
    # elas = elastic_net(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
    #                    feature_name=DTIName, path=crohnPath, cv_val=False)
    #
    # X_train_new, X_test_new, featureName, _, _, index = elas.elasticnet_shuffle(shuffle_time=100,
    #                                                                             l1_range=1.0,
    #                                                                             alphas_range=0.1)

    index = [106, 21, 22, 23, 419, 597, 614, 654, 266]
    featureName = DTIName[index]

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
                                                                                   'gamma': 0.1,
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
"""
if __name__ == '__main__':
    # 设定结果存储路径并创建文件夹
    timeDir = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    crohnPath = os.path.join('crohn', 'fc', timeDir)
    os.makedirs(crohnPath, exist_ok=True)

    # 读取数据

    ###############
    #####DTI数据####
    ###############
    '''
    dataFA = pd.read_csv('./DTI_RSFC_feature/DTI/FA_feature_3mm.csv', sep=',', engine='python')
    dataMD = pd.read_csv('./DTI_RSFC_feature/DTI/MD_feature_3mm.csv', sep=',', engine='python')
    dataRD = pd.read_csv('./DTI_RSFC_feature/DTI/RD_feature_3mm.csv', sep=',', engine='python')
    dataL1 = pd.read_csv('./DTI_RSFC_feature/DTI/L1_feature_3mm.csv', sep=',', engine='python')
    dataL2 = pd.read_csv('./DTI_RSFC_feature/DTI/L2_feature_3mm.csv', sep=',', engine='python')
    dataL3 = pd.read_csv('./DTI_RSFC_feature/DTI/L3_feature_3mm.csv', sep=',', engine='python')

    # 读取label
    label = pd.read_csv('./DTI_RSFC_feature/label.csv', sep=',', engine='python', header=None)

    # 数据统一转换为numpy格式
    faNumpy = np.array(dataFA.iloc[:, :])
    mdNumpy = np.array(dataMD.iloc[:, :])
    rdNumpy = np.array(dataRD.iloc[:, :])
    l1Numpy = np.array(dataL1.iloc[:, :])
    l2Numpy = np.array(dataL2.iloc[:, :])
    l3Numpy = np.array(dataL3.iloc[:, :])
    # 标签转换为numpy格式
    y = np.array(label.iloc[:, :]).ravel()
    # 规定病人为0，健康人为1
    labelName = ['0', '1']
    posLabel = 1

    # 存储特征名称
    faName = dataFA.columns._values[:]
    mdName = dataMD.columns._values[:]
    rdName = dataRD.columns._values[:]
    l1Name = dataL1.columns._values[:]
    l2Name = dataL2.columns._values[:]
    l3Name = dataL3.columns._values[:]

    # 对结构进行拼接
    DTI = np.concatenate((faNumpy, mdNumpy, rdNumpy, l1Numpy, l2Numpy, l3Numpy), axis=1)
    DTIName = np.concatenate((faName, mdName, rdName, l1Name, l2Name, l3Name), axis=0)
    # 对结构数据进行z标准化
    nor_DTI = normalization(DTI)
    DTI = nor_DTI.z_score()
    # 结构数据的特征数量
    DTI_num = len(DTIName)

    # X为最终的输入
    X = l3Numpy
    DTIName = l3Name
    '''

    ###############
    ####功能数据####
    ###############
    # 读取数据
    dataFalff = pd.read_csv('./DTI_RSFC_feature/RSFC/fALFF_feature.csv', sep=',', engine='python')
    dataFc = pd.read_csv('./DTI_RSFC_feature/RSFC/FC_feature.csv', sep=',', engine='python')
    dataReho = pd.read_csv('./DTI_RSFC_feature/RSFC/ReHo_feature.csv', sep=',', engine='python')

    # 读取label
    label = pd.read_csv('./DTI_RSFC_feature/label.csv', sep=',', engine='python', header=None)

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

    # ttest = Ttest(fcNumpy, fcName,
    #               dis_num=65, hc_num=40, thres=0.05, all_value=False)
    utest = Utest(fcNumpy, fcName,
                  dis_num=65, hc_num=40, thres=0.01, all_value=False, stat_path=crohnPath)

    selectFeature, selectFeatureName, _, _, uTestIndex = utest.utest_loo_freq(K_value=40, choice='percent')

    # 对数据进行标准化并拼接
    # 拼接功能数据
    fMRI = np.concatenate((falffNumpy, selectFeature, rehoNumpy), axis=1)
    fMRIName = np.concatenate((falffName, selectFeatureName, rehoName), axis=0)
    # 对功能数据进行z标准化
    nor_fMRI = normalization(fMRI)
    fMRI = nor_fMRI.z_score()
    # 功能数据的特征数量
    fMRI_num = len(fMRIName)

    # X为最终的输入
    X = fcNumpy
    fMRIName = fcName


    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

    # lasso = lasso(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
    #               feature_name=DTIName, path=crohnPath, cv_val=True)
    #
    # X_train_new, X_test_new, \
    # featureName, feature_freq, feature_coef, index = lasso.lasso_shuffle(shuffle_time=100,
    #                                                                      alpha_range=np.arange(0.01, 0.1, 0.01))

    # 使用elastic net进行特征选择
    elas = elastic_net(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                       feature_name=fMRIName, path=crohnPath, cv_val=False)

    X_train_new, X_test_new, featureName, _, _, index = elas.elasticnet_shuffle(shuffle_time=100,
                                                                                l1_range=1.0,
                                                                                alphas_range=0.014)

    # index = [106, 21, 22, 23, 419, 597, 614, 654, 266]
    featureName = fMRIName[index]

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
"""

#############################################################
# 使用svm_shuffle返回结果并且绘制ROC曲线的示意图(为crohn的静息态数据为例)
#############################################################
'''
if __name__ == '__main__':
    # 读取数据
    dataFalff = pd.read_csv('./all_feature/ALFF/mfALFF_G_90.csv', sep=',', engine='python')
    dataFc = pd.read_csv('./all_feature/FC/FC_feature_90_R_G.csv', sep=',', engine='python')
    dataReho = pd.read_csv('./all_feature/ReHo/smReHo_G_90.csv', sep=',', engine='python')

    # 读取label
    label = pd.read_csv('./DTI_RSFC_feature/label.csv', sep=',', engine='python', header=None)

    # 数据统一转换为numpy格式
    falffNumpy = np.array(dataFalff.iloc[:, :])
    fcNumpy = np.array(dataFc.iloc[:, :])
    rehoNumpy = np.array(dataReho.iloc[:, :])

    # 标签转换为numpy格式
    y = np.array(label.iloc[:, :])
    y = y.ravel()
    # 规定病人为0，健康人为1
    labelName = ['0', '1']
    posLabel = 1

    # 存储特征名称
    falffName = dataFalff.columns._values
    fcName = dataFc.columns._values
    rehoName = dataReho.columns._values

    from stat_model import Ttest
    from stat_model import Utest
    # ttest = Ttest(fcNumpy, fcName,
    #               dis_num=65, hc_num=40, thres=0.05, all_value=False)
    utest = Utest(fcNumpy, fcName, stat_path='./',
                  dis_num=65, hc_num=40, thres=0.01, all_value=False)

    # select_feature, select_feature_name, select_pvalue, select_tvalue = ttest.ttest_only()
    select_feature, select_feature_name, \
    select_feature_name_freq, select_pvalue, index = utest.utest_loo_freq(K_value=40, choice='percent')
    print('--------------feature-----------------')
    print(select_feature)
    print('--------------feature_name-------------')
    print(select_feature_name)
    print('--------------feature p value--------------')
    print(select_pvalue)

    # 对数据进行拼接
    # 拼接功能数据
    fMRI = np.concatenate((falffNumpy, fcNumpy, rehoNumpy), axis=1)
    fMRIName = np.concatenate((falffName, fcName, rehoName), axis=0)

    # fMRIName = fMRIName.tolist()
    # list = ['alff_0', 'alff_60', 'fc_1504', 'fc_1950', 'fc_2154', 'fc_2186', 'fc_2448', 'fc_2520', 'fc_2694',
    #         'fc_29', 'fc_2914', 'fc_3278', 'fc_3376', 'fc_3695', 'fc_3900', 'fc_482', 'fc_62', 'reho_23']
    #
    # indexList = []
    # for i in list:
    #     indexOLd = fMRIName.index(i)
    #     indexList.append(indexOLd)
    #     print(indexOLd)
    # print(indexList)

    index1 = [0, 60, 1594, 2040, 2244, 2276, 2538, 2610, 2784, 119, 3004, 3368, 3466, 3785, 3990, 572, 152, 4118]
    print(index1)

    # 对数据进行z标准化
    nor = normalization(fMRI)
    X = nor.z_score()

    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y)

    # from ml_model import lasso
    #
    # lasso = lasso(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
    #               feature_name=fMRIName, path='crohn/fmri/', cv_val=True)
    #
    # X_train_new, X_test_new, \
    # feature_name, feature_freq, feature_coef, index = lasso.lasso_shuffle(shuffle_time=100,
    #                                                                       alpha_range=np.arange(0.1, 0.5, 0.01))
    # print(feature_name)
    # print('------------------------')
    # print(feature_freq)
    # print('------------------------')
    # print(feature_coef)
    # print('------------------------')
    # print(index)

    X_new = X[:, index1]

    # 使用svm_shuffle返回结果并且绘制ROC曲线的示意图
    from ml_model import SVM
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svm = SVM(X_new, y, path='svm_result')
    train_means, train_std, test_means, test_std, roc_dict = svm.svm_shuffle(outer=10,
                                                                   para={'kernel': 'linear', 'C': 10},
                                                                   svm_metrics=['accuracy', 'sensitivity', 'specificity'])
    # train_means, train_std, test_means, test_std, roc_dict = svm.svm_shuffle(X, y, outer=10,
    #                                                                          para={'kernel': 'rbf',
    #                                                                                'C': 10,
    #                                                                                'gamma': 0.1},
    #                                                                          svm_metrics=['accuracy'])

    # train_means, train_std, test_means, test_std, roc_dict = svm.svm_nested(X, y, outer=10, shuffle_time=5,
    #                                                                          para=para,
    #                                                                          svm_metrics=['accuracy', 'precision'])

    x_axis = np.arange(1, 101)
    fig, axe = plt.subplots()
    axe.fill_between(x_axis,
                     np.array(test_means['accuracy']) - np.array(test_std['accuracy']),
                     np.array(test_means['accuracy']) + np.array(test_std['accuracy']),
                     alpha=0.2)
    axe.plot(x_axis, np.array(test_means['accuracy']), '--', color='g',
             alpha=1)

    plt.show()

    std_auc_train = np.std(roc_dict['auc_train'])
    std_auc_test = np.std(roc_dict['auc_test'])

    plt.plot(roc_dict['fpr_train'], roc_dict['tpr_train'], color='#0a4099',
             label=r'Train ROC (AUC = %0.2f)' % roc_dict['auc_train'],
             lw=2, alpha=.8)
    plt.plot(roc_dict['fpr_test'], roc_dict['tpr_test'], color='#b07c0e',
             label=r'Test ROC (AUC = %0.2f)' % roc_dict['auc_test'],
             lw=2, alpha=.8)

    std_tpr_train = np.std(roc_dict['tpr_list_train'], axis=0)
    std_tpr_test = np.std(roc_dict['tpr_list_test'], axis=0)

    tprs_upper = np.minimum(roc_dict['tpr_test'] + std_tpr_test, 1)
    tprs_lower = np.maximum(roc_dict['tpr_test'] - std_tpr_test, 0)

    plt.fill_between(roc_dict['fpr_test'], tprs_lower, tprs_upper, color='grey', alpha=.2)

    tprs_upper = np.minimum(roc_dict['tpr_test'] + std_tpr_train, 1)
    tprs_lower = np.maximum(roc_dict['tpr_test'] - std_tpr_train, 0)

    plt.fill_between(roc_dict['fpr_train'], tprs_lower, tprs_upper, color='grey', alpha=.2)

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    # plt.title('Cross-Validation ROC of SVM', fontsize=18)
    plt.legend(loc="lower right", prop={'size': 15})
    plt.show()
'''



###########################
######alff collection######
###########################
'''
if __name__ == '__main__':
    # 设定结果存储路径并创建文件夹
    timeDir = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    crohnPath = os.path.join('crohn', 'alff_results', timeDir)
    os.makedirs(crohnPath, exist_ok=True)

    # 读取数据
    dataAlff = pd.read_csv('./ROI_feature_multi_ALFF/mfALFF_classic_global.csv', sep=',', engine='python')

    # 读取label
    label = pd.read_csv('./DTI_RSFC_feature/label.csv', sep=',', engine='python', header=None)

    # 数据统一转换为numpy格式
    alffNumpy = np.array(dataAlff.iloc[:, :])

    # 标签转换为numpy格式
    y = np.array(label.iloc[:, :]).ravel()
    # 规定病人为0，健康人为1
    labelName = ['0', '1']
    posLabel = 1

    # 存储特征名称
    alffName = dataAlff.columns._values

    # 单独对FC使用留一法U检验进行特征选择

    # ttest = Ttest(fcNumpy, fcName,
    #               dis_num=65, hc_num=40, thres=0.05, all_value=False)
    # utest = Utest(fcNumpy, fcName,
    #               dis_num=65, hc_num=40, thres=0.01, all_value=False, stat_path=crohnPath)

    # selectFeature, selectFeatureName, _, _, uTestIndex = utest.utest_loo_freq(K_value=40, choice='percent')

    # 对功能数据进行z标准化
    nor_fMRI = normalization(alffNumpy)
    fMRI = nor_fMRI.z_score()
    # 功能数据的特征数量
    fMRI_num = int(len(alffName))

    # X为最终的输入
    X = fMRI

    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

    # lasso = lasso(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
    #               feature_name=alffName, path=crohnPath, cv_val=True)
    #
    # X_train_new, X_test_new, \
    # featureName, feature_freq, feature_coef, index = lasso.lasso_shuffle(shuffle_time=100,
    #                                                                      alpha_range=np.arange(0.01, 0.1, 0.01))

    # 使用elastic net进行特征选择
    # elas = elastic_net(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
    #                    feature_name=multimodalName, path=crohnPath, cv_val=False)
    #
    # X_train_new, X_test_new, featureName, _, _, index = elas.elasticnet_shuffle(shuffle_time=100,
    #                                                                             l1_range=1.0,
    #                                                                             alphas_range=0.05)

    index = [18, 55, 45, 30, 62, 83, 85, 0, 69, 10, 63, 64, 84]
    featureName = alffName[index]

    # 不进行模态区分，为了对比试验和特征相关性矩阵
    X_new = X[:, index]

    # 使用svm_shuffle返回结果并且绘制ROC曲线的示意图
    svmShufflePath = os.path.join(crohnPath, 'svm_shuffle')
    os.makedirs(svmShufflePath, exist_ok=True)
    # 设置参数
    para = {'kernel': ['linear'],
            'C': np.arange(1, 100, 10)}
    svmTime = 100
    svm = SVM(X_new, y, path=crohnPath)
    train_means, train_std, test_means, test_std, roc_dict = svm.svm_shuffle(outer=10, shuffle_time=svmTime,
                                                                             para={'kernel': 'linear',
                                                                                   'C': 10  },
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


###########################
#############pe############
###########################
'''
if __name__ == '__main__':
    # 设定结果存储路径并创建文件夹
    timeDir = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    pePath = os.path.join('pe', timeDir, 'fmri')
    os.makedirs(pePath, exist_ok=True)

    # 读取数据
    dataAlff = pd.read_csv('./pe_feature/DATA_Recog/data_ALFF.csv', sep=',', engine='python')
    dataFalff = pd.read_csv('./pe_feature/DATA_Recog/data_fALFF.csv', sep=',', engine='python')
    dataDc = pd.read_csv('./pe_feature/DATA_Recog/data_DC.csv', sep=',', engine='python')
    dataVmhcGlobal = pd.read_csv('./pe_feature/DATA_Recog/data_global_VMHC.csv', sep=',', engine='python')
    dataVmhc = pd.read_csv('./pe_feature/DATA_Recog/data_NonGlobal_VMHC.csv', sep=',', engine='python')
    dataReho = pd.read_csv('./pe_feature/DATA_Recog/data_ReHo.csv', sep=',', engine='python')
    dataVbm = pd.read_csv('./pe_feature/DATA_Recog/data_VBM.csv', sep=',', engine='python')

    # 读取label
    label = pd.read_csv('./pe_feature/DATA_Recog/label.csv', sep=',', engine='python', header=None)

    # 数据统一转换为numpy格式
    alffNumpy = np.array(dataAlff.iloc[:, :])
    falffNumpy = np.array(dataFalff.iloc[:, :])
    dcNumpy = np.array(dataDc.iloc[:, :])
    vmhcGlobalNumpy = np.array(dataVmhcGlobal.iloc[:, :])
    vmhcNumpy = np.array(dataVmhc.iloc[:, :])
    rehoNumpy = np.array(dataReho.iloc[:, :])
    vbmNumpy = np.array(dataVbm.iloc[:, :])

    # 标签转换为numpy格式
    y = np.array(label.iloc[:, :]).ravel()
    # 规定病人为0，健康人为1
    labelName = ['0', '1']
    posLabel = 1

    # 存储特征名称
    alffName = dataAlff.columns._values
    falffName = dataFalff.columns._values
    dcName = dataDc.columns._values
    vmhcGlobalName = dataVmhcGlobal.columns._values
    vmhcName = dataVmhc.columns._values
    rehoName = dataReho.columns._values
    vbmName = dataVbm.columns._values

    # 拼接功能数据
    fMRI = np.concatenate((alffNumpy, falffNumpy, dcNumpy, vmhcGlobalNumpy, vmhcNumpy, rehoNumpy, vbmNumpy), axis=1)
    fMRIName = np.concatenate((alffName, falffName, dcName, vmhcGlobalName, vmhcName, rehoName, vbmName), axis=0)
    # 对功能数据进行z标准化
    nor_fMRI = normalization(fMRI)
    fMRI = nor_fMRI.z_score()
    # 功能数据的特征数量
    fMRI_num = len(fMRIName)

    X = fMRI

    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y)

    lasso = lasso(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                  feature_name=fMRIName, path=pePath, cv_val=True)

    X_train_new, X_test_new, \
    feature_name, feature_freq, feature_coef, index = lasso.lasso_shuffle(shuffle_time=100,
                                                                          alpha_range=np.arange(0.1, 0.15, 0.01))

    X_new = X[:, index]

    # 使用svm_shuffle返回结果并且绘制ROC曲线的示意图
    svmShufflePath = os.path.join(pePath, 'svm_shuffle')
    os.makedirs(svmShufflePath, exist_ok=True)
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_new, y, path=pePath)
    train_means, train_std, test_means, test_std, roc_dict = svm.svm_nested(outer=5, shuffle_time=svmTime,
                                                                             para=para,
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
    corr_mat = corr(X_new, feature_name, pePath)
    corr_mat.corr_heatmap()
'''


###########################
#############mci###########
###########################
'''
if __name__ == '__main__':
    # 设定结果存储路径并创建文件夹
    timeDir = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    pePath = os.path.join('mci', timeDir, 'alff')
    os.makedirs(pePath, exist_ok=True)

    # 读取数据
    dataAlff = pd.read_csv('./MCI/ALFF_90_MCI_.csv', sep=',', engine='python')

    # 读取label
    label = pd.read_csv('./MCI/label.csv', sep=',', engine='python', header=None)

    # 数据统一转换为numpy格式
    alffNumpy = np.array(dataAlff.iloc[:, :116])

    # 标签转换为numpy格式
    y = np.array(label.iloc[:, :]).ravel()
    # 规定病人为0，健康人为1
    labelName = ['0', '1']
    posLabel = 1

    # 存储特征名称
    alffName = dataAlff.columns._values
    # 拼接功能数据
    fMRI = alffNumpy
    fMRIName = alffName[:116]
    # 对功能数据进行z标准化
    nor_fMRI = normalization(fMRI)
    fMRI = nor_fMRI.z_score()
    # 功能数据的特征数量
    fMRI_num = len(fMRIName)

    X = fMRI

    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y)

    lasso = lasso(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                  feature_name=fMRIName, path=pePath, cv_val=True)

    X_train_new, X_test_new, \
    feature_name, feature_freq, feature_coef, index = lasso.lasso_shuffle(shuffle_time=100,
                                                                          alpha_range=np.arange(0.05, 0.1, 0.01))

    X_new = X[:, index]

    # 使用svm_shuffle返回结果并且绘制ROC曲线的示意图
    svmShufflePath = os.path.join(pePath, 'svm_shuffle')
    os.makedirs(svmShufflePath, exist_ok=True)
    # 设置参数
    para = {'kernel': ['rbf'],
            'gamma': np.arange(0.1, 1, 0.1),
            'C': np.arange(1, 100, 1)}
    svmTime = 100
    svm = SVM(X_new, y, path=pePath)
    train_means, train_std, test_means, test_std, roc_dict = svm.svm_nested(outer=5, shuffle_time=svmTime,
                                                                             para=para,
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
    corr_mat = corr(X_new, feature_name, pePath)
    corr_mat.corr_heatmap()
'''
