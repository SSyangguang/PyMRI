#!/usr/bin/python
# -*- coding: UTF-8 -*-
BrainRegion = {0: 'Precentral_L',
1: 'Precentral_R',
2: 'Frontal_Sup_L',
3: 'Frontal_Sup_R',
4: 'Frontal_Sup_Orb_L',
5: 'Frontal_Sup_Orb_R',
6: 'Frontal_Mid_L',
7: 'Frontal_Mid_R',
8: 'Frontal_Mid_Orb_L',
9: 'Frontal_Mid_Orb_R',
10: 'Frontal_Inf_Oper_L',
11: 'Frontal_Inf_Oper_R',
12: 'Frontal_Inf_Tri_L',
13: 'Frontal_Inf_Tri_R',
14: 'Frontal_Inf_Orb_L',
15: 'Frontal_Inf_Orb_R',
16: 'Rolandic_Oper_L',
17: 'Rolandic_Oper_R',
18: 'Supp_Motor_Area_L',
19: 'Supp_Motor_Area_R',
20: 'Olfactory_L',
21: 'Olfactory_R',
22: 'Frontal_Sup_Medial_L',
23: 'Frontal_Sup_Medial_R',
24: 'Frontal_Mid_Orb_L',
25: 'Frontal_Mid_Orb_R',
26: 'Rectus_L',
27: 'Rectus_R',
28: 'Insula_L',
29: 'Insula_R',
30: 'Cingulum_Ant_L',
31: 'Cingulum_Ant_R',
32: 'Cingulum_Mid_L',
33: 'Cingulum_Mid_R',
34: 'Cingulum_Post_L',
35: 'Cingulum_Post_R',
36: 'Hippocampus_L',
37: 'Hippocampus_R',
38: 'ParaHippocampal_L',
39: 'ParaHippocampal_R',
40: 'Amygdala_L',
41: 'Amygdala_R',
42: 'Calcarine_L',
43: 'Calcarine_R',
44: 'Cuneus_L',
45: 'Cuneus_R',
46: 'Lingual_L',
47: 'Lingual_R',
48: 'Occipital_Sup_L',
49: 'Occipital_Sup_R',
50: 'Occipital_Mid_L',
51: 'Occipital_Mid_R',
52: 'Occipital_Inf_L',
53: 'Occipital_Inf_R',
54: 'Fusiform_L',
55: 'Fusiform_R',
56: 'Postcentral_L',
57: 'Postcentral_R',
58: 'Parietal_Sup_L',
59: 'Parietal_Sup_R',
60: 'Parietal_Inf_L',
61: 'Parietal_Inf_R',
62: 'SupraMarginal_L',
63: 'SupraMarginal_R',
64: 'Angular_L',
65: 'Angular_R',
66: 'Precuneus_L',
67: 'Precuneus_R',
68: 'Paracentral_Lobule_L',
69: 'Paracentral_Lobule_R',
70: 'Caudate_L',
71: 'Caudate_R',
72: 'Putamen_L',
73: 'Putamen_R',
74: 'Pallidum_L',
75: 'Pallidum_R',
76: 'Thalamus_L',
77: 'Thalamus_R',
78: 'Heschl_L',
79: 'Heschl_R',
80: 'Temporal_Sup_L',
81: 'Temporal_Sup_R',
82: 'Temporal_Pole_Sup_L',
83: 'Temporal_Pole_Sup_R',
84: 'Temporal_Mid_L',
85: 'Temporal_Mid_R',
86: 'Temporal_Pole_Mid_L',
87: 'Temporal_Pole_Mid_R',
88: 'Temporal_Inf_L',
89: 'Temporal_Inf_R',
90: 'Cerebellum',
91: 'Cerebellum',
92: 'Cerebellum',
93: 'Cerebellum',
94: 'Cerebellum',
95: 'Cerebellum',
96: 'Cerebellum',
97: 'Cerebellum',
98: 'Cerebellum',
99: 'Cerebellum',
100: 'Cerebellum',
101: 'Cerebellum',
102: 'Cerebellum',
103: 'Cerebellum',
104: 'Cerebellum',
105: 'Cerebellum',
106: 'Cerebellum',
107: 'Cerebellum',
108: 'Vermis',
109: 'Vermis',
110: 'Vermis',
111: 'Vermis',
112: 'Vermis',
113: 'Vermis',
114: 'Vermis',
115: 'Vermis',
}
FC = dict()
fn = dict()
length = dict()
fa_matrix = dict()
feature_txt = []

def Print_FC_Region(X,FC):
    # Use a breakpoint in the code line below to debug your script.

    for i in range(1, 91):
        for j in range(1, i):
            if sum(range(1, i-1)) + j == X:
                FC[sum(range(1, i-1)) + j] = (BrainRegion[i-1], BrainRegion[j-1])
    return FC

    # 删除两边字符
    # print(X.strip())



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # # a = ['Alff_0', 'Alff_34', 'Alff_49', 'Alff_52', 'Alff_6', 'Alff_83', 'Alff_15', 'Alff_17', 'Alff_35', 'Alff_37', 'Alff_55', 'Alff_63', 'Alff_64', 'Alff_73', 'Alff_87', 'Alff_25', 'Alff_62', 'Alff_84', 'Alff_59', 'Alff_66', 'Alff_45']
    # feature_list = input("输入你的特征结果： ")
    # feature_txt = feature_list.split(",")
    # for i in range(0, len(feature_txt)):
    #     feature_txt[i] = feature_txt[i][1:-1]
    feature_txt = ['ad_2', 'ad_40', 'ad_6', 'ad_82', 'alff_83', 'fa_matrix_1049', 'fa_matrix_1550', 'fa_matrix_171', 'fa_matrix_2258', 'fa_matrix_360', 'fc_1119', 'fc_144', 'fc_1444', 'fc_145', 'fc_1504', 'fc_1669', 'fc_1796', 'fc_1950', 'fc_196', 'fc_2154', 'fn_1554', 'fn_18', 'fn_2887', 'fn_3075', 'fn_3412', 'fn_3701', 'fn_3866', 'fn_411', 'fn_43', 'length_1049', 'length_2887', 'length_3540', 'length_360', 'length_3788', 'md_40', 'md_41', 'rd_40', 'rd_75']


    fa = dict()
    ad = dict()   # 这就是lamda1'ad_40', 'ad_45', 'alff_0', 'fa_matrix_1049', 'fa_matrix_1550', 'fa_matrix_2183', 'fa_matrix_2570', 'fa_matrix_266', 'fa_matrix_2830', 'fa_matrix_3359', 'fa_matrix_3442', 'fa_matrix_360', 'fc_1073', 'fc_1119', 'fc_144', 'fc_1444', 'fc_1498', 'fc_1504', 'fc_1669', 'fc_1670', 'fc_1786', 'fc_1787', 'fc_1796', 'fc_1929', 'fc_1950', 'fc_2185', 'fc_2186', 'fn_154', 'fn_2504', 'fn_2927', 'fn_3866', 'fn_43', 'length_1049', 'length_126', 'length_1550', 'length_171', 'length_2310', 'length_3540', 'length_360', 'md_74', 'md_83', 'rd_74', 'rd_75', 'reho_18', 'reho_55', 'reho_63', 'reho_82'
    lamda3 = dict()
    md = dict()
    rd = dict()
    alff = dict()
    falff = dict()
    reho = dict()


    for i in range(len(feature_txt)):
        if len(feature_txt[i]) <= 7 and feature_txt[i].find("fa_") == 0:
            fa[int(feature_txt[i][3:])] = BrainRegion[int(feature_txt[i][3:])]
        if feature_txt[i].find("ad_") == 0:
            ad[int(feature_txt[i][3:])+1] = BrainRegion[int(feature_txt[i][3:])]
        if feature_txt[i].find("L3_") == 0:
            lamda3[int(feature_txt[i][3:])+1] = BrainRegion[int(feature_txt[i][3:])]
        if feature_txt[i].find("md_") == 0:
            md[int(feature_txt[i][3:])] = BrainRegion[int(feature_txt[i][3:])]
        if feature_txt[i].find("rd_") == 0:
            rd[int(feature_txt[i][3:])] = BrainRegion[int(feature_txt[i][3:])]

        if feature_txt[i].find("alff_") == 0:
            alff[int(feature_txt[i][5:])] = BrainRegion[int(feature_txt[i][5:])]
        if feature_txt[i].find("reho_") == 0:
            reho[int(feature_txt[i][5:])] = BrainRegion[int(feature_txt[i][5:])]

        if feature_txt[i].find("fc_") == 0:
            Print_FC_Region(int(feature_txt[i][3:]), FC)
        if feature_txt[i].find("fa_matrix_") == 0:
            Print_FC_Region(int(feature_txt[i][10:]), fa_matrix)
        if feature_txt[i].find("fn_") == 0:
            Print_FC_Region(int(feature_txt[i][3:]), fn)
        if feature_txt[i].find("length_") == 0:
            Print_FC_Region(int(feature_txt[i][7:]), length)

    print("\n", "FA region:")
    for k in fa.keys():
        print(k, fa[k])
    print("\n", "AD region:")
    for k in ad.keys():
        print(k, ad[k])
    print("\n", "Lamda_3 region:")
    for k in lamda3.keys():
        print(k, lamda3[k])
    print("\n", "MD region:")
    for k in md.keys():
        print(k, md[k])
    print("\n", "RD region:")
    for k in rd.keys():
        print(k, rd[k])
    print("\n", "ALFF region:")
    for k in alff.keys():
        print(k, alff[k])
    print("\n", "ReHo region:")
    for k in reho.keys():
        print(k, reho[k])
    print("\n", "FC region:")
    for k in FC.keys():
        print(k, FC[k][0], "--", FC[k][1])
    print("\n", "fa matrix region:")
    for k in fa_matrix.keys():
        print(k, fa_matrix[k][0], "--", fa_matrix[k][1])
    print("\n", "fn:")
    for k in fn.keys():
        print(k, fn[k][0], "--", fn[k][1])
    print("\n", "length matrix:")
    for k in length.keys():
        print(k, length[k][0], "--", length[k][1])


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
