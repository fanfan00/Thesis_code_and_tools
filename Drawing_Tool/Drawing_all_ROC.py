#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/4 10:47
# @Author  : fanyalei
# @File    : fan_huitu.py

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import numpy as np
from matplotlib.pyplot import MultipleLocator

def huitu_ROC_all(f_SVM_label ,f_SVM_pred ,
                  f_Inception_fix_label,f_Inception_fix_pred,
                  f_Inception_ft_label,f_Inception_ft_pred,
                  f_VGG_label,f_VGG_pred,
                  f_Res50_label,f_Res50_pred,
                  f_Res101_label,f_Res101_pred):
    SVM_label = np.loadtxt(f_SVM_label)
    SVM_pred = np.loadtxt(f_SVM_pred)

    Inception_fix_label = np.loadtxt(f_Inception_fix_label)
    Inception_fix_pred = np.loadtxt(f_Inception_fix_pred)

    Inception_ft_label = np.loadtxt(f_Inception_ft_label)
    Inception_ft_pred = np.loadtxt(f_Inception_ft_pred)

    VGG_label = np.loadtxt(f_VGG_label)
    VGG_pred = np.loadtxt(f_VGG_pred)

    Res50_label = np.loadtxt(f_Res50_label)
    Res50_pred = np.loadtxt(f_Res50_pred)

    Res101_label = np.loadtxt(f_Res101_label)
    Res101_pred = np.loadtxt(f_Res101_pred)

    #---------------------------------------------------------------------------------------
    # y真实prob预测
    fpr_1, tpr_1, threshold_1 = roc_curve(SVM_label, SVM_pred)  ###计算真阳性率和假阳性率
    roc_auc_1 = auc(fpr_1, tpr_1)  ###计算auc的值

    fpr_2, tpr_2, threshold_2 = roc_curve(Inception_fix_label, Inception_fix_pred)
    roc_auc_2 = auc(fpr_2, tpr_2)

    fpr_3, tpr_3, threshold_3 = roc_curve(Inception_ft_label, Inception_ft_pred)
    roc_auc_3 = auc(fpr_3, tpr_3)

    fpr_4, tpr_4, threshold_4 = roc_curve(VGG_label, VGG_pred)
    roc_auc_4 = auc(fpr_4, tpr_4)

    fpr_5, tpr_5, threshold_5 = roc_curve(Res50_label, Res50_pred)
    roc_auc_5 = auc(fpr_5, tpr_5)

    fpr_6, tpr_6, threshold_6 = roc_curve(Res101_label, Res101_pred)
    roc_auc_6 = auc(fpr_6, tpr_6)
    #----------------------------------------------------------------------------------------

    lw = 2
    plt.figure(figsize=(10, 10))

    plt.plot(fpr_1, tpr_1, color='r', lw=lw,label='SVM curve (area = %0.3f)' % roc_auc_1)
    plt.plot(fpr_2, tpr_2, color='b', lw=lw,label='InceptionV3-fix curve (area = %0.3f)' % roc_auc_2)
    plt.plot(fpr_3, tpr_3, color='g', lw=lw, label='InceptionV3-fine tuning curve (area = %0.3f)' % roc_auc_3)
    plt.plot(fpr_4, tpr_4, color='darkorange', lw=lw, label='Faster-RCNN VGG16 curve (area = %0.3f)' % roc_auc_4)
    plt.plot(fpr_5, tpr_5, color='m', lw=lw, label='Faster-RCNN ResNet50 curve (area = %0.3f)' % roc_auc_5)
    plt.plot(fpr_6, tpr_6, color='y', lw=lw, label='Faster-RCNN ResNet101 (area = %0.3f)' % roc_auc_6)


    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate(sensitivity)')
    plt.ylabel('True Positive Rate(specificity)')
    plt.title('ROC curve')
    #-------------------------------------------------------------------------------------------
    x_major_locator = MultipleLocator(0.1)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(0.1)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)

    plt.xlim(0, 1)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(0, 1.05)
    # 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白

    plt.legend(loc="lower right")

    plt.grid(linewidth=0.75)
    #--------------------------------------------------------------------------------------------
    plt.show()

if __name__ == '__main__':
    f_SVM_label ='E:/Document/myporjects/Thesis2_allprojects_and_dataset/Drawing_data/SVM/SVM_label.txt'
    f_SVM_pred = 'E:/Document/myporjects/Thesis2_allprojects_and_dataset/Drawing_data/SVM/SVM_pred_1.txt'


    f_Inception_fix_label ='E:/Document/myporjects/Thesis2_allprojects_and_dataset/Drawing_data/Inception/inception_label.txt'
    f_Inception_fix_pred ='E:/Document/myporjects/Thesis2_allprojects_and_dataset/Drawing_data/Inception/inception_pred_1.txt'

    f_Inception_ft_label='E:/Document/myporjects/Thesis2_allprojects_and_dataset/Drawing_data/Inception_FT/inception_finetuning_label.txt'
    f_Inception_ft_pred='E:/Document/myporjects/Thesis2_allprojects_and_dataset/Drawing_data/Inception_FT/inception_finetuning_pred_1.txt'

    f_VGG_label='E:/Document/myporjects/Thesis2_allprojects_and_dataset/Drawing_data/VGG16/VGG16_label.txt'
    f_VGG_pred='E:/Document/myporjects/Thesis2_allprojects_and_dataset/Drawing_data/VGG16/VGG16_pred_1.txt'

    f_Res50_label='E:/Document/myporjects/Thesis2_allprojects_and_dataset/Drawing_data/ResNet/res50/ResNet50_label.txt'
    f_Res50_pred='E:/Document/myporjects/Thesis2_allprojects_and_dataset/Drawing_data/ResNet/res50/ResNet50_pred_1.txt'

    f_Res101_label='E:/Document/myporjects/Thesis2_allprojects_and_dataset/Drawing_data/ResNet/res101/ResNet101_label.txt'
    f_Res101_pred='E:/Document/myporjects/Thesis2_allprojects_and_dataset/Drawing_data/ResNet/res101/ResNet101_pred_1.txt'

    huitu_ROC_all(f_SVM_label, f_SVM_pred,
                  f_Inception_fix_label, f_Inception_fix_pred,
                  f_Inception_ft_label, f_Inception_ft_pred,
                  f_VGG_label, f_VGG_pred,
                  f_Res50_label, f_Res50_pred,
                  f_Res101_label, f_Res101_pred)