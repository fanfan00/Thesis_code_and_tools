#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/31 9:57
# @Author  : fanyalei
# @File    : Drawing_ROC.py
import matplotlib.pyplot as plt
#from keras.utils import to_categorical
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import numpy as np
def main_SVM_ROC():
    file_SVM_label = 'E:/Document/myporjects/Thesis2_allprojects_and_dataset/Drawing_data/SVM/SVM_label.txt'
    file_SVM_pred_0 = 'E:/Document/myporjects/Thesis2_allprojects_and_dataset/Drawing_data/SVM/SVM_pred_0.txt'
    file_SVM_pred_1 = 'E:/Document/myporjects/Thesis2_allprojects_and_dataset/Drawing_data/SVM/SVM_pred_1.txt'
    file_SVM_pred_10= 'E:/Document/myporjects/Thesis2_allprojects_and_dataset/Drawing_data/SVM/SVM_pred_10.txt'

    SVM_label = np.loadtxt(file_SVM_label)
    SVM_pred_1 = np.loadtxt(file_SVM_pred_1)
    SVM_pred_10 =np.loadtxt(file_SVM_pred_10)

    # y真实prob预测
    fpr, tpr, threshold = roc_curve(SVM_label, SVM_pred_1)  ###计算真阳性率和假阳性率
    roc_auc = auc(fpr, tpr)  ###计算auc的值

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC-SVM')
    plt.legend(loc="lower right")

    plt.show()

def main_Inception_ROC():
    file_SVM_label = 'E:/Document/myporjects/Thesis2_allprojects_and_dataset/Drawing_data/Inception/inception_label.txt'

    file_SVM_pred_1 = 'E:/Document/myporjects/Thesis2_allprojects_and_dataset/Drawing_data/Inception/inception_pred_1.txt'


    SVM_label = np.loadtxt(file_SVM_label)
    SVM_pred_1 = np.loadtxt(file_SVM_pred_1)


    # y真实prob预测
    fpr, tpr, threshold = roc_curve(SVM_label, SVM_pred_1)  ###计算真阳性率和假阳性率
    roc_auc = auc(fpr, tpr)  ###计算auc的值

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC-InceptionV3')
    plt.legend(loc="lower right")

    plt.show()

if __name__ == "__main__":
    main_SVM_ROC()
    #main_Inception_ROC()

