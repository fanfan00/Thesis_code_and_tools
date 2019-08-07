# 写一个程序，用来处理fasterRCNN-RES生成的“ det_foreignobj.txt ”

import os
import numpy as np

labe_list =[]
pred_list =[]

id_list = []
score_list = []

def remove_chongfu_res50():
    TXT_dir = '/home/agv_server/fanyalei/fan_717/Faster-RCNN_Tensorflow-master/output/evaluate_result_pickle/FasterRCNN_20180527/'
    with open(TXT_dir + 'det_foreignobj.txt','r') as file:
        for i ,line in enumerate (file.readlines() ):
            id_list.append(line.split(" ")[0])
            score_list.append(line.split(' ')[1])

        remove_repeat_list = list(set(id_list))
        print(remove_repeat_list)
        print(len(remove_repeat_list))

        for i in remove_repeat_list:
            index = id_list.index(i)
            score = score_list[index]

            #print(i.split("0")[0])
            if (i.split("0")[0]) == 'noobj':
                labe_list.append(0)
            else:
                labe_list.append(1)
            pred_list.append(score)

        np.savetxt('/home/agv_server/fanyalei/fan_717/Faster-RCNN_Tensorflow-master/fanyalei/ResNet50_label.txt',np.array(labe_list).reshape(-1,1))
        np.savetxt('/home/agv_server/fanyalei/fan_717/Faster-RCNN_Tensorflow-master/fanyalei/ResNet50_pred_1.txt',np.array(pred_list).reshape(-1, 1),fmt= '%s')

def remove_chongfu_res101():
    TXT_dir = '/home/agv_server/fanyalei/fan_717/Faster-RCNN_Tensorflow-master/output/evaluate_result_pickle/FasterRCNN_20180517/'
    with open(TXT_dir + 'det_foreignobj.txt','r') as file:
        for i ,line in enumerate (file.readlines() ):
            id_list.append(line.split(" ")[0])
            score_list.append(line.split(' ')[1])

        remove_repeat_list = list(set(id_list))
        print(remove_repeat_list)
        print(len(remove_repeat_list))

        for i in remove_repeat_list:
            index = id_list.index(i)
            score = score_list[index]

            #print(i.split("0")[0])
            if (i.split("0")[0]) == 'noobj':
                labe_list.append(0)
            else:
                labe_list.append(1)
            pred_list.append(score)

        np.savetxt('/home/agv_server/fanyalei/fan_717/Faster-RCNN_Tensorflow-master/fanyalei/ResNet101_label.txt',np.array(labe_list).reshape(-1,1))
        np.savetxt('/home/agv_server/fanyalei/fan_717/Faster-RCNN_Tensorflow-master/fanyalei/ResNet101_pred_1.txt',np.array(pred_list).reshape(-1, 1),fmt= '%s')


def load_txt():
    pred = np.loadtxt('/home/agv_server/fanyalei/fan_717/Faster-RCNN_Tensorflow-master/fanyalei/ResNet50_pred_1.txt')
    print(type(pred))

if __name__ == '__main__':
    remove_chongfu_res50()
    #remove_chongfu_res101()
    #load_txt()



