import numpy as np
from keras import backend
EPS = 1e-12


def get_iou(gt, pr, n_classes=7):
    class_wise = np.zeros(n_classes)
    gt = gt.eval()
    pr = pr.eval()
    for cl in range(n_classes):
        intersection = np.sum((gt == cl)*(pr == cl))
        union = np.sum(np.maximum((gt == cl), (pr == cl)))
        iou = float(intersection)/(union + EPS)
        class_wise[cl] = iou
        miou = np.mean(class_wise)
    return miou


import tensorflow.keras.backend as K


def miou(y_true, y_pred):
    # 将预测值转化为 0 或 1
    y_pred = K.round(y_pred)

    # 计算交集和并集
    intersection = K.sum(y_true * y_pred, axis=[1, 2])
    union = K.sum(y_true + y_pred, axis=[1, 2]) - intersection

    # 计算iou
    iou = intersection / union

    # 返回iou的平均值作为miou指标
    return K.mean(iou)


import tensorflow as tf


def weighted_cross_entropy_loss(y_true, y_pred):
    # 定义交叉熵损失函数
    cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred)

    # 定义权重
    weight_0 = 1.0  # 类别0的权重
    weight_1 = 4.0  # 类别1的权重

    # 对交叉熵损失函数进行加权
    weighted_loss = (y_true * weight_1) * cross_entropy_loss + ((1 - y_true) * weight_0) * cross_entropy_loss

    return weighted_loss

