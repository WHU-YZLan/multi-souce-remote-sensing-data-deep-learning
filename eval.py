import os
import numpy as np
from osgeo import gdal
from predict import predict, model_from_checkpoint_path
import cv2
import mydataloader

# 定义IOU计算函数
def iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# 计算IOU和MIOU
def compute_iou_miou(image_dir, model):
    iou_list = []
    num = 1
    for filename in os.listdir(image_dir):
        if filename.endswith(".tif"):
            image_path = os.path.join(image_dir, filename)
            # 预测
            image = gdal.Open(image_path)
            inp = mydataloader.Multiband2Array(image)
            pred = predict(model=model, inp=inp, prediction_width=256, prediction_height=256)

            # 加载标签
            label_path = os.path.join(image_dir.replace("val", "val_label"), filename)
            label = np.array(cv2.imread(label_path, 0))
            pred = cv2.resize(pred, (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.uint8)  # 调整图像大小为256*256

            # 计算IOU
            iou_score = iou(label == 1, pred == 1)  # 计算第一类的IOU
            iou_list.append(iou_score)

            print(num)
            num += 1

    miou = np.mean(iou_list)

    return miou





if __name__ == '__main__':

    #config
    n_classes = 2
    fold_path = "D:/chibi/infrared/photovoltaic/val/"
    class_labels = {0: '背景', 1: '光伏板', 2: '砖石道路', 3: '水体', 4: '建筑', 5: '植被'}

    count = 0
    miou = 0.0
    iou = np.zeros(n_classes)
    tp = np.zeros(n_classes, np.uint32)
    fp = np.zeros(n_classes, np.uint32)
    fn = np.zeros(n_classes, np.uint32)

    #读取模型
    model = model_from_checkpoint_path("hrnet")

    # 遍历文件夹下所有图像
    for filename in os.listdir(fold_path):
        if filename.endswith('.tif'):
            # 加载图像
            img = gdal.Open(os.path.join(fold_path, filename))
            inp = mydataloader.Multiband2Array(img)

            # 预测结果
            pred = predict(model=model, inp=inp, prediction_width=256, prediction_height=256)
            pred = cv2.resize(pred, (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.uint8)  # 调整图像大小为256*256
            cv2.imwrite("D:/chibi/infrared/test/"+filename, pred)

            #读取标签
            label_path = os.path.join(fold_path.replace("val", "val_label"), filename)
            gt_label = np.array(cv2.imread(label_path, 0))

            for cls in range(n_classes):
                tp[cls] += np.sum(np.logical_and(pred == cls, gt_label == cls))
                fp[cls] += np.sum(np.logical_and(pred == cls, gt_label != cls))
                fn[cls] += np.sum(np.logical_and(pred != cls, gt_label == cls))


            count += 1
            print(count)

    # 输出IOU和MIOU
    for cls in range(n_classes):
        iou[cls] = tp[cls]/(tp[cls] + fp[cls] + fn[cls])
        print('IOU for %s: %.4f' % (class_labels[cls], iou[cls]))
    miou = np.mean(iou)
    print('MIOU: %.4f' % miou)


