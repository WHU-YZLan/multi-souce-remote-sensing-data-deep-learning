import os

import cv2
import mydataloader
from predict import predict, model_from_checkpoint_path, visualize_segmentation
from osgeo import gdal
import numpy as np


def my_predict():
    _model = model_from_checkpoint_path("mymodel\\vggunetmergenew")
    for x in [400, 2400, 4400, 6400]:
        for i in range(250):
            img_idx = str(x + i)
            pr = predict(model=_model,
                         inp="GID_dataset/test_img/" + img_idx + ".tif",
                         out_fname="predict/gid_unet/" + img_idx + "_color.png",
                         colors=[(0, 0, 0), (200, 0, 0), (250, 0, 150), (200, 150, 150),
                                 (250, 150, 150), (0, 200, 0), (150, 250, 0), (150, 200, 150),
                                 (200, 0, 200), (150, 0, 250), (150, 150, 250), (250, 200, 0),
                                 (200, 200, 0), (0, 0, 200), (0, 150, 200), (0, 200, 250)],
                         prediction_width=256,
                         prediction_height=256)
            pr2 = cv2.resize(pr, (256, 256), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite("predict/gid_unet/" + img_idx + ".png", pr2)


def whole_predict():
    _model = model_from_checkpoint_path("unet")
    # inp_new = cv2.imread(r"cequ/test_infrared.tif")
    img = gdal.Open("cequ/test.tif")
    inp_new = mydataloader.Multiband2Array(img)
    inp_new = inp_new[:, :, :11]
    # inp_new[:, :, :8] = 0
    _rows, _cols = inp_new.shape[0], inp_new.shape[1]
    res = np.zeros(((int(_rows / 256) + 1) * 256, (int(_cols / 256) + 1) * 256), np.uint8)
    for i in range(0, _rows, 256):
        for j in range(0, _cols, 256):

            if (i + 256) > _rows:
                if (j + 256) > _cols:
                    tmp_img = inp_new[i:_rows, j:_cols, :]
                else:
                    tmp_img = inp_new[i:_rows, j:j + 256, :]
            elif (j + 256) > _cols:
                tmp_img = inp_new[i:i + 256, j:_cols, :]
            else:
                tmp_img = inp_new[i:i + 256, j:j + 256, :]
            tmp_pr = predict(model=_model,
                             inp=tmp_img,
                             colors=[(0, 0, 0), (0, 0, 255),(0, 0, 0),(0, 0, 0),(0, 0, 0),(0, 0, 0),(0, 0, 0),(0, 0, 0),(0, 0, 0),(0, 0, 0),(0, 0, 0),(0, 0, 0),(0, 0, 0),(0, 0, 0),(0, 0, 0),(0, 0, 0),(0, 0, 0),(0, 0, 0),(0, 0, 0),(0, 0, 0),(0, 0, 0),(0, 0, 0)],
                             prediction_width=256,
                             prediction_height=256)
            res[i:i + 256, j:j + 256] = cv2.resize(tmp_pr, (256, 256), interpolation=cv2.INTER_NEAREST)
    # seg_img = visualize_segmentation(res[0:_rows, 0:_cols], img, n_classes=16,
    #                                  colors=[(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
    #                                          (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
    #                                          (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
    #                                          (0, 0, 0), (0, 0, 200), (0, 150, 200), (0, 200, 250)],
    #                                  overlay_img=False,
    #                                  show_legends=False,
    #                                  class_names=None,
    #                                  prediction_width=_cols,
    #                                  prediction_height=_rows)
    """
    colors=[(0, 0, 0), (200, 0, 0), (250, 0, 150), (200, 150, 150),
                                     (250, 150, 150), (0, 200, 0), (150, 250, 0), (150, 200, 150),
                                     (200, 0, 200), (150, 0, 250), (150, 150, 250), (250, 200, 0),
                                     (200, 200, 0), (0, 0, 200), (0, 150, 200), (0, 200, 250)],
    """
    cv2.imwrite("predict/pre.tif", res[0:_rows,0:_cols])



#滑动窗口膨胀预测
def sliding_window_predict(model, input_image, prediction_size, stride):
    # 获取输入图像的大小
    height, width, layer = input_image.shape

    #生成膨胀图像
    dilate_image = np.ones((height+stride, width+stride, layer), dtype=np.uint8)*255
    # print((stride/2), (stride/2+height), (stride/2+width))
    dilate_image[int(stride/2):int(stride/2)+height, int(stride/2):int(stride/2)+width, :] = input_image

    # 计算输出图像的大小
    output_width = (width - prediction_size + stride) // stride + 1
    output_height = (height - prediction_size + stride) // stride + 1

    # 初始化输出图像
    output_image = np.zeros((int(height/4), int(width/4)), dtype=np.uint8)


    # 在输入图像上进行滑动窗口预测
    for i in range(output_height):
        for j in range(output_width):
            # 计算当前窗口的位置
            x = j * stride
            y = i * stride

            # 提取当前窗口
            window = dilate_image[y:y+prediction_size, x:x+prediction_size]

            # 对窗口进行预测
            prediction = \
                predict(model=model, inp=window, prediction_width=prediction_size, prediction_height=prediction_size)

            # prediction = np.argmax(prediction, axis=-1).astype(np.uint8)
            # prediction = \
            #     cv2.resize(prediction, (prediction_size, prediction_size), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            prediction = prediction.astype(np.uint8)
            if i + j < 10:
                cv2.imwrite("D:/chibi/infrared/test/"+str(i*output_width+j)+".tif", prediction)
            # print(np.int32(y+prediction_size/2), np.int32(x+prediction_size/2))
            # 更新输出图像和权重图
            # output_image[y:int(y+prediction_size/2), x:int(x+prediction_size/2)] += \
            #     prediction[int(stride/2):int(stride/2 + prediction_size/2), int(stride/2):int(stride/2 + prediction_size/2)]
            output_image[int(y/4):int(y/4+prediction_size/8), int(x/4):int(x/4+prediction_size/8)] += \
                prediction[int(stride/8):int(stride/8 + prediction_size/8), int(stride/8):int(stride/8 + prediction_size/8)]


    return output_image


if __name__ == '__main__':
    # my_predict()
    # model = model_from_checkpoint_path("hrnet_tcolor")
    # # img = gdal.Open("cequ/0.tif")
    # # inp_new = mydataloader.Multiband2Array(img)
    # inp_new = cv2.imread("cequ/1.tif", 1)
    # pre = sliding_window_predict(model, inp_new, 256, 256, 256)
    # cv2.imwrite("predict/1.tif", pre)

    # whole_predict()

    # #读取模型和研究区影像
    # model = model_from_checkpoint_path("hrnet")
    # img = gdal.Open("D:/chibi/infrared/image/DJI_0041_R.jpg")
    # inp_new = mydataloader.Multiband2Array(img)
    # # inp_new = cv2.imread("D:/chibi/infrared/image/DJI_0001_R.jpg")
    # #选择波段组合
    # # inp_new = inp_new[:, :, 0:5]
    #
    # # z = np.zeros((6060, 6060, 11), dtype=np.uint8)
    # # for i in [5, 6, 7, 8, 9, 10]:
    # #     inp_new[:, :, i] = z
    # # # inp_new = cv2.imread("cequ/test.tif", 1)
    # # x = inp_new[1000, 1000, :]
    # # z[:, :, 8:11] = inp_new
    # inp_new = cv2.resize(inp_new, (2560, 2048), interpolation=cv2.INTER_NEAREST)
    # #滑动窗口膨胀预测
    # pre = sliding_window_predict(model, inp_new, 256, 128)
    # cv2.imwrite("D:/chibi/infrared/DJI_0041_R.tif", pre)

    #批量读取影像预测
    path = "D:/chibi/infrared/image/"
    model = model_from_checkpoint_path("hrnet")
    for filename in os.listdir(path):
        img = gdal.Open(path+filename)
        inp_new = mydataloader.Multiband2Array(img)
        inp_new = cv2.resize(inp_new, (2560, 2048), interpolation=cv2.INTER_NEAREST)
        #滑动窗口膨胀预测
        pre = sliding_window_predict(model, inp_new, 256, 128)
        cv2.imwrite("D:/chibi/infrared/"+filename.replace(".JPG", ".tif"), pre)