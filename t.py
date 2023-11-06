import cv2
import numpy as np
import os
from osgeo import gdal

def read_directory(path):
    array_of_img = []
    file_list = os.listdir(path)
    file_list.sort()
    for filename in file_list:
        img = cv2.imread(path + filename, 0)
        array_of_img.append(img)

    return array_of_img, file_list

def Multiband2Array(src_ds):


    xcount=src_ds.RasterXSize # 宽度
    ycount=src_ds.RasterYSize # 高度
    ibands=src_ds.RasterCount # 波段数

    # print "[ RASTER BAND COUNT ]: ", ibands
    for band in range(ibands):
        band += 1
        # print "[ GETTING BAND ]: ", band
        srcband = src_ds.GetRasterBand(band) # 获取该波段
        if srcband is None:
            continue

        # Read raster as arrays 类似RasterIO（C++）
        dataraster = srcband.ReadAsArray(0, 0, xcount, ycount).astype(np.float32) # 这里得到的矩阵大小为 ycount x xcount
        if band==1:
            data=dataraster.reshape((ycount,xcount,1))
        else:
            # 将每个波段的数组很并到一个3维数组中
            data=np.append(data,dataraster.reshape((ycount,xcount,1)),axis=2)


    return data

if __name__ == '__main__':
    # rgb = cv2.imread("predict/pre_multi_rgb.tif", 0)
    # pre = np.zeros((2201, 8812), np.uint8)
    # rgb_sband = cv2.imread("predict/pre_multi_rgbsband.tif", 0)
    # multi = cv2.imread("predict/pre_multi.tif", 0)
    # sband = cv2.imread("predict/pre_multi_sband.tif", 0)
    #
    # #3>1>5>4>2
    # pre[multi == 2] = 2
    # pre[rgb_sband == 4] = 4
    # pre[sband == 5] = 5
    # pre[rgb == 1] = 1
    # pre[multi == 3] = 3
    #
    # cv2.imwrite("predict/pre.tif", pre)

    # label = cv2.imread("predict/pre_infrared.tif", 0)
    # color = np.zeros((2201, 8812, 3), np.uint8)
    # color[label == 1] = [0, 0, 128]
    # color[label == 2] = [0, 128, 0]
    # color[label == 3] = [0, 128, 128]
    # color[label == 4] = [128, 0, 0]
    # color[label == 5] = [128, 0, 128]
    #
    # cv2.imwrite("color/infrared.tif", color)

    # path = "water/val_label/"
    # file_list = os.listdir(path)
    # for i in range(len(file_list)):
    #     if "sat_" in file_list[i]:
    #         continue
    #     else:
    #         img = cv2.imread(path + file_list[i], 0)
    #         label_new = np.zeros((512, 512), np.uint8)
    #         label_new[img == 3] = 1
    #         cv2.imwrite(path + file_list[i], label_new)
    #     print(i)

    # img = cv2.imread("D:/Multimodal/data/TrueColor.jpg")
    # img = cv2.resize(img, (1942, 1367), interpolation=cv2.INTER_NEAREST)
    # cv2.imwrite("D:/Multimodal/downsampling/TrueColor.tif", img)

    image_gdal = gdal.Open('multichannel_2505.tif')
    image = Multiband2Array(image_gdal)
    image1 = image[:, :, 5:8]
    image2 = image[:, :, 8:11]
    image3 = image[:, :, 0]
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
    image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
    cv2.imwrite('infrared.jpg', image1)
    cv2.imwrite('rgb.jpg', image2)
    cv2.imwrite('sband.jpg', image3)