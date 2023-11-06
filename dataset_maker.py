import cv2
from osgeo import gdal
import os
import numpy as np


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



if __name__ == "__main__":
    image = gdal.Open('D:/Multimodal/1101/train.tif')
    image = Multiband2Array(image)
    label = cv2.imread('D:/Multimodal/1101/train_label.tif', 0)
    spectral = image[:, :, :5]
    infrared = image[:, :, 5:8]
    rgb = image[:, :, 8:11]

    row, col, _ = rgb.shape

    dilate_rgb = np.zeros((row + 128, col + 128, 3), dtype=np.uint8)
    dilate_infrared = np.zeros((row + 128, col + 128, 3), dtype=np.uint8)
    dilate_spectral = np.zeros((row + 128, col + 128, 5), dtype=np.uint8)
    dilate_label = np.zeros((row + 128, col + 128), dtype=np.uint8)
    dilate_rgb[:row, :col, :] = rgb
    dilate_infrared[:row, :col, :] = infrared
    dilate_spectral[:row, :col, :] = spectral
    dilate_label[:row, :col] = label

    # 计算输出图像的大小
    output_width = (col - 512 + 128) // 128 + 1
    output_height = (row - 512 + 128) // 128 + 1

    for i in range(output_height):
        for j in range(output_width):
            # 计算当前窗口的位置
            x = j * 128
            y = i * 128

            # 提取当前窗口
            window_rgb = dilate_rgb[y:y+512, x:x+512, :]
            window_infrared = dilate_infrared[y:y + 512, x:x + 512, :]
            window_spectral = dilate_spectral[y:y + 512, x:x + 512, :]
            window_label = dilate_label[y:y + 512, x:x + 512]

            window_rgb = cv2.cvtColor(window_rgb, cv2.COLOR_RGB2BGR)
            window_infrared = cv2.cvtColor(window_infrared, cv2.COLOR_RGB2BGR)



            output_file = str(i*output_width+j) + '.tif'
            cv2.imwrite('D:/Multimodal/1101/infrared_train/'+output_file, window_infrared)
            cv2.imwrite('D:/Multimodal/1101/rgb_train/'+output_file, window_rgb)
            cv2.imwrite('D:/Multimodal/1101/label_train/'+output_file, window_label)
            driver = gdal.GetDriverByName("GTiff")
            dataset = driver.Create('D:/Multimodal/1101/spectral_train/'+output_file, 512, 512, 5, gdal.GDT_Byte)

            # 将图像数据写入各个波段
            for band_num in range(5):
                band = dataset.GetRasterBand(band_num + 1)
                band.WriteArray(window_spectral[:, :, band_num])

            dataset = None

    pass