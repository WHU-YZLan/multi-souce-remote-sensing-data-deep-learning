from osgeo import gdal
import itertools
import os
import random
import six
import numpy as np
import cv2

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found, disabling progress bars")


    def tqdm(iter):
        return iter

from models.config import IMAGE_ORDERING
from data_utils.augmentation import augment_seg, custom_augment_seg

class_colors = [(0, 0, 0), (150, 100, 0), (0, 255, 255), (0, 0, 255), (255, 255, 0), (0, 255, 0), (0, 128, 128)]

ACCEPTABLE_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tif"]
ACCEPTABLE_SEGMENTATION_FORMATS = [".png", ".bmp",".tif"]


class DataLoaderError(Exception):
    pass


def get_image_list_from_path(images_path):
    image_files = []
    for dir_entry in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, dir_entry)) and \
                os.path.splitext(dir_entry)[1] in ACCEPTABLE_IMAGE_FORMATS:
            file_name, file_extension = os.path.splitext(dir_entry)
            image_files.append(os.path.join(images_path, dir_entry))
    return image_files

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


def get_pairs_from_paths(images_path, segs_path, ignore_non_matching=False, other_inputs_paths=None):
    """ Find all the images from the images_path directory and
        the segmentation images from the segs_path directory
        while checking integrity of data """

    image_files = []
    segmentation_files = {}

    for dir_entry in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, dir_entry)) and \
                os.path.splitext(dir_entry)[1] in ACCEPTABLE_IMAGE_FORMATS:
            file_name, file_extension = os.path.splitext(dir_entry)
            image_files.append((file_name, file_extension,
                                os.path.join(images_path, dir_entry)))

    if other_inputs_paths is not None:
        other_inputs_files = []

        for i, other_inputs_path in enumerate(other_inputs_paths):
            temp = []

            for y, dir_entry in enumerate(os.listdir(other_inputs_path)):
                if os.path.isfile(os.path.join(other_inputs_path, dir_entry)) and \
                        os.path.splitext(dir_entry)[1] in ACCEPTABLE_IMAGE_FORMATS:
                    file_name, file_extension = os.path.splitext(dir_entry)

                    temp.append((file_name, file_extension,
                                 os.path.join(other_inputs_path, dir_entry)))

            other_inputs_files.append(temp)

    for dir_entry in os.listdir(segs_path):
        if os.path.isfile(os.path.join(segs_path, dir_entry)) and \
                os.path.splitext(dir_entry)[1] in ACCEPTABLE_SEGMENTATION_FORMATS:
            file_name, file_extension = os.path.splitext(dir_entry)
            full_dir_entry = os.path.join(segs_path, dir_entry)
            if file_name in segmentation_files:
                raise DataLoaderError("Segmentation file with filename {0}"
                                      " already exists and is ambiguous to"
                                      " resolve with path {1}."
                                      " Please remove or rename the latter."
                                      .format(file_name, full_dir_entry))

            segmentation_files[file_name] = (file_extension, full_dir_entry)

    return_value = []
    # Match the images and segmentations
    for image_file, _, image_full_path in image_files:
        if image_file in segmentation_files:
            if other_inputs_paths is not None:
                other_inputs = []
                for file_paths in other_inputs_files:
                    success = False

                    for (other_file, _, other_full_path) in file_paths:
                        if image_file == other_file:
                            other_inputs.append(other_full_path)
                            success = True
                            break

                    if not success:
                        raise ValueError("There was no matching other input to", image_file, "in directory")

                return_value.append((image_full_path,
                                     segmentation_files[image_file][1], other_inputs))
            else:
                return_value.append((image_full_path,
                                     segmentation_files[image_file][1]))
        elif ignore_non_matching:
            continue
        else:
            # Error out
            raise DataLoaderError("No corresponding segmentation "
                                  "found for image {0}."
                                  .format(image_full_path))

    return return_value


def get_image_array(image_input,
                    width, height,
                    imgNorm="divide", ordering='channels_first', read_image_type=1):
    """ Load image array from input """

    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_image_array: path {0} doesn't exist"
                                  .format(image_input))
        img = gdal.Open(image_input)
    else:
        raise DataLoaderError("get_image_array: Can't process input type {0}"
                              .format(str(type(image_input))))

    if imgNorm == "sub_and_divide":
        img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
    elif imgNorm == "sub_mean":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = np.atleast_3d(img)

        means = [103.939, 116.779, 123.68]

        for i in range(min(img.shape[2], len(means))):
            img[:, :, i] -= means[i]

        img = img[:, :, ::-1]
    elif imgNorm == "divide":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = img / 255.0


    if ordering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
    return img


def get_segmentation_array(image_input, nClasses,
                           width, height, no_reshape=False, read_image_type=0):
    """ Load segmentation array from input """

    seg_labels = np.zeros((height, width, nClasses))

    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_segmentation_array: "
                                  "path {0} doesn't exist".format(image_input))
        img = cv2.imread(image_input, read_image_type)
    else:
        raise DataLoaderError("get_segmentation_array: "
                              "Can't process input type {0}"
                              .format(str(type(image_input))))

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    img = img[:, :]

    for c in range(nClasses):
        seg_labels[:, :, c] = (img == c).astype(int)

    if not no_reshape:
        seg_labels = np.reshape(seg_labels, (width * height, nClasses))

    return seg_labels

def myverify_segmentation_dataset(images_path, segs_path,
                                n_classes, show_all_errors=False):
    try:
        img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
        if not len(img_seg_pairs):
            print("Couldn't load any data from images_path: "
                  "{0} and segmentations path: {1}"
                  .format(images_path, segs_path))
            return False

        return_value = True
        for im_fn, seg_fn in tqdm(img_seg_pairs):
            img = gdal.Open(im_fn)
            seg = gdal.Open(seg_fn)
            # Check dimensions match
            if not True:
                return_value = False
                print("The size of image {0} and its segmentation {1} "
                      "doesn't match (possibly the files are corrupt)."
                      .format(im_fn, seg_fn))
                if not show_all_errors:
                    break
            else:
                max_pixel_value = np.max(seg.ReadAsArray(0,0,128,128)[:,:])
                if max_pixel_value >= n_classes:
                    return_value = False
                    print("The pixel values of the segmentation image {0} "
                          "violating range [0, {1}]. "
                          "Found maximum pixel value {2}"
                          .format(seg_fn, str(n_classes - 1), max_pixel_value))
                    if not show_all_errors:
                        break
        if return_value:
            print("Dataset verified! ")
        else:
            print("Dataset not verified!")
        return return_value
    except DataLoaderError as e:
        print("Found error during data loading\n{0}".format(str(e)))
        return False


def myimage_segmentation_generator(images_path, segs_path, batch_size,
                                 n_classes, input_height, input_width,
                                 output_height, output_width,
                                 do_augment=False,
                                 augmentation_name="aug_all",
                                 custom_augmentation=None,
                                 other_inputs_paths=None, preprocessing=None,
                                 read_image_type=cv2.IMREAD_COLOR, ignore_segs=False):
    if not ignore_segs:
        img_seg_pairs = get_pairs_from_paths(images_path, segs_path, other_inputs_paths=other_inputs_paths)
        random.shuffle(img_seg_pairs)
        zipped = itertools.cycle(img_seg_pairs)
    else:
        img_list = get_image_list_from_path(images_path)
        random.shuffle(img_list)
        img_list_gen = itertools.cycle(img_list)

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            if other_inputs_paths is None:

                if ignore_segs:
                    im = next(img_list_gen)
                    seg = None
                else:
                    im, seg = next(zipped)
                    ##图像读取
                    my_seg = gdal.Open(seg)
                    ##将11通道图像转换为网络能够接受的numpy.ndarray类型数据
                    mynew_seg=Multiband2Array(my_seg)

                my_im=gdal.Open(im)
                ##将11通道图像转换为网络能够接受的numpy.ndarray类型数据
                mynew_im=Multiband2Array(my_im)




                if do_augment:

                    assert ignore_segs == False, "Not supported yet"

                    if custom_augmentation is None:
                        im, seg[:, :, 0] = augment_seg(im, seg[:, :, 0],
                                                       augmentation_name)
                    else:
                        im, seg[:, :, 0] = custom_augment_seg(im, seg[:, :, 0],
                                                              custom_augmentation)

                if preprocessing is not None:
                    im = preprocessing(im)


                X.append(get_image_array(mynew_im, input_width,
                                         input_height, ordering=IMAGE_ORDERING))
            else:

                assert ignore_segs == False, "Not supported yet"

                im, seg, others = next(zipped)

                im = gdal.Open(im)
                seg = gdal.Open(seg)
                imnew = im.ReadAsArray(0, 0, 0, 256, 256)

                segnew = seg.ReadAsArray(0, 0, 256, 256)


                oth = []
                for f in others:
                    oth.append(cv2.imread(f, read_image_type))

                if do_augment:
                    if custom_augmentation is None:
                        ims, seg[:, :, 0] = augment_seg(im, seg[:, :, 0],
                                                        augmentation_name, other_imgs=oth)
                    else:
                        ims, seg[:, :, 0] = custom_augment_seg(im, seg[:, :, 0],
                                                               custom_augmentation, other_imgs=oth)
                else:
                    ims = [im]
                    ims.extend(oth)

                oth = []
                for i, image in enumerate(ims):
                    oth_im = get_image_array(image, input_width,
                                             input_height, ordering=IMAGE_ORDERING)

                    if preprocessing is not None:
                        if isinstance(preprocessing, Sequence):
                            oth_im = preprocessing[i](oth_im)
                        else:
                            oth_im = preprocessing(oth_im)

                    oth.append(oth_im)

                X.append(oth)

            if not ignore_segs:
                Y.append(get_segmentation_array(
                    mynew_seg, n_classes, output_width, output_height))

        if ignore_segs:
            yield np.array(X)
        else:
            yield np.array(X), np.array(Y)


