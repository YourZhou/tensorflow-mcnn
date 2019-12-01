# -*-coding:utf-8-*-
# @Author : YourZhou
# @Time : 2019-11-30


import random
import cv2 as cv
from scipy.io import loadmat
from GaussianDensity import *
import numpy as np
from matplotlib import pyplot as plt

np.set_printoptions(threshold=np.inf)


def get_cropped_crowd_image(ori_crowd_img, points, crop_size):
    """
    随机裁剪原始图像
    :param ori_crowd_img: 原始人群图像，形状为[h，w，channel]
    :param points: 所有点的原始位置集
    :param crop_size: 我们需要的裁剪人群图像大小
    :return: 裁剪的图像、裁剪的点、裁剪的点计数
    """
    # 获得图像的高和宽
    h, w = ori_crowd_img.shape[0], ori_crowd_img.shape[1]

    # 如果原始图像大小<裁剪后大小，请减小裁剪大小（paddle不需要）
    # if h < crop_size or w < crop_size:
    #     crop_size = crop_size // 2

    # 随机获取图像面积
    x1 = random.randint(0, h - crop_size)
    y1 = random.randint(0, w - crop_size)
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    # 裁剪后的人群图像
    cropped_crowd_img = ori_crowd_img[x1:x2, y1:y2, ...]

    # 裁剪后计算点
    cropped_points = []
    # 遍历所有的点,之保留裁剪后得到的点
    for i in range(len(points)):
        if x1 <= points[i, 1] <= x2 and y1 <= points[i, 0] <= y2:
            points[i, 0] = points[i, 0] - y1
            points[i, 1] = points[i, 1] - x1
            cropped_points.append(points[i])
    cropped_points = np.asarray(cropped_points)
    # 获得列表长度，得到点总数
    cropped_crowd_count = len(cropped_points)
    # 裁剪的图像、裁剪的点、裁剪的点计数
    return cropped_crowd_img, cropped_points, cropped_crowd_count


def get_scaled_crowd_image_and_points(crowd_img, points, scale):
    """
    获取相应密度图的缩放原图像和缩放点
    :param crowd_image: 被缩放生成地面真实密度图的人群图像（被缩放的原图）
    :param points: 要缩放以生成地面真密度图的所有点的位置集（被缩放的点）
    :param scale: 缩放的倍数
    :return: 缩放后图像，缩放后的点
    """
    # 获得图像的高和宽
    h = crowd_img.shape[0]
    w = crowd_img.shape[1]
    # 将图片俺按缩放倍数进行缩放
    scaled_crowd_img = cv.resize(crowd_img, (w // scale, h // scale))
    # 点的位置也按缩放位置进行缩放变化
    for i in range(len(points)):
        points[i] = points[i] / scale

    # 得到缩放后的图、缩放后图的点的位置
    return scaled_crowd_img, points


def read_crop_train_data(img_path, gt_path, crop_size=256, scale=4, knn_phase=True, k=2, min_head_size=16,
                         max_head_size=200):
    """
    从数据集中读取数据，并输入和标记网络
    :param img_path: 人群图像路径
    :param gt_path: 人群图像密度标注路径
    :param crop_size: 裁剪的尺寸
    :param scale: 比例因子，与累积的下采样因子相对应
    :param knn_phase: 是否打开自适应高斯核，确定是使用几何自适应高斯核还是一般高斯核
    :param k:  整数，近邻的数目
    :param min_head_size: 原始人群图像中头部尺寸的最小值
    :param max_head_size: 原始人群图像中头部尺寸的最大值
    :return: 网络输入原图，标注密度图，地面真相群众数
    """

    # 读取初始图像原始
    ori_crowd_img = cv.imread(img_path)

    # 读取数据集中的.mat文件
    label_data = loadmat(gt_path)

    # 获得每个点标注信息
    points = label_data['image_info'][0][0]['location'][0][0]

    # 将图片随机裁剪成256X256图像
    # 得到裁剪后的图像、裁剪后的点、裁剪后的点数量
    cropped_crowd_img, cropped_points, cropped_crowd_count = get_cropped_crowd_image(ori_crowd_img, points,
                                                                                     crop_size=crop_size)

    # 将图像进行缩放，缩放为256/4 = 64 （64x64）
    # 的搭配缩放后图像和缩放后的点
    cropped_scaled_crowd_img, cropped_scaled_points = get_scaled_crowd_image_and_points(cropped_crowd_img,
                                                                                        cropped_points, scale=scale)

    # 获得缩放后图片尺寸
    cropped_scaled_crowd_img_size = [cropped_scaled_crowd_img.shape[0], cropped_scaled_crowd_img.shape[1]]
    # 缩放原始人群图像中头部尺寸的最小值和最大值
    scaled_min_head_size = min_head_size / scale
    scaled_max_head_size = max_head_size / scale

    # 修剪和缩放后，生成需要尺寸的密度图（64x64）
    density_map = get_density_map(cropped_scaled_crowd_img_size, cropped_scaled_points,
                                  knn_phase, k, scaled_min_head_size, scaled_max_head_size)

    # cropped_crowd_img = np.asarray(cropped_crowd_img)
    # cropped_crowd_img = cropped_crowd_img.reshape(
    #     (1, 3, cropped_crowd_img.shape[0], cropped_crowd_img.shape[1]))

    # print(cropped_crowd_img.shape)
    # cv.imshow("123", cropped_crowd_img)

    # im = cropped_crowd_img.transpose().astype('float32')
    # im = np.expand_dims(im, axis=0)
    # density_map = density_map.transpose().astype('float32')
    #
    # # print(im.shape)
    # # cv.waitKey(0)
    # # cv.destroyAllWindows()
    #
    # cropped_crowd_count = np.asarray(cropped_crowd_count).reshape((1, 1))
    # cropped_scaled_density_map = density_map.reshape((1, 1, density_map.shape[0], density_map.shape[1]))
    #

    cropped_crowd_img = cropped_crowd_img.reshape(
        (1, cropped_crowd_img.shape[0], cropped_crowd_img.shape[1], cropped_crowd_img.shape[2]))
    cropped_crowd_count = np.asarray(cropped_crowd_count).reshape((1, 1))
    cropped_scaled_density_map = density_map.reshape((1, density_map.shape[0], density_map.shape[1], 1))


    return cropped_crowd_img, cropped_scaled_density_map, cropped_crowd_count


def read_resize_train_data(img_path, gt_path, scale=4, knn_phase=True, k=2, min_head_size=16,
                           max_head_size=200):
    """
    从数据集中读取数据，并输入和标记网络
    :param img_path: 人群图像路径
    :param gt_path: 人群图像密度标注路径
    :param scale: 比例因子，与累积的下采样因子相对应
    :param knn_phase: 是否打开自适应高斯核，确定是使用几何自适应高斯核还是一般高斯核
    :param k:  整数，近邻的数目
    :param min_head_size: 原始人群图像中头部尺寸的最小值
    :param max_head_size: 原始人群图像中头部尺寸的最大值
    :return: 网络输入原图，标注密度图，地面真相群众数
    """

    # 读取初始图像原始
    ori_crowd_img = cv.imread(img_path)

    # 读取数据集中的.mat文件
    label_data = loadmat(gt_path)

    # 获得每个点标注信息
    points = label_data['image_info'][0][0]['location'][0][0]

    # 将图像进行缩放，缩放为256/4 = 64 （64x64）
    # 的搭配缩放后图像和缩放后的点
    # 获得图像的高和宽
    h = ori_crowd_img.shape[0]
    w = ori_crowd_img.shape[1]
    # 获得长宽各缩放的倍数
    h_multiple = h / 256.0
    w_multiple = w / 256.0
    # 将图片俺按缩放倍数进行缩放
    scaled_img = cv.resize(ori_crowd_img, (256, 256))
    # 点的位置也按缩放位置进行缩放变化
    for i in range(len(points)):
        points[i, 0] = points[i, 0] / w_multiple
        points[i, 1] = points[i, 1] / h_multiple

    # 得到缩放后的图、缩放后图的点的位置
    cropped_crowd_count = len(points)
    # print(len(points))

    # 将图像进行缩放，缩放为256/4 = 64 （64x64）
    # 的搭配缩放后图像和缩放后的点
    cropped_scaled_crowd_img, cropped_scaled_points = get_scaled_crowd_image_and_points(scaled_img,
                                                                                        points, scale=scale)

    # 获得缩放后图片尺寸
    cropped_scaled_crowd_img_size = [cropped_scaled_crowd_img.shape[0], cropped_scaled_crowd_img.shape[1]]
    # 缩放原始人群图像中头部尺寸的最小值和最大值
    scaled_min_head_size = min_head_size / scale
    scaled_max_head_size = max_head_size / scale

    # 修剪和缩放后，生成需要尺寸的密度图（64x64）
    density_map = get_density_map(cropped_scaled_crowd_img_size, cropped_scaled_points,
                                  knn_phase, k, scaled_min_head_size, scaled_max_head_size)

    # cropped_crowd_img = np.asarray(cropped_crowd_img)
    # cropped_crowd_img = scaled_img.reshape(
    #     (1, 3, scaled_img.shape[0], scaled_img.shape[1]))

    # print(density_map.shape)
    # print(scaled_img.shape)
    # cv.imshow("123", scaled_img)

    # im = scaled_img.transpose().astype('float32')
    # im = np.expand_dims(im, axis=0)
    # density_map = density_map.transpose().astype('float32')
    #
    # # print(im.shape)
    # # cv.waitKey(0)
    # # cv.destroyAllWindows()
    #
    # cropped_crowd_count = np.asarray(cropped_crowd_count).reshape((1, 1))
    # cropped_scaled_density_map = density_map.reshape((1, 1, density_map.shape[0], density_map.shape[1]))

    cropped_crowd_img = cropped_scaled_crowd_img.reshape(
        (1, cropped_scaled_crowd_img.shape[0], cropped_scaled_crowd_img.shape[1], cropped_scaled_crowd_img.shape[2]))
    cropped_crowd_count = np.asarray(cropped_crowd_count).reshape((1, 1))
    cropped_scaled_density_map = density_map.reshape((1, density_map.shape[0], density_map.shape[1], 1))

    return cropped_crowd_img, cropped_scaled_density_map, cropped_crowd_count


def read_test_data(img_path, gt_path, scale=8, deconv_is_used=False, knn_phase=True, k=2, min_head_size=16,
                   max_head_size=200):
    """
    read_the testing data from datasets ad the input and label of network
    :param img_path: the crowd image path
    :param gt_path: the label(ground truth) data path
    :param scale: the scale factor, accorting to the accumulated downsampling factor
    :param knn_phase: True or False, determines wheather to use geometry-adaptive Gaussain kernel or general one
    :param k:  a integer, the number of neareat neighbor
    :param min_head_size: the minimum value of the head size in original crowd image
    :param max_head_size: the maximum value of the head size in original crowd image
    :return: the crwod image as the input of network, the scaled density map as the ground truth of network,
             the ground truth crowd count
    """

    ori_crowd_img = cv.imread(img_path)

    # read the .mat file in dataset
    label_data = loadmat(gt_path)
    points = label_data['image_info'][0][0]['location'][0][0]
    crowd_count = label_data['image_info'][0][0]['number'][0][0]
    h, w = ori_crowd_img.shape[0], ori_crowd_img.shape[1]

    if deconv_is_used:
        h_ = h - (h // scale) % 2
        rh = h_ / h
        w_ = w - (w // scale) % 2
        rw = w_ / w
        ori_crowd_img = cv.resize(ori_crowd_img, (w_, h_))
        points[:, 1] = points[:, 1] * rh
        points[:, 0] = points[:, 0] * rw
    # scaled_crowd_img, scaled_points = ori_crowd_img,points
    scaled_crowd_img, scaled_points = get_scaled_crowd_image_and_points(ori_crowd_img, points, scale=scale)
    # scaled_crowd_count = crowd_count
    scaled_crowd_img_size = [scaled_crowd_img.shape[0], scaled_crowd_img.shape[1]]
    scaled_min_head_size = min_head_size / scale
    scaled_max_head_size = max_head_size / scale

    # after cropped and scaled
    density_map = get_density_map(scaled_crowd_img_size, scaled_points, knn_phase, k, scaled_min_head_size,
                                  scaled_max_head_size)
    ori_crowd_img = ori_crowd_img.reshape((1, ori_crowd_img.shape[0], ori_crowd_img.shape[1], ori_crowd_img.shape[2]))
    crowd_count = np.asarray(crowd_count).reshape((1, 1))
    scaled_density_map = density_map.reshape((1, density_map.shape[0], density_map.shape[1], 1))

    return ori_crowd_img, scaled_density_map, crowd_count


def show_density_map(density_map):
    """
    show the density map to help us analysis the distribution of the crowd
    :param density_map: the density map, the shape is [h, w]
    """

    plt.imshow(density_map, cmap='jet')
    plt.show()


if __name__ == '__main__':
    crowd_img, density_map, cropped_crowd_count = read_resize_train_data(
        'D:\\Scenic_file\\Scenic_dataset\\Scenic_photo\\Temple1\\photo\\Scenic_191.jpg',
        'D:\\Scenic_file\\Scenic_dataset\\Scenic_photo\\Temple1\\mat\\GT_Scenic_191.mat', scale=4)


    # print(density_map.shape)
    # redmp = density_map[0, :, :, 0]
    # ori_crowd_img = cv.imread('D:\\Scenic_file\\Scenic_dataset\\Scenic_photo\\Temple1\\photo\\Scenic_191.jpg')
    # img = ori_crowd_img.reshape((ori_crowd_img.shape[0], ori_crowd_img.shape[1], ori_crowd_img.shape[2]))
    # final_img = image_processing(redmp)
    # # cv.imshow("really", final_img)
    # final_img = image_add_heatmap(ori_crowd_img, final_img, 0.5)
    # cv.imshow("really", final_img)
    #
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    sum = np.sum(np.sum(density_map))
    print(sum, cropped_crowd_count)
    print(density_map.shape)
    result = density_map[0, :, :, 0]
    print(result.shape)
    show_density_map(result)
    show_density_map(crowd_img[0, :, :, 0])
