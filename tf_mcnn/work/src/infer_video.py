#!/usr/bin/env python
# -*-coding:utf-8-*-
# @Author : YourZhou
# @Time :

import cv2 as cv
import tensorflow as tf

from networks import multi_column_cnn
from utils import *

np.set_printoptions(threshold=np.inf)


# 密度图生成
def image_processing(input):
    # 高斯模糊
    kernel_size = (5, 5)
    sigma = 15
    r_img = cv.GaussianBlur(input, kernel_size, sigma)

    # 灰度图标准化
    norm_img = np.zeros(r_img.shape)
    norm_img = cv.normalize(r_img, norm_img, 0, 255, cv.NORM_MINMAX)
    norm_img = np.asarray(norm_img, dtype=np.uint8)
    # r_img = cv.resize(r_img, (720, 420))
    # utils.show_density_map(r_img)

    # 灰度图颜色反转
    imgInfo = norm_img.shape
    heigh = imgInfo[0]
    width = imgInfo[1]
    dst = np.zeros((heigh, width, 1), np.uint8)
    for i in range(0, heigh):
        for j in range(0, width):
            grayPixel = norm_img[i, j]
            dst[i, j] = 255 - grayPixel

    # 生成热力图
    heat_img = cv.applyColorMap(dst, cv.COLORMAP_JET)  # 注意此处的三通道热力图是cv2专有的GBR排列
    output = cv.cvtColor(heat_img, cv.COLOR_BGR2RGB)  # 将BGR图像转为RGB图像

    return output


# 密度图与原图叠加
def image_add_heatmap(frame, heatmap, alpha=0.5):
    img_size = frame.shape
    heatmap = cv.resize(heatmap, (img_size[1], img_size[0]))
    overlay = frame.copy()
    cv.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), -1)  # 设置蓝色为热度图基本色
    cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # 将背景热度图覆盖到原图
    cv.addWeighted(heatmap, alpha, frame, 1 - alpha, 0, frame)  # 将热度图覆盖到原图
    return frame


def infer():
    set_gpu(0)
    cap = cv.VideoCapture("D:/YourZhouDownloads/FFPUT/school_2/school_01.mp4")
    # img_path = 'D:\\YourZhouProject\\mcnn_project\\pytorch_mcnn\\part_A_final\\test_data\\images\\IMG_45.jpg'
    model_path = 'D:\\YourZhouProject\\mcnn_project\\tf_mcnn\\work\\ckpts\\mcnn\\v1-2200'
    # crop_size = 256

    # place holder位置保持器
    input_img_placeholder = tf.placeholder(tf.float32, shape=([None, None, None, 3]))
    density_map_placeholder = tf.placeholder(tf.float32, shape=(None, None, None, 1))

    inference_density_map = multi_column_cnn(input_img_placeholder)

    while True:
        ret, image = cap.read()
        ori_crowd_img = cv.flip(image, 1)

        # ori_crowd_img = cv.imread(img_path)
        # h, w = ori_crowd_img.shape[0], ori_crowd_img.shape[1]
        img = ori_crowd_img.reshape((ori_crowd_img.shape[0], ori_crowd_img.shape[1], ori_crowd_img.shape[2]))

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, model_path)
            result = sess.run(inference_density_map, feed_dict={input_img_placeholder: [(img - 127.5) / 128]})

        print(result.sum())
        dmap_img = result[0, :, :, 0]
        # utils.show_density_map(dmap_img)

        final_img = image_processing(dmap_img)
        final_img = image_add_heatmap(ori_crowd_img, final_img,0.3)

        cv.imshow("really", final_img)
        if (cv.waitKey(1) == 27):
            cv.destroyAllWindows()
            break


if __name__ == '__main__':
    infer()
