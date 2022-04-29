import cv2
from keras.layers import Lambda
import numpy
from requests import patch
import tensorflow as tf
from skimage import data, io, filters
import numpy as np
from numpy import array
from numpy.random import randint
import cv2 as cv
import os
from keras import backend as K
import sys
import itertools
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

def preprocess_input(image, mean, std):
    image = (image/255 - mean)/std
    return image



def show_result(num_epoch, G_net, imgs_lr, imgs_hr):
    test_images = G_net.predict(imgs_lr)

    fig, ax = plt.subplots(1, 3)

    for j in itertools.product(range(3)):
        ax[j].get_xaxis().set_visible(False)
        ax[j].get_yaxis().set_visible(False)
    

    ax[0].cla()
    ax[0].set_title("lr_Images")
    ax[0].imshow((imgs_lr[0] * 0.5 + 0.5))

    ax[1].cla()
    ax[1].set_title("Fake_Hr_Images")
    ax[1].imshow((test_images[0] * 0.5 + 0.5))

    ax[2].cla()
    ax[2].set_title("True_Hr_Images")
    ax[2].imshow((imgs_hr[0] * 0.5 + 0.5))

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    plt.savefig("results/epoch_" + str(num_epoch) + "_results.png")
    plt.close('all')  #避免内存泄漏

def tf_log10(x):
    numerator = tf.compat.v1.log(x)
    denominator = tf.compat.v1.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def PSNR(y_true, y_pred):
    y_true = (y_true * 0.5 + 0.5) * 255
    y_pred = (y_pred * 0.5 + 0.5) * 255
    max_pixel = 255.0
    y_pred = K.clip(y_pred, 0.0, 255.0)
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))
if __name__ == "__main__":
    pass