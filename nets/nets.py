
from keras.layers import Dense,GlobalAveragePooling2D ,Lambda
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import UpSampling2D
from keras.layers import Input
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.layers import LeakyReLU, PReLU
from keras.layers import add
from keras.applications.vgg19 import VGG19
import keras.backend as K
from keras.models import Model
import tensorflow as tf
# 生成器中的残差块
def res_block_gen(x, kernal_size, filters, strides):
    
    gen = x
    
    x = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(x)
    x = BatchNormalization(momentum = 0.5)(x)
    # Using Parametric ReLU
    x = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(x)
    x = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(x)
    x = BatchNormalization(momentum = 0.5)(x)
        
    x = add([gen, x])
    
    return x

#上采样样块
def up_sampling_block(x, kernal_size, filters, strides):
    x = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(x)
    x = UpSampling2D(size = 2)(x)
    x = LeakyReLU(alpha = 0.2)(x)
    
    return x
#--------------------------------------
# 亚像素卷积上采样块
# 生成器 还是用的 UpSampling2D
# 如果有需要可以自己更改
# -------------------------------------
def SubpixelConv2D(input_shape, scale=4):
    def subpixel_shape(input_shape):
        dims = [input_shape[0],input_shape[1] * scale,input_shape[2] * scale,int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape
    
    def subpixel(x):
        return tf.depth_to_space(x, scale)
        
    return Lambda(subpixel, output_shape=subpixel_shape)


#判别器中的卷积块
def discriminator_block(x, filters, kernel_size, strides):
    
    x = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(x)
    x = BatchNormalization(momentum = 0.5)(x)
    x = LeakyReLU(alpha = 0.2)(x)
    
    return x


#生成器

def Generator(input_shape=[128,128,3]):
    

    gen_input = Input(input_shape)
    x = Conv2D(filters = 64, kernel_size = 9, strides = 1, padding = "same")(gen_input)
    x = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(x)
	    
    gen_x = x
        
    # 16 个残差快
    for index in range(16):
            x = res_block_gen(x, 3, 64, 1)
	    
    x = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(x)
    x = BatchNormalization(momentum = 0.5)(x)
    x = add([gen_x, x])
	    
	#两个上采样 -> 放大四倍
    for index in range(2):
        x = up_sampling_block(x, 3, 256, 1)
	    
    x = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same")(x)
    x = Activation('tanh')(x)
	   
    generator_x = Model(inputs = gen_input, outputs = x)
        
    return generator_x



def Discriminator(image_shape=[512,512,3]):
        
        dis_input = Input(image_shape)
        
        x = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(dis_input)
        x = LeakyReLU(alpha = 0.2)(x)
        
        x = discriminator_block(x, 64, 3, 2)
        x = discriminator_block(x, 128, 3, 1)
        x = discriminator_block(x, 128, 3, 2)
        x = discriminator_block(x, 256, 3, 1)
        x = discriminator_block(x, 256, 3, 2)
        x = discriminator_block(x, 512, 3, 1)
        x = discriminator_block(x, 512, 3, 2)
        
        #x = Flatten()(x) # 这里采用Flatten 太浪费现存了 改为 全局池化
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024)(x)
        x = LeakyReLU(alpha = 0.2)(x)
       
        x = Dense(1)(x)
        x = Activation('sigmoid')(x) 
        
        discriminator_x = Model(inputs = dis_input, outputs = x)
        
        return discriminator_x


class VGG_LOSS(object):

    def __init__(self, image_shape):
        
        self.image_shape = image_shape

    # 用VGG19 计算 高清图和生成的高清图之间的差别
    def vgg_loss(self, y_true, y_pred):
    
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        vgg19.trainable = False
        # Make trainable as False
        for l in vgg19.layers:
            l.trainable = False
        model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        model.trainable = False
    
        return K.mean(K.square(model(y_true) - model(y_pred)))
if __name__=="__main__":
    Generator().summary()
    Discriminator().summary()