
from ctypes import util
from unicodedata import name
from nets.nets import Discriminator,Generator,VGG_LOSS
import random
from keras.models import Model
from keras.layers import Input
from tqdm import tqdm
import numpy as np
from utils import PSNR, show_result

from dataloader import SRganDataset
import tensorflow as tf  
tf.config.experimental_run_functions_eagerly(True)
#得到完整的gan 网络
def get_gan(discriminator, shape, generator, optimizer, vgg_loss):

    discriminator.trainable = False
    gan_input = Input(shape=shape)
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=[x,gan_output])
    gan.compile(loss=[vgg_loss, "binary_crossentropy"],
                loss_weights=[1., 1e-3],
                optimizer=optimizer,metrics=[PSNR])

    return gan


def train(epochs, batch_size, model_save_dir):

    train_annotation_path = 'dataset.txt'
    #下采样倍数
    downscale_factor = 4

    #输入图片形状
    hr_shape = (384,384,3)
    #加载数据集
    with open(train_annotation_path, encoding='utf-8') as f:
         train_lines = f.readlines()
    #计算 生成图片 和 原高清图 之间的loss
    loss = VGG_LOSS(hr_shape) 
    #打乱 
    random.shuffle(train_lines)
    batch_count = int(len(train_lines)/ batch_size)
    lr_shape = (hr_shape[0]//downscale_factor, hr_shape[1]//downscale_factor, hr_shape[2])
    
    generator = Generator(lr_shape)
    discriminator = Discriminator(hr_shape)

    optimizer =tf.optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    generator.compile(loss=loss.vgg_loss, optimizer=optimizer)
    discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)
    gen                 = SRganDataset(train_lines, lr_shape[:2], hr_shape[:2], batch_size)
    gan = get_gan(discriminator, lr_shape, generator, optimizer,loss.vgg_loss)
    loss_file = open(model_save_dir + 'losses.txt' , 'w+')
    loss_file.close()
    
    for epoch in range(0,epochs):
        print ('-'*15, 'Epoch %d' % epoch, '-'*15)
        with tqdm(total=batch_count,desc=f'Epoch {epoch + 1}/{epochs}',postfix=dict,mininterval=0.3) as pbar:
            for iteration, batch in enumerate(gen):
                if iteration >= batch_count:
                    break
                imgs_lr, imgs_hr        = batch
                #生成器生成图片
                gen_img = generator.predict(imgs_lr)

                real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
                fake_data_Y = np.random.random_sample(batch_size)*0.2
                
                discriminator.trainable = True
                
                d_loss_real = discriminator.train_on_batch(imgs_hr, real_data_Y)
                d_loss_fake = discriminator.train_on_batch(gen_img, fake_data_Y)
                discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

            

                gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
                discriminator.trainable = False
                gan_loss = gan.train_on_batch(imgs_lr, [imgs_hr,gan_Y])
                pbar.set_postfix(**{'G_loss'        : gan_loss[0] , 
                                    'D_loss'        : discriminator_loss,
                                    'PSNR'          : gan_loss[4]
                                    },)
                pbar.update(1)  
            print("discriminator_loss : %f" % discriminator_loss)
            print("gan_loss :", gan_loss)
            gan_loss = str(gan_loss)
            
            loss_file = open(model_save_dir + 'losses.txt' , 'a')
            loss_file.write('epoch%d : gan_loss = %s ; discriminator_loss = %f\n' %(epoch, gan_loss, discriminator_loss) )
            loss_file.close()

            
            show_result(epoch,generator,imgs_lr,imgs_hr)
            
            generator.save(model_save_dir + 'gen_model%d.h5' % epoch)
            discriminator.save(model_save_dir + 'dis_model%d.h5' % epoch)

         
if __name__ =="__main__":
    train(epochs=100,batch_size=4,model_save_dir='loss/')
