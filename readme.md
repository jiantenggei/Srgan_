  

# Srgan_重置的 Srgan 代码训练自己 超分模型
这是复现的SRGAN 网络用于训练自己的超分模型

# 环境要求
| 工具           | 版本  |
| -------------- | ----- |
| python         | 3.8.0 |
| TensorFlow-gpu | 2.5.0 |
| keras          | 2.4.3 |

# 资源地址：
csdn 博客地址： https://no-coding.blog.csdn.net/article/details/121682740

数据集地址：https://pan.baidu.com/s/1UBle5Cu74TRifcAVz14cDg 提取码：luly



# conda虚拟环境一键导入：

```bash
conda env create -f srgan_tf2.5.yaml
```



# How2Train

数据集图片放入dataset 目录下。

```bash
 ├─dataset
    │	└─1.png
    │	└─2.png
    │	└─3.png
```

运行 annotation.py。生成dataset.txt.

**train.py** :

```python
def train(epochs, batch_size, model_save_dir): 
    # 设置dataset.txt 路径
    train_annotation_path = 'dataset.txt'
    #下采样倍数 
    downscale_factor = 4

    #输入图片形状
    hr_shape = (384,384,3)
    #加载数据集
```

然后就可以训练了。

# How2Predict

```python


from pickle import NONE
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from nets.nets import Generator
#图片路径
before_image = Image.open(r"0.jpg")

before_image = before_image.convert("RGB")
gen_model = Generator([None,None,3]) #生成器模型
gen_model.load_weights('loss\gen_model99.h5')
# gen_model.summary()
new_img = Image.new('RGB', before_image.size, (128, 128, 128))
new_img.paste(before_image)
# plt.imshow(new_img)
# plt.show()

new_image = np.array(new_img)/127.5 - 1
# 三维变4维  因为神经网络的输入是四维的
new_image = np.expand_dims(new_image, axis=0)  # [batch_size,w,h,c]
fake = (gen_model.predict(new_image)*0.5 + 0.5)*255
#将np array 形式的图片转换为unit8  把数据转换为图
fake = Image.fromarray(np.uint8(fake[0]))

fake.save("out.png")
titles = ['Generated', 'Original']
plt.subplot(1, 2, 1)
plt.imshow(before_image)
plt.subplot(1, 2, 2)
plt.imshow(fake)
plt.show()
```

# 其他

上采样时，使用keras 自带的UpSampling2D。上采样两次。在net.py 中有亚像素卷积上采样。需要更改的话自己从代码中更改就好。

```python
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
        return tf.compat.v1.depth_to_space(x, scale)
        
    return Lambda(subpixel, output_shape=subpixel_shape)
```



其他问题私信：[1308659229@qq.com](mailto:1308659229@qq.com)

**如果觉得有用清给我点star**

