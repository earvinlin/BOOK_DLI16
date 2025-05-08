import numpy as np
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout
from tensorflow.keras.layers import BatchNormalization, Flatten
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D
from tensorflow.keras.optimizers import RMSprop

import pandas as pd
from matplotlib import pyplot as plt

# 下面路徑僅適用於公司的mac-mini
#data_dir = '/Users/earvin/workspaces/GithubProjects/BOOK_DLI16/SAMPLE_CODES/Ch14/quickdraw_data'
data_dir = '/Users/earvin/workspaces/SOURCE_DATA/深度學習的16堂課(F1383)/Ch14/quickdraw_data'
input_images = data_dir + '/apple.npy'

print("data_dir= ", data_dir)

data = np.load(input_images)


# 查看資料集內容
#print(data.shape) # --> (144722, 784)
#print(data[4242])

data = data / 255
data = np.reshape(data, (data.shape[0], 28, 28, 1))
img_w, img_h = data.shape[1:3]
print("data.shape= ", data.shape)

plt.imshow(data[4242,:,:,0], cmap='Greys')
plt.show()

# 14.3 建構識別器(discriminator)神經網路
def build_discriminator(depth=64, p=0.4) :
#   定義輸入
    image = Input((img_w, img_h, 1))
#   卷積層
    conv1 = Conv2D(depth*1, 5, strides=2,
            padding='same', activation='relu')(image)
    conv1 = Dropout(p)(conv1)

    conv2 = Conv2D(depth*2, 5, strides=2,
            padding='same', activation='relu')(conv1)
    conv2 = Dropout(p)(conv2)

    conv3 = Conv2D(depth*4, 5, strides=2,
            padding='same', activation='relu')(conv2)
    conv3 = Dropout(p)(conv3)

    conv4 = Conv2D(depth*8, 5, strides=1,
            padding='same', activation='relu')(conv3)
    
    conv4 = Flatten()(Dropout(p)(conv4))

#   輸出層
    prediction = Dense(1, activation='sigmoid')(conv4)

#   定義模型
    model = Model(inputs=image, outputs=prediction)

    return model

discriminator = build_discriminator()

discriminator.summary()

# 編譯鑑別器
"""
discriminator.compile(loss='binary_crossentropy',
        optimizer=RMSprop(lr=0.0008, decay=6e-8, clipvalue=1.0),
        metrics=['accuracy'])
"""
discriminator.compile(loss='binary_crossentropy', 
        optimizer=RMSprop(lr=0.0008, decay=6e-8, clipvalue=1.0), 
        metrics=['accuracy'])

z_dimensions = 32


# 14.4 建構生成器(generator)神經網路
def build_generator(latent_dim=z_dimensions, depth=64, p=0.4):
    # 定義輸入
    noise = Input((latent_dim,))
    
    # 第 1 密集層
    dense1 = Dense(7*7*depth)(noise)
    dense1 = BatchNormalization(momentum=0.9)(dense1) # default momentum for moving average is 0.99
    dense1 = Activation(activation='relu')(dense1)
    dense1 = Reshape((7,7,depth))(dense1)
    dense1 = Dropout(p)(dense1)
    
    # 反卷積層
    conv1 = UpSampling2D()(dense1)
    conv1 = Conv2DTranspose(int(depth/2), kernel_size=5, padding='same', activation=None,)(conv1)
    conv1 = BatchNormalization(momentum=0.9)(conv1)
    conv1 = Activation(activation='relu')(conv1)
    
    conv2 = UpSampling2D()(conv1)
    conv2 = Conv2DTranspose(int(depth/4), kernel_size=5, padding='same', activation=None,)(conv2)
    conv2 = BatchNormalization(momentum=0.9)(conv2)
    conv2 = Activation(activation='relu')(conv2)
    
    conv3 = Conv2DTranspose(int(depth/8), kernel_size=5, padding='same', activation=None,)(conv2)
    conv3 = BatchNormalization(momentum=0.9)(conv3)
    conv3 = Activation(activation='relu')(conv3)
    
    # 輸出層
    image = Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(conv3)
    
    # 定義模型   
    model = Model(inputs=noise, outputs=image)
    
    return model

    generator = build_generator()

    generator.summary()





