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

data_dir = '/Users/earvin/workspaces/GithubProjects/tensorflow/scripts/BOOK_DLI16/Ch14/quickdraw_data'
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

# 建構識別器(discriminator)神經網路
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










