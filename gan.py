# %% mnist veri sesti yukleme
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, Flatten, Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.optimizers import Adam


from tqdm import tqdm # for dongusune ilerleme sayaci ekleme

import warnings
warnings.filterwarnings("ignore")

(x_train, _) ,(_, _) = mnist.load_data()
# normalizasyon
x_train = x_train / 255.0
# boyutlarin ayarlanmasi (28x28) -> (28,28,1)
x_train = np.expand_dims(x_train, axis=-1)
# %% Create Discriminator and Generator

# gan parametreli
z_dim = 100 # gurultu vektorunun bouyu

# discriminator modelini tanimla
def build_discriminator():
    model = Sequential()
    
    # Conv2D: 64 filtre, 3x3 cekirdek (kernel), stride = 2, padding=same, activation =LeakyReLU
    model.add(Conv2D(64, kernel_size = 3, strides = 2,padding ="same"), input_shape = (28,28,1))
    model.add(LeakyReLU(alpha = 0.2))
    # Conv2D: 128 filtre, 3x3 cekirdek (kernel), stride = 2, padding=same, activation =LeakyReLU
    model.add(Conv2D(128, kernel_size = 3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    # flatten -> output layer
    model.add(Flatten())
    model.add(Dense(1, activation ="sigmoid")) # output layer
    
    #compile
    model.compile(loss = "binary_crossentropy", optimizer=Adam(0.0002, 0.5),metrics = ["accuracy"])
    
# generator modelini tanimla

# %% Create GAN model

# compile

# model olusturma

# %% Train GAN



# %% GAN'lar tarafindan uretilen goruntulerin karsilastirilmasi








