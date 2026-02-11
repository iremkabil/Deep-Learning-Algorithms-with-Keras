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

# discriminator modelini tanimla

# generator modelini tanimla

# %% Create GAN model

# compile

# model olusturma

# %% Train GAN



# %% GAN'lar tarafindan uretilen goruntulerin karsilastirilmasi








