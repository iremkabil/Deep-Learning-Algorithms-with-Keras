# %% load dataset 

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # data augmentation
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras import layers


from tensorflow.keras.applications import MobileNetV2

from pathlib import Path
import os.path

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore")

# load data
dataset = "Drug Vision/Data Combined"
image_dir = Path(dataset)
filepaths = list(image_dir.glob(r"**/*.jpg")) + list(image_dir.glob(r"**/*.png"))

labels = list(map(lambda x: os.path.split(os.path.split(x)[0]) [1], filepaths))

filepaths = pd.Series(filepaths, name = "filepath").astype("str")
labels = pd.Series(labels, name = "label")

image_df = pd.concat([filepaths, labels], axis=1)

# data visualization
random_index = np.random.randint(0,len(image_df), 25)
fig, axes = plt.subplots(nrows = 5, ncols =5, figsize = (11,11))
for i,ax in enumerate(axes.flat):
    ax.imshow(plt.imread(image_df.filepath[random_index[i]]))
    ax.set_title(image_df.label[random_index[i]])
plt.tight_layout()
# %% preprocessing


# train test split

# data augmentation 

# resize (goruntuleri boyutlandir) ve rescale (normalizasyon)

# %% define mobilenet, training

# mobilenet-v2

# callback

# build model while using pretrained model (mobilenet-v2)

# compile



# training 
