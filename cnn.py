# %% veri setini iceriye aktar ve preprocessing: normalizasyon, one hot encoding
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical # encoding
from tensorflow.keras.models import Sequential   #sirali model
from tensorflow.keras.layers import Conv2D, MaxPooling2D  # feature extraction
from tensorflow.keras.layers import Flatten, Dense, Dropout  #classification
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator # data augmentation

from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore")

# load cifar10
(x_train,y_train), (x_test,y_test) = cifar10.load_data()

# gorsellestirme
class_labels = ["Airplane","Automobile","Bird","Cat","Deer","Dog","Frog","Horse","Ship","Truck"]

# bazi goruntuleri ve etiketlerini gorsellestir
fig, axes = plt.subplots(1, 5, figsize=(15,10))

for i in range(5):
    axes[i].imshow(x_train[i])
    label = class_labels[int(y_train[i])]
    axes[i].set_title(label)
    axes[i].axis("off")
plt.show()

# veri seti normalizasyonu
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255

#one-hot encoding
y_train = to_categorical(y_train, 10) # 10 class var
y_test = to_categorical(y_test, 10)


# %% Veri arttirma Data Augmentation



# %% Create, compile and train model



# %% Test model and evaluate performance


