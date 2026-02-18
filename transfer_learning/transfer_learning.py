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
from tensorflow.keras import Model , layers # Functional API ile model bağlamak


from tensorflow.keras.applications import MobileNetV2 # önceden ImageNet’te eğitilmiş “hazır özellik çıkarıcı” backbone

from pathlib import Path # klasör içinde dosya aramak için
import os.path # path parçalamak (etiketi klasör adından almak için)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore")

# load data
dataset = "Drug Vision/Data Combined"
image_dir = Path(dataset)
# alt klasörler dahil tüm .jpg dosyalarını bulur
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
plt.tight_layout()  # bosluklari duzeltir
# %% preprocessing

# train test split
train_df, test_df = train_test_split(image_df, test_size = 0.2, random_state=42, shuffle = True)

# data augmentation 
train_generator = ImageDataGenerator(
                    preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input, # mobile net v2 icin on isleme
                    validation_split = 0.2 # egitim verisinin % 20'sini validation icin ayir
    )

test_generator = ImageDataGenerator(
                    preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input
    )

# flow_from_dataframe: DataFrame’den batch üretmek
# egitim verilerini dataframe'den akisa alalim
train_images = train_generator.flow_from_dataframe(
                dataframe=train_df, 
                x_col = "filepath",   # goruntu yolu
                y_col = "label", # hedef etiket, goruntulerin etiketi
                target_size =(224,224), # goruntulerin hedef boyutu
                color_mode = "rgb", # renkli goruntu modu
                class_mode = "categorical", # coklu sinif siniflandirmasi
                shuffle = True, # goruntuleri karistirma
                seed =42, # rastgelelik icin tohum
                subset = "training" # egitim seti
    )

val_images = train_generator.flow_from_dataframe(
                dataframe=train_df, 
                x_col = "filepath",   # goruntu yolu
                y_col = "label", # hedef etiket, goruntulerin etiketi
                target_size =(224,224), # goruntulerin hedef boyutu
                color_mode = "rgb", # renkli goruntu modu
                class_mode = "categorical", # coklu sinif siniflandirmasi
                shuffle = True, # goruntuleri karistirma
                seed =42, # rastgelelik icin tohum
                subset = "validation" # egitim seti
    )

test_images = test_generator.flow_from_dataframe(
                dataframe=test_df, 
                x_col = "filepath",   # goruntu yolu
                y_col = "label", # hedef etiket, goruntulerin etiketi
                target_size =(224,224), # goruntulerin hedef boyutu
                color_mode = "rgb", # renkli goruntu modu
                class_mode = "categorical", # coklu sinif siniflandirmasi
    )
"""
Ya generator preprocess_input kullan,
Ya da model içinde resize/rescale + uygun preprocess kullan.

İkisini birden kullanırsan “çifte preprocessing” yapıp performansı bozabilirsin
"""
# resize (goruntuleri boyutlandir) ve rescale (normalizasyon)
resize_and_rescale = tf.keras.Sequential([
        layers.Resizing(224,224), # goruntuleri 224x224 boyutunda yeniden boyutlandirma
        layers.Rescaling(1.0/255) # normalization
    
    ])


# %% define mobilenet, training

# mobilenet-v2 -> onceden  egitilmis model
pretrained_model= MobileNetV2(
                    input_shape = (224,224,3), # girdilerin yani goruntulerin boyutu
                    # Sadece özellik çıkaran gövde (backbone) kalır.
                    include_top = False, # mobile net siniflandirma katmanini dahil etme
                    weights = "imagenet", # Model, ImageNet'te onceden egitilmis agirliklarla gelir.
                    pooling = "avg" # backbone çıkışındaki feature map’i GlobalAveragePooling ile tek vektöre çevirir.
    )
pretrained_model.trainable = False # MobileNetV2’nin ağırlıklarını eğitim sırasında değiştirme

# callback
checkpoint_path="pharmacuetical_drugs_and_vitamins_classification_model_checkpoint.weights.h5"
checkpoint_callback = ModelCheckpoint(checkpoint_path,
                save_weights_only = True, # modelin sadece agirliklarini kaydet
                monitor = "val_accuracy",
                save_best_only = True # en iyi val_accuracy geldiginde kaydet
                )
# 5 epoch kadar val_accuracy degerinde iyilesme yoksa egitimi durdur 
early_stopping = EarlyStopping(monitor = "val_accuracy",
                               patience = 5,
                               restore_best_weights= True # durduğunda en iyi epoch’un ağırlıklarını geri yükler
                               )

# build model while using pretrained model (mobilenet-v2)
# mobilenet backbone girişini al
inputs = pretrained_model.input

# backbone çıktısını al (feature extractor kısmı)
x = pretrained_model.output

# classification head (senin katmanların)
x = Dense(256, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.2)(x)

# final output layer
outputs = Dense(10, activation="softmax")(x)

# modeli oluştur
model = Model(inputs=inputs, outputs=outputs)

# compile
model.compile(optimizer=Adam(0.0001), loss= "categorical_crossentropy", metrics = ["accuracy"])

# training 
history = model.fit(
    train_images,
    steps_per_epoch = len(train_images),
    validation_data= val_images,
    validation_steps = len(val_images),
    epochs= 10,
    callbacks = [early_stopping, checkpoint_callback]
    )






