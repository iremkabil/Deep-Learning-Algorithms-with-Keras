# %% load dataset and preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder # etiketleri sayisal formata cevirir
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer # metin verisini sayilara cevirir
from tensorflow.keras.preprocessing.sequence import pad_sequences # dizileri ayni uzunluga getirir
from tensorflow.keras.models import Sequential # kerasta model olusturma sinifi
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings("ignore")

newsgroup = fetch_20newsgroups(subset = "all") # all ile hem egitim hem test verileri yuklenir
X = newsgroup.data
y = newsgroup.target

# Metin verisini tokenize etme ve padding

# Tüm metni tarıyor => Her kelimeyi sayiyor => En sık kelimelere index veriyor
# LSTM tum inputlarin ayni boyutlu olmasini ister.
# 100 kelimeden kisa ise basina 0 ekler. Fazla ise keser.

tokenizer = Tokenizer(num_words = 10000) # num_words = en cok kullanilan kelime sayisi
tokenizer.fit_on_texts(X) # tokenizer'i metin verisi ile fit edelim.
X_sequences =tokenizer.texts_to_sequences(X) # metinleri sayisala cevirir
X_padded = pad_sequences(X_sequences,maxlen = 100) # metinleri ayni uzunluga getirir


# Etiketleri sayisal hale donustur (label encoding)

# y icindeki tum unique siniflari bul => alfabetik siraya koy => sayiya cevir
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Tokenizer → X'i sayıya çevirir
# LabelEncoder → y'yi sayıya çevirir

# Train test split 

X_train, X_test, y_train, y_test = train_test_split(X_padded, y_encoded, test_size= 0.2, random_state=42)

# %% create (build) lstm model

from tensorflow.keras import backend as K
import tensorflow as tf

def f1_macro(y_true, y_pred):
    # y_true: (batch,) int
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.argmax(y_pred, axis=-1)               # (batch,)
    y_pred = tf.cast(y_pred, tf.int32)

    num_classes = 20
    y_true = tf.one_hot(y_true, depth=num_classes)    # (batch, C)
    y_pred = tf.one_hot(y_pred, depth=num_classes)    # (batch, C)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred, axis=0)
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())

    return tf.reduce_mean(f1)


def build_lstm_model():
    model = Sequential()
    
    # katmanlar: embedding + lstm + dropout + output
    
    # input_dim: kelime vektorlerin toplam boyutu
    # output_dim: kelime vektorlerin boyutu
    # input_length her giris metninin uzunlugu
    
    model.add(Embedding(input_dim =10000, output_dim=64, input_length = 100))
    
    # return_sequence: sonuclarin tum zaman adimlari yerine sadece son adimda return etmesi
    model.add(LSTM(units=64, return_sequences=False)) # 64 adet hucre, 
    
    model.add(Dropout(0.5))
    
    model.add(Dense(20, activation = "softmax")) # 20 sinif => 20 noron
    
    # model compile
    model.compile(optimizer="adam",
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy", f1_macro])
    return model
    
# model olusturma 
model = build_lstm_model()
model.summary()

# %% train lstm

# callbacks = early stop
early_stopping = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)

# model train

history = model.fit(X_train, y_train,
                    epochs=5,
                    batch_size=32,
                    validation_split=0.1,
                    callbacks = [early_stopping])
# %% model evaluation

# test veri seti ile degerlendirme

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss: .3f}, Test Accuracy: {accuracy: .3f}")

# history kullanarak accuracy ve loss degerlerini gorsellestirme
plt.figure()

# training loss ve val loss
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label = "Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid("True")

# training accuracy ve val accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label = "Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid("True")






