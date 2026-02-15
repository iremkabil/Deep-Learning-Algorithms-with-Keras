# %% load data and preprocessing
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

from skimage.metrics import structural_similarity as ssim

(x_train, _),(x_test, _) = fashion_mnist.load_data()

# data normalization
x_train = x_train.astype("float32")/255.0
x_test = x_test.astype("float32")/255.0

# gorsellestirme
plt.figure()
for i in range(4):
    plt.subplot(1,4, i+1)
    plt.imshow(x_train[i], cmap = "gray")
    plt.axis("off")
plt.show()

# veriyi duzlestir 28x28 boyutundaki goruntuleri 784 boyutuna bir vektore cevir
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# %% encoder ve decoder mimarisi olusturma ve bunlari autoencoders mimarisine ekleme

# autoencoders icin model parametrelerinin tanimlanmasi
input_dim = x_train.shape[1] #giris boyutu (784)
encoding_dim = 64 # latent boyutu (daha kucuk boyut)

# encoder kisminin insa edilmesi
input_image =Input(shape = (input_dim,)) # girdi boyutunu belirliyoruz (784)
encoded = Dense(256, activation="relu")(input_image) # ilk gizli katman(256 noron)
encoded = Dense(128, activation = "relu")(encoded) # ilk gizli katman(256 noron)
encoded = Dense(encoding_dim, activation="relu")(encoded) # sıkıstırma katmani(64 boyut)

# decoder kisminin insa edilmesi
decoded = Dense(128, activation="relu")(encoded) # ilk genisletme katmani
decoded = Dense(256, activation="relu")(decoded) # ikinci genisletme katmani
decoded = Dense(input_dim, activation = "sigmoid")(decoded) #cikti katmani (784 boyutlu)

# autoencoders olusturma = ensoder + decoder
autoencoder = Model(input_image, decoded ) # giristen ciktiya tum yapiya tanimliyoruz
# modelin compile edilmesi
autoencoder.compile(optimizer = Adam(), loss = "binary_crossentropy")

# modelin train edilmesi
history = autoencoder.fit(x_train, x_train, # girdi ve hedef ayni deger olmali (otonom ogrenme)
                          epochs= 50,
                          batch_size =64,
                          shuffle =True, # egitim verilerini karistir
                          validation_data = (x_test, x_test),
                          verbose= 1)

# %% model testi

# modeli encoder ve decoder olarak ikiye ayir
encoder = Model(input_image, encoded)

# decoder = encoded
encoded_input = Input(shape= (encoding_dim,))
decoder_layer1 = autoencoder.layers[-3](encoded_input)
decoder_layer2 = autoencoder.layers[-2](decoder_layer1)
decoder_output = autoencoder.layers[-1](decoder_layer2)

decoder =Model(encoded_input, decoder_output)

# test verisi ile encoder ve decoder ile sıkıstırma ve yeniden yapilandirma islemi
encoded_images = encoder.predict(x_test) # latent temsili elde ederiz
decoded_images = decoder.predict(encoded_images) # latent temsillerini orijinal forma geri cevir

# orijinal ve yeniden yapilandirilmis (decoded_images) goruntuleri gorsellestir
n = 10 # 10 samples

plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28), cmap="gray") # orijinal goruntu boyutuna ceviriyoruz
    ax.get_xaxis().set_visible(False) # x eksenini gizliyoruz
    ax.get_yaxis().set_visible(False) 
    
    # decoded edilmis yani yeniden yapilandirilmis goruntu
    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_images[i].reshape(28,28),cmap="gray")
    ax.get_xaxis().set_visible(False) # x eksenini gizliyoruz
    ax.get_yaxis().set_visible(False) 
plt.show()















