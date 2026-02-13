# %% mnist veri seti yukleme
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LeakyReLU # relu varyasyonudur. girdi +x ise cikti +x olur, girdi -x ise cikti -ax olur. a = cok kucuk bir sayi
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.optimizers import Adam

from tqdm import tqdm # for dongusune ilerleme sayaci ekleme

import warnings 
warnings.filterwarnings("ignore")

# mnist veri setini yukle
(x_train, _ ), ( _, _ ) = mnist.load_data() # sadece training yukle

# normalizasyon
x_train = x_train / 255.0

# boyutlarin ayarlanmasi (28x28) -> (28,28,1)
x_train = np.expand_dims(x_train, axis=-1)
# %% Create Discriminator and Generator

# gan parametreleri
z_dim = 100 # gurultu vektorunun boyutu

# discriminator modeline tanimla
def build_discriminator():
    
    model = Sequential()
    # Conv2D: 64 filtre, 3x3 cekirdek (kernel), stride = 2, padding = same, activation= LeakyRelu
    model.add(Conv2D(64, kernel_size = 3, strides = 2, padding = "same", input_shape = (28,28,1)))
    model.add(LeakyReLU(alpha = 0.2))
    # Conv2D: 128 filtre, 3x3 cekirdek (kernel), stride = 2, padding = same, activation= LeakyRelu
    model.add(Conv2D(128, kernel_size=3, strides = 2, padding="same"))
    model.add(LeakyReLU(alpha = 0.2))
    # flatten -> output layer
    model.add(Flatten()) # goruntuyu tek boyutlu vektore donusturur
    model.add(Dense(1, activation = "sigmoid")) # output layer 
    # compile 
    model.compile(loss = "binary_crossentropy", optimizer = Adam(0.0002, 0.5), metrics = ["accuracy"])
    
    return model

# generator modelini tanimla
def build_generator():
    
    model = Sequential()
    
    model.add(Dense(7*7*128, input_dim=z_dim)) # gurultu vektorlerinden yuksek boyutlu uzaya donusum
    model.add(LeakyReLU(alpha = 0.2))
    model.add(Reshape((7,7,128))) # cikisi (7x7x128) olacak sekilde ayarliyoruz
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(64, kernel_size=3, strides = 2, padding = "same"))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(1, kernel_size = 3, strides = 2, padding = "same", activation = "tanh"))
    
    return model
    
# %% Create GAN model

# gan modeli olusturma
def build_gan(generator, discriminator):
    
    discriminator.trainable = False # discriminator egitilemez
    
    model = Sequential()
    model.add(generator) # gan yapisina ilk olarak generator ekliyoruz
    model.add(discriminator) # gan yapisina discriminator ekle
    model.compile(loss="binary_crossentropy", optimizer = Adam(0.0002, 0.5))
    
    return model
   
discriminator = build_discriminator() # discriminator modeli olustur
generator = build_generator() # generator modelini olustur 
gan = build_gan(generator, discriminator) # gan modeli olustur

print(gan.summary())
# %% Train GAN
epochs = 10000 # toplam epoch sayisi
batch_size = 64 
half_batch = batch_size // 2

# egitim dongusu
for epoch in tqdm(range(epochs), desc = "Training Process"): # tqdm ile ilerleme cubugu eklenir.
    
    # gercek veriler ile discriminator egitimi yapilacak
    idx = np.random.randint(0, x_train.shape[0], half_batch) # x_train icerisinden rastgele 32 adet veri sec
    real_images = x_train[idx] # gercek goruntuler
    real_labels = np.ones((half_batch, 1)) # gercek etiketler = 1
    
    # fake verileri (generator'in urettigi) ile discriminator egitimi
    noise = np.random.normal(0, 1, (half_batch, z_dim)) # gurultu vektorleri
    fake_images = generator.predict(noise, verbose = 0) # uretilen goruntuler
    fake_labels = np.zeros((half_batch, 1)) # sahte etiketler = 0 
    
    # update discriminator
    d_loss_real = discriminator.train_on_batch(real_images, real_labels) # gercek verilerle kayip hesaplama
    d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels) # sahte verilerle kayip hesaplama
    d_loss = np.add(d_loss_real, d_loss_fake) * 0.5 # ortalama kayip
    
    # train gan
    noise = np.random.normal(0, 1, (batch_size, z_dim)) # gurultu vektorleri
    valid_y = np.ones((batch_size, 1)) # dogru etiketler
    g_loss = gan.train_on_batch(noise, valid_y) # gan'in icinde bulunan generator training
        
    if epoch % 100 == 0:
        print(f"\n{epoch}/{epochs} D loss: {d_loss[0]}, G loss: {g_loss}")
        

# %% GAN'lar tarafindan uretilen goruntulerin karsilastirilmasi

# uretilen goruntuleri gorsellestirme
def plot_generated_images(generator, epoch, examples = 10, dim=(1,10)):
    
    noise = np.random.normal(0, 1, (examples, z_dim)) # gurultu vektorleri
    gen_images = generator.predict(noise, verbose = 0) # uretilen goruntulerimiz
    gen_images = 0.5*gen_images + 0.5
    
    plt.figure(figsize = (10,1))
    for i in range(gen_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1) # alt grafikleri olusturma
        plt.imshow(gen_images[i, :,:,0], cmap = "gray") # goruntuyu gri tonlama olarak goster
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()

# uretilen goruntuleri plot ettirme
plot_generated_images(generator, epochs)


























