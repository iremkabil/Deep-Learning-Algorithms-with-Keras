# %% load data and preprocessing
import tensorflow as tf

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from kerastuner import HyperModel, RandomSearch # hyperparameter tuning icin kullanalim

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# veri setini reshape yapalim (28x28) -> (28x28x1)
train_images = train_images.reshape(-1,28,28,1).astype("float32") / 255.0 # -1 -> kendin belirle diyorum
test_images = test_images.reshape(-1,28,28,1).astype("float32") / 255.0

#etiketleri one-hot encoding ile hazirlayalim
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# %% residual block
def residual_block(x, filters, kernel_size = 3, stride = 1):
    
    shortcut = x
    # 1. conv layer
    x= Conv2D(filters, kernel_size = kernel_size, strides = stride, padding="same")(x)
    x = BatchNormalization()(x)
    x= Activation("relu")(x)
    
    # 2. conv layer
    x = Conv2D(filters, kernel_size = kernel_size, strides=stride, padding="same")(x)
    x= BatchNormalization()(x)
    
    # eger giristen gelen verinin boyutu filtre sayisina esit degilse
    if shortcut.shape[-1] !=filters:
        # giris verisinin boyutunu esitlemek icin 1x1 konvolusyon uygulayalim
        shortcut = Conv2D(filters, kernel_size=1, strides=stride,padding="same")(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    # residual baglanti: giris verisi ile cikis verisini toplayalim
    x = Add()([x, shortcut])
    x = Activation("relu")(x)

    return x
# %% create and compile resnet

class ResNetModel(HyperModel):
    def build(self, hp): # hp = hyperparameter tuning icin kullanilacak parametre
    
        inputs = Input(shape = (28,28,1))
    
        # 1. conv layer
        x = Conv2D(filters = hp.Int("initial_filters", min_value = 32,max_value = 128,step =32),
                  kernel_size=3, padding="same" ,activation="relu")(inputs)
        x = BatchNormalization()(x)
        
        # residual blok ekleyelim
        for i in range(hp.Int("num_blocks", min_value = 1, max_value = 3, step=1)):
            x = residual_block(x, hp.Int("res_filters_" + str(i), min_value=32,max_value=128,step=32))
        
        # siniflandirma katmani
        x = Flatten()(x)
        x = Dense(128, activation = "relu")(x)
        outputs = Dense(10, actvation = "softmax")(x)
        
        model = Model(inputs, outputs)
        
        # compile model
        model.compile(optimizer =Adam(hp.Float("Learing_rate", min_value= 1e-4,max_value=1e-2,samping = "LOG")),
                      loss = "categorical_crossentropy",
                      metrics = ["accuracy"])
        return model
        




# %% training, hyperparameter tuning and model evaluation











