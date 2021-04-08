#
#
#
#
# %%
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer, MinMaxScaler, minmax_scale
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Lambda, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras import backend as kb

from tensorflow.keras.datasets import cifar10
(X_train, _), (X_test, labels_test) = cifar10.load_data()
X_train = X_train.astype('float32').reshape(-1,32*32*3) / 255.
X_test = X_test.astype('float32').reshape(-1,32*32*3) / 255.
labels_name = {0: 'airplane',
    2: 'bird',
    1: 'automobile',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'}

# %% 
original_dim = 32*32*3
intermediate_dim = 256
latent_dim = 128

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = kb.random_normal(shape=(kb.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=0.1)
    return z_mean + kb.exp(z_log_sigma) * epsilon

# %%
inputs = tf.keras.Input(shape=(original_dim,))
h = Reshape((32,32,3))(inputs)
h = Conv2D(32, (3,3), activation='relu', padding='same')(h)
h = Conv2D(32, (3,3), activation='relu', padding='same')(h)
h = MaxPooling2D((2,2))(h)
h = Conv2D(16, (3,3), activation='relu', padding='same')(h)
h = MaxPooling2D((2,2))(h)
h = Conv2D(8, (3,3), activation='relu', padding='same')(h)
h = Flatten()(h)
h = tf.keras.layers.Dense(intermediate_dim, activation='relu')(h)
z_mean = tf.keras.layers.Dense(latent_dim)(h)
z_log_sigma = tf.keras.layers.Dense(latent_dim)(h)

z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_sigma])

encoder = tf.keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

encoder.summary()

# %%

# Create decoder
latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
x = tf.keras.layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
x = Dense(512)(x) # activate ??

x = Reshape((8,8,8))(x)
x = Conv2DTranspose(16, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
x = Conv2DTranspose(32, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
x = Conv2DTranspose(32, (3,3), activation='relu', padding='same')(x)
x = Conv2DTranspose(3, (3,3), activation='sigmoid', padding='same')(x)
outputs = Flatten()(x)
decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')
decoder.summary()
# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = tf.keras.Model(inputs, outputs, name='vae_mlp')
# %%
reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, outputs)
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_sigma - kb.square(z_mean) - kb.exp(z_log_sigma)
kl_loss = kb.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = kb.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

# %%
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, min_delta=0.30)
batch_size=32
vae.fit(X_train, X_train,
        epochs=10,
        batch_size=batch_size,
        callbacks=[early_stopping],
        validation_data=(X_test, X_test))

encoder = Model(inputs, vae.layers[1].output)
decoder = Model(vae.layers[2].input, vae.layers[2].output)
# %% predict img 

decoded_imgs = vae.predict(X_test)
import matplotlib.pyplot as plt

decoded_imgs_classified=np.zeros((10,1000,32,32,3))
original_imgs_classified=np.zeros((10,1000,32,32,3))
for i in range(10): # it should be better this way if each category has the same number of items
    decoded_imgs_classified[i,:,:,:,:] = decoded_imgs[labels_test.reshape(-1,)==i].reshape(-1,32,32,3)
    original_imgs_classified[i,:,:,:,:] = X_test[labels_test.reshape(-1,)==i].reshape(-1,32,32,3)

# %%  plot 10 classes
n = 10  # how many images we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(original_imgs_classified[i,0])
    plt.gray()
    ax.set_xticks([])
    ax.set_yticks([])

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs_classified[i,0])
    plt.gray()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(labels_name[i])
#plt.savefig('images/cifar10_ae/cifar10_reconstructed_all.png', dpi=120)
plt.show()


# %% 1 class
n = 10  # how many images we will display
fig = plt.figure(figsize=(20, 4))
selected_class = 0 # the class to be shown
for i in range(n):
    # display original
    ax = fig.add_subplot(2, n, i + 1)
    plt.imshow(original_imgs_classified[selected_class,i])
    plt.gray()
    ax.set_xticks([])
    ax.set_yticks([])

    # display reconstruction
    ax = fig.add_subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs_classified[selected_class,i])
    plt.gray()
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle(labels_name[selected_class])
#plt.savefig('images/cifar10_ae/cifar10_reconstructed_'+str(selected_class)+'.png', dpi=120)
plt.show()

# %% calc loss
bce = tf.keras.losses.BinaryCrossentropy()
mse = tf.keras.losses.MeanSquaredError()
loss_test = np.zeros(10)
for i in range(10):
    loss_test[i] = bce(original_imgs_classified[i].reshape(-1,32*32*3),
        decoded_imgs_classified[i].reshape(-1,32*32*3)).numpy()

# %% bce loss
fig = plt.figure(figsize=(10,7.5))
ax = fig.add_subplot(111)

ax.bar(range(10),loss_test)
ax.set_xticks(range(10))
ax.set_xticklabels(labels_name.values())

ax.set_ylabel(ylabel='Binary Cross Entropy', fontsize=15)

ax.tick_params(labelsize=15)
ax.set_title('CIFAR10', fontsize=18)

fig.autofmt_xdate()

#plt.savefig('images/cifar10_ae/cifar10_ae_bce.png',dpi=120)
plt.show()

# %%
n = 15  # figure with 15x15 clothes
img_size = 32
figure = np.zeros((img_size * n, img_size * n, 3))
# We will sample n points within [-15, 15] standard deviations
grid_x = np.linspace(-30, 30, n)
grid_y = np.linspace(-30, 30, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.zeros(latent_dim)
        z_sample[0] = xi
        z_sample[1] = yi
        x_decoded = decoder.predict(z_sample.reshape((-1, latent_dim)))
        img = x_decoded[0].reshape(img_size, img_size, 3)
        figure[i * img_size: (i + 1) * img_size,
               j * img_size: (j + 1) * img_size] = img

plt.figure(figsize=(10, 10))
plt.imshow(figure)
#plt.savefig('images/cifar10_vae/cifar10_vae_generative.png', dpi=120)
plt.show()
# %% 
test_sample = X_test[123].reshape(-1,3072)
n = 10
fig = plt.figure(figsize=(10,4))

test_sample_encoded = encoder.predict(test_sample)
test_mean = test_sample_encoded[0]
test_log_sigma = test_sample_encoded[1]
ax = plt.subplot(1, n+1, 1)
plt.imshow(test_sample.reshape(32,32,3))
ax.set_xticks([])
ax.set_yticks([])
for i in range(n):
    ax = plt.subplot(1, n+1, i+2)
    plt.imshow(vae.predict(test_sample).reshape(32,32,3))
    plt.gray()
    ax.set_xticks([])
    ax.set_yticks([])

# %%
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(range(latent_dim),test_mean.reshape(128))
plt.show()
#ax.plot(len(test_mean),test_log_sigma)

# %%
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(range(latent_dim),test_log_sigma.reshape(128))
plt.show()
# %%
