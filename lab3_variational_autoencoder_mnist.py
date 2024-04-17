import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Завантаження та підготовка датасету MNIST
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Нормалізація вхідних даних
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Решейп даних до вигляду (28, 28, 1)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Побудова моделі варіаційного автокодувальника
latent_dim = 64  # Розмір скороченого представлення (latent representation)

# Енкодер
encoder_inputs = tf.keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(encoder_inputs)
x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation='relu')(x)
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
z = layers.Lambda(lambda inputs: tf.random.normal(tf.shape(inputs)))(z_mean, z_log_var)
encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

# Декодер
latent_inputs = tf.keras.Input(shape=(latent_dim,))
x = layers.Dense(7*7*64, activation='relu')(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)
decoder = tf.keras.Model(latent_inputs, decoder_outputs, name='decoder')

# Автокодувальник
outputs = decoder(encoder(encoder_inputs)[2])
vae = tf.keras.Model(encoder_inputs, outputs, name='vae')

# Визначення функції втрат
reconstruction_loss = tf.keras.losses.binary_crossentropy(encoder_inputs, outputs)
reconstruction_loss *= 28 * 28
reconstruction_loss = tf.reduce_mean(reconstruction_loss)
kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
kl_loss = tf.reduce_mean(kl_loss)
kl_loss *= -0.5
vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)

# Компіляція та навчання моделі
vae.compile(optimizer='adam')
vae.fit(x_train, epochs=10, batch_size=128, validation_data=(x_test, None))

# Візуалізація результатів
decoded_imgs = vae.predict(x_test)

plt.figure(figsize=(10, 4))
for i in range(10):
    # Оригінальне зображення
    plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    
    # Відтворене зображення
    plt.subplot(2, 10, i + 11)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()
