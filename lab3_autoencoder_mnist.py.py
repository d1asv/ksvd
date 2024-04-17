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

# Розмір тренувального та тестового набору
print('Розмір тренувального набору:', x_train.shape)
print('Розмір тестового набору:', x_test.shape)

# Побудова моделі автокодувальника
latent_dim = 64  # Розмір скороченого представлення (latent representation)

# Енкодер
encoder_inputs = tf.keras.Input(shape=(28, 28, 1))
x = layers.Flatten()(encoder_inputs)
x = layers.Dense(256, activation='relu')(x)
latent_space = layers.Dense(latent_dim, activation='relu')(x)
encoder = tf.keras.Model(encoder_inputs, latent_space, name='encoder')

# Декодер
decoder_inputs = tf.keras.Input(shape=(latent_dim,))
x = layers.Dense(256, activation='relu')(decoder_inputs)
x = layers.Dense(28*28, activation='sigmoid')(x)
outputs = layers.Reshape((28, 28, 1))(x)
decoder = tf.keras.Model(decoder_inputs, outputs, name='decoder')

# Автокодувальник
autoencoder_inputs = tf.keras.Input(shape=(28, 28, 1))
encoded = encoder(autoencoder_inputs)
decoded = decoder(encoded)
autoencoder = tf.keras.Model(autoencoder_inputs, decoded, name='autoencoder')

autoencoder.summary()

# Компіляція та навчання моделі
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
history = autoencoder.fit(x_train, x_train,
                          epochs=10,
                          batch_size=128,
                          shuffle=True,
                          validation_data=(x_test, x_test))

# Побудова графіка функції втрат
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
