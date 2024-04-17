import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Завантаження датасету
(x_train, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
# Нормалізація даних
x_train = x_train.astype('float32') / 255
x_train = np.expand_dims(x_train, axis=-1)  # Розширення зображень до третього виміру

# Завдання розмірності вхідного шуму для генератора
latent_dim = 100

# Створення генератора
generator = models.Sequential([
    layers.Dense(7 * 7 * 128, input_dim=latent_dim),
    layers.LeakyReLU(alpha=0.2),
    layers.Reshape((7, 7, 128)),
    layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
    layers.LeakyReLU(alpha=0.2),
    layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
    layers.LeakyReLU(alpha=0.2),
    layers.Conv2D(1, (7, 7), activation='sigmoid', padding='same')
])

# Створення дискримінатора
discriminator = models.Sequential([
    layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
    layers.LeakyReLU(alpha=0.2),
    layers.Dropout(0.4),
    layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'),
    layers.LeakyReLU(alpha=0.2),
    layers.Dropout(0.4),
    layers.Flatten(),
    layers.Dense(1, activation='sigmoid')
])

# З'єднання генератора та дискримінатора в єдину модель GAN
discriminator.compile(optimizer='adam', loss='binary_crossentropy')
discriminator.trainable = False
gan_input = tf.keras.Input(shape=(latent_dim,))
fake_image = generator(gan_input)
gan_output = discriminator(fake_image)
gan = models.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Навчання моделі GAN
batch_size = 128
iterations = 10000
for step in range(iterations):
    noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
    generated_images = generator.predict(noise)
    real_images = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]
    x_combined = np.concatenate([real_images, generated_images])  # Виправлено
    y_combined = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
    d_loss = discriminator.train_on_batch(x_combined, y_combined)
    noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
    y_mislabeled = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, y_mislabeled)
    if step % 100 == 0:
        print(f"Step {step}/{iterations} | Discriminator Loss: {d_loss} | Generator Loss: {g_loss}")

# Візуалізація результату
n = 10  # кількість зображень для візуалізації
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(generated_images[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.show()
