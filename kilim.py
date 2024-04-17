import numpy as np
import matplotlib.pyplot as plt

def sierpinski_carpet(order, size):
    # Перевірка базового випадку
    if order == 0:
        return np.ones((size, size))
    
    # Зменшення розміру зображення в 3 рази
    small_square = sierpinski_carpet(order - 1, size // 3)
    
    # Створення пустої карти
    carpet = np.zeros((size, size))
    
    # Розташування маленьких квадратів всередині великого
    for i in range(3):
        for j in range(3):
            if not (i == 1 and j == 1):
                carpet[i * size // 3: (i + 1) * size // 3, j * size // 3: (j + 1) * size // 3] = small_square
    
    return carpet

# Визначення порядку і розміру килиму
order = 5
size = 243  # 3 ** 5

# Створення килиму Серпінського
carpet = sierpinski_carpet(order, size)

# Відображення килиму Серпінського
plt.figure(figsize=(6, 6))
plt.imshow(carpet, cmap='binary', interpolation='none')
plt.axis('off')
plt.show()
