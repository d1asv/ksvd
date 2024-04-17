import numpy as np
import matplotlib.pyplot as plt

def _koch_snowflake_complex(order):
    if order == 0:
        return [complex(0, 0), complex(1, 0)]
    else:
        points = _koch_snowflake_complex(order - 1)
        new_points = []
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            one_third = (p2 - p1) / 3
            two_third = 2 * (p2 - p1) / 3
            new_points.extend([p1, p1 + one_third, p1 + one_third + one_third * np.exp(np.pi * 1j / 3), p1 + two_third])
        new_points.append(points[-1])
        return new_points

def koch_snowflake(order):
    points = _koch_snowflake_complex(order)
    x = np.real(points)
    y = np.imag(points)
    plt.figure(figsize=(8, 8))
    plt.plot(x, y)
    plt.title(f"Koch Snowflake (Order {order})")
    plt.axis('equal')
    plt.show()

# Використання функції для побудови крижинки Коха з рівнем рекурсії 3
koch_snowflake(order=3)
