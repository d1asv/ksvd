import pandas as pd

# Зчитування даних з CSV файлу
df = pd.read_csv('bestsellers_with_categories.csv')

# Виведення схеми даних
print("Схема даних:")
print(df.dtypes)
print()

# Виведення назв стовпців і типів даних
print("Назви стовпців і типи даних:")
print(df.dtypes)
print()

# Виведення перших п'яти рядків
print("Перші п'ять рядків:")
print(df.head())
print()

# Виведення першого рядка
print("Перший рядок:")
print(df.iloc[0])
print()

# Опис датасету
print("Опис датасету:")
print(df.describe())
print()

# Список стовпців
print("Список стовпців:")
print(df.columns.tolist())
print()

# Загальна кількість рядків даних
print("Загальна кількість рядків даних:", len(df))
print()

# Заміна непридатних значень на NaN та типів даних на числовий
df = df.replace('NaN', pd.NA)
df[['User Rating', 'Reviews', 'Price']] = df[['User Rating', 'Reviews', 'Price']].apply(pd.to_numeric, errors='coerce')

# Заміна пропущених значень на середні за стовпцем
df[['User Rating', 'Reviews', 'Price']] = df[['User Rating', 'Reviews', 'Price']].fillna(df[['User Rating', 'Reviews', 'Price']].mean())

# Зміна назв всіх колонок
df.columns = ['name', 'author', 'user_rating', 'reviews', 'price', 'year', 'genre']

# Знайдення автора з найвищим рейтингом
print("Автор з найвищим рейтингом:")
print(df[df['user_rating'] == df['user_rating'].max()]['author'])
print()

# Знайдення автора з найнижчим рейтингом
print("Автор з найнижчим рейтингом:")
print(df[df['user_rating'] == df['user_rating'].min()]['author'])
print()

# Знайдення автора з найбільшою кількістю рецензій
print("Автор з найбільшою кількістю рецензій:")
print(df[df['reviews'] == df['reviews'].max()]['author'])
print()

# Побудова гістограми рейтингу 10 найкращих книг
print("Гістограма рейтингу 10 найкращих книг:")
top_10_books = df.nlargest(10, 'user_rating')
top_10_books['user_rating'].hist()
plt.show()

# Побудова матриці кореляції
print("Матриця кореляції:")
print(df[['user_rating', 'reviews', 'price']].corr())
