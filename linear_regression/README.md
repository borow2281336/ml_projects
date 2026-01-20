# Diamonds Price Prediction (Linear Regression)

Небольшой учебный проект: предсказание **цены бриллианта (`price`)** по его характеристикам из датасета **Diamonds** с помощью **линейной регрессии** и регуляризации (**Ridge / Lasso**).


---

## Что сделано в ноутбуке

1. **Загрузка данных**

   * Датасет `diamonds` загружается через `seaborn.load_dataset("diamonds")`.

2. **Быстрый EDA**

   * Просмотр структуры, размеров, пропусков.
   * Описательная статистика (`describe()`).
   * Корреляции по числовым признакам + heatmap.

3. **Подготовка признаков**

   * Целевая переменная: `price`.
   * Числовые признаки берутся как есть.
   * Категориальные признаки (`cut`, `color`, `clarity`) кодируются через **OneHotEncoder** (`drop='first'`).
   * Итоговый датасет собирается конкатенацией числовых и one-hot признаков.

4. **Обучение моделей**

   * Train/Test split: **70/30**
   * Масштабирование признаков: **StandardScaler**
   * Модели:

     * `LinearRegression`
     * `Ridge(alpha=10)`
     * `Lasso(alpha=10)`
     * `LassoCV` (подбор `alpha` по кросс-валидации)

5. **Оценка качества**

   * Метрика: **MSE (Mean Squared Error)** на train и test.

---

## Результаты (по текущему запуску ноутбука)

* Linear Regression:

  * MSE train ≈ **1,268,454.96**
  * MSE test ≈ **1,314,517.44**

* Ridge (alpha=10):

  * MSE train ≈ **1,268,581.41**
  * MSE test ≈ **1,315,226.66**

* Lasso (alpha=10):

  * MSE train ≈ **1,330,335.87**
  * MSE test ≈ **1,379,315.77**

* Lasso (после подбора / alpha=0.1 в ноутбуке):

  * MSE train ≈ **1,268,463.07**
  * MSE test ≈ **1,314,152.59**

---

## Как запустить

### Вариант 1: локально (Jupyter)

1. Клонируй репозиторий:

   * `git clone <repo>`
   * `cd <repo>`

2. Создай виртуальное окружение и установи зависимости:

   * `python -m venv .venv`
   * Windows: `.venv\Scripts\activate`
   * macOS/Linux: `source .venv/bin/activate`
   * `pip install -U pip`
   * `pip install pandas numpy seaborn matplotlib scikit-learn jupyter`

3. Запусти ноутбук:

   * `jupyter notebook`
   * Открой `Lin_regr_Diamonds.ipynb` и выполни **Run All**.

### Вариант 2: Google Colab

* Загрузи `.ipynb` в Colab и запусти все ячейки.

---

## Стек

* Python
* pandas, numpy
* seaborn, matplotlib
* scikit-learn


