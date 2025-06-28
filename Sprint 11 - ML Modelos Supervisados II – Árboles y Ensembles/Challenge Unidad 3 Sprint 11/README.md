# 🧠 ML Toolbox

**ml_Toolbox** es una colección de utilidades para la **preparación de datos y selección de variables** en problemas de regresión desarrollada como parte de un Team Challenge en el bootcamp de Data Science en The Bridge.

Su propósito es acelerar la fase exploratoria y ofrecer herramientas estandarizadas para tomar decisiones informadas antes de entrenar modelos de machine learning.

---

## 📂 Contenido del repositorio

```
├── toolbox_ML.py # Módulo principal con funciones reutilizables
├── Team_Challenge_ToolBox.ipynb # Notebook demostrativo con ejemplo de uso
├── data/ # Carpeta con datasets de prueba o ejemplo
│ └── Marketing-Customer-Analysis.csv # (Ejemplo) Dataset para validar funciones
├── README.md # Este archivo :)
```

---

## 🧰 Funcionalidades del módulo

El script `toolbox_ML.py` ofrece una serie de funciones divididas en dos bloques principales:

### 🔎 Exploración y análisis

| Función              | Descripción |
|----------------------|-------------|
| `describe_df()`      | Analiza el DataFrame: tipos, nulos, cardinalidad, únicos |
| `tipifica_variables()` | Clasifica variables como numéricas, categóricas, binarias o discretas |

### 📈 Selección y visualización de variables

| Función                      | Descripción |
|------------------------------|-------------|
| `get_features_num_regression()`   | Filtra variables numéricas relacionadas con el target (correlación y p-value) |
| `plot_features_num_regression()`  | Representa gráficamente las relaciones con `seaborn.pairplot()` |
| `get_features_cat_regression()`   | Detecta variables categóricas significativas usando ANOVA / Mann-Whitney |
| `plot_features_cat_regression()`  | Visualiza la relación entre variables categóricas y el target mediante histogramas |

---

## 🚀 Acciones

### 1. Puedes importar el módulo como te indicamos a continuación

```python
from toolbox_ML import (
    describe_df,
    tipifica_variables,
    get_features_num_regression,
    plot_features_num_regression,
    get_features_cat_regression,
    plot_features_cat_regression
)
```

### 2. Consulta el notebook de ejemplo

Abre `Team_Challenge_ToolBox.ipynb` para ver ejemplos de uso en un dataset real.

---

## 👨‍👩‍👧‍👦 Autores

Este proyecto fue desarrollado por el siguiente equipo durante el bootcamp de Data Science en The Bridge:

- Helene Vancaloen (Helenevc)
- Mario Simarro (msimgit)

---

## ⚙️ Requisitos

- Python 3.8+
- Pandas
- NumPy
- Seaborn
- Scikit-learn
- SciPy


---

## 📌 Licencia de uso

Este proyecto está publicado con fines educativos. Siéntete libre de usar y modificar el código según tus necesidades.
