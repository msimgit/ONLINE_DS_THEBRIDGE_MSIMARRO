# --------------------------------------------------------------------------------------------------------------------------
# Lista de librerías a importar
# --------------------------------------------------------------------------------------------------------------------------

import pandas as pd                                                                             # Para manejar dataframes
import numpy as np                                                                              # Para operaciones numéricas
import seaborn as sns                                                                           # Para generar los pairplots
import matplotlib.pyplot as plt                                                                 # Para mostrar los gráficos
from scipy.stats import pearsonr, mannwhitneyu, shapiro, levene, f_oneway, kruskal, ttest_ind   # Para valores estadísticos
import warnings
warnings.filterwarnings("ignore")                                                               # Para ignorar warnings scipy   



# --------------------------------------------------------------------------------------------------------------------------
# 4. Funcion: plot_features_num_regression
# --------------------------------------------------------------------------------------------------------------------------

def plot_features_num_regression(df, target_col = "", columns = [], umbral_corr = 0, pvalue = None):
    """
    
    Permite crear gráficos de dispersión múltiple (pairplots) entre nuestra variable target y aquellas columnas numéricas
    que superen un umbral de correlación especificado. También permiten aplicar un filtro adicional basado en
    un valor de significancia estadística, p-value. 
    Los gráficos se generan en grupos de hasta 5 variables (incluyendo el target).
    
    Parámetros:
    - df (pd.DataFrame):    Conjunto de datos.
    - target_col (str):     Nombre de la variable objetivo. Obligatorio.
    - columns (list[str]):  Lista de variables numéricas a considerar. Si se omite, se usan todas las disponibles.
    - umbral_corr (float):  Mínimo valor absoluto de correlación requerido. Por defecto es 0.
    - pvalue (float o None):Nivel de significancia estadística (entre 0 y 1). Si es None, no se aplica este filtro.

    Develve:
    - list[str] o None:     la lista de variables que cumplen los criterios, o None en caso de error.
    
    Si la lista no está vacía, la función pintará una pairplot del dataframe considerando la columna designada por "target_col" 
    y aquellas incluidas en "column" que cumplan que su correlación con "target_col" es superior en valor absoluto a 
    "umbral_corr", y que, en el caso de ser pvalue diferente de "None", además cumplan el test de correlación para el 
    nivel 1-pvalue de significación estadística. La función devolverá los valores de "columns" que cumplan con las condiciones 
    anteriores. 
    
 
    """

    # Verficaciones iniciales
    if not isinstance(df, pd.DataFrame):
        print(f"Error: {df} no es un DataFrame.")
        return None
    if target_col not in df.columns:
        print(f"Error: La columna '{target_col}' no está disponible en {df}.")
        return None
    if not np.issubdtype(df[target_col].dtype, np.number):
        print(f"Error: {target_col} debe ser de tipo numérico.")
        return None
    if not (0 <= umbral_corr <= 1):
        print("Error: El umbral de correlación debe estar comprendido entre 0 y 1.")
        return None
    if pvalue is not None and not (0 < pvalue < 1):
        print("Error: El 'pvalue' introducido debe estar entre 0 y 1.")
        return None

    # Selección de variables. 
    # Considerando los argumentos proporcionados o los valores por defecto.
    if not columns:
        columns = df.select_dtypes(include=np.number).columns.drop(target_col).tolist()
    else:
        columns = [col for col in columns if col in df.columns and np.issubdtype(df[col].dtype, np.number)]
    
    variables_seleccionadas = []

    for var in columns:
        datos_validos = df[[var, target_col]].dropna()
        if datos_validos.shape[0] < 2:
            continue

        correlacion, p = pearsonr(datos_validos[var], datos_validos[target_col])

        if abs(correlacion) >= umbral_corr:
            if pvalue is None or p < (1 - pvalue):
                variables_seleccionadas.append(var)

    if not variables_seleccionadas:
        print("No se han encontrado variables que cumplan con los criterios establecidos.")
        return []

    # Visualización en bloques de hasta 4 variables + la objetivo
    max_x_bloque = 4
    for i in range(0, len(variables_seleccionadas), max_x_bloque):
        variables = variables_seleccionadas[i:i + max_x_bloque]
        columnas = [target_col] + variables
        sns.pairplot(df[columnas].dropna())
        plt.suptitle(f"Pairplot: {', '.join(columnas)}", y=1.02)
        plt.show()

    return variables_seleccionadas


# --------------------------------------------------------------------------------------------------------------------------
# 5. get_features_cat_regresion
# --------------------------------------------------------------------------------------------------------------------------

from scipy.stats import mannwhitneyu, shapiro, levene, f_oneway, kruskal
import pandas as pd
import numpy as np


def get_features_cat_regression(df, target_col, pvalue = 0.05):
    """
    
    Identifica variables categóricas con relación estadísticamente significativa con una variable objetivo numérica.
    Utiliza automáticamente U de Mann-Whitney, ANOVA o Kruskal-Wallis según el número de grupos y los supuestos estadísticos.

    Argumentos:
    - df(pd.DataFrame): DataFrame, parámetro obligatorio sin valor por defecto. 
    - target_col(str):  Parámetro obligatorio sin valor por defecto. 
                        Señala una variable numérica de alta cardinalidad contenida en el DataFrame (df), 
                        contra la que se efectuarán las comparaciones estadísticas. 
    - pvalue (float):   Opcional. Valor por defecto = 0.05 
                        Probabilidad de que un valor estadístico calculado sea posible dada una hipótesis nula cierta.
                        En caso de que el valor calculado sea superior se considerarán las variables como no relacionadas. 
                        Si el valor calculado es igual o inferior a la probabilidad ('pvalue'), las variables se consideran 
                        como relacionadas estadísticamente

    Devuelve:
    - list[str] o None: Lista de nombres de columnas categóricas significativamente relacionadas con la variable objetivo.
    
    """

    # Validaciones iniciales
    if not isinstance(df, pd.DataFrame):
        print("Error: El argumento 'df' no es un DataFrame.")
        return None
    if target_col not in df.columns:
        print(f"Error: La columna '{target_col}' no se encuentra en el DataFrame.")
        return None
    if not np.issubdtype(df[target_col].dtype, np.number):
        print("Error: La variable objetivo debe ser numérica.")
        return None
    if df[target_col].nunique() < 10:
        print("Error: La variable objetivo no tiene suficiente cardinalidad.")
        return None
    if not (0 < pvalue < 1):
        print("Error: El valor de 'pvalue' debe estar entre 0 y 1.")
        return None

    # Selección de variables categóricas
    cat_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not cat_vars:
        print("No se encontraron variables categóricas en el DataFrame.")
        return []

    features = []

    for var in cat_vars:
        if df[var].nunique() < 2:
            continue

        grupos = [df[df[var] == val][target_col].dropna() for val in df[var].unique()]
        if any(len(g) < 2 for g in grupos):
            continue

        try:
            if len(grupos) == 2:
                stat, p = mannwhitneyu(grupos[0], grupos[1])
            else:
                normalidad = all(shapiro(g)[1] > 0.05 for g in grupos if len(g) >= 3)
                homocedasticidad = levene(*grupos)[1] > 0.05

                if normalidad and homocedasticidad:
                    stat, p = f_oneway(*grupos)
                else:
                    stat, p = kruskal(*grupos)

            if p <= pvalue:
                features.append(var)
        except Exception as e:
            print(f"Error al analizar la variable '{var}': {e}")
            return None

    return features


# --------------------------------------------------------------------------------------------------------------------------
# 6. plot_features_cat_regression
# --------------------------------------------------------------------------------------------------------------------------

def plot_features_cat_regression(df, target_col = "", columns = [], pvalue = 0.05, with_individual_plot = False):
    """
    
    Permite crear histogramas entre nuestra variable target y aquellas columnas categóricas
    que superen un umbral de significancia estadística especificado. Los gráficos se generan en grupos de hasta 5 variables
    (incluyendo el target).

    Argumentos:
    - df (pd.DataFrame):            Conjunto de datos.
    - target_col (str):             Nombre de la variable objetivo. Obligatorio.
    - columns (list[str]):          Lista de variables categóricas a considerar. Si se omite, se usan todas las disponibles.
    - pvalue (float):               Nivel de significancia estadística. Por defecto es 0.05.
    - with_individual_plot (bool):  Si es True, genera un gráfico individual para cada variable significativa.

    Devuelve:
    - list[str] o None:             la lista de variables que cumplen los criterios, o None en caso de error.
    
    La función pintará los histogramas agrupados de la variable "target_col" para cada uno de los valores de las variables 
    categóricas incluidas en columns que cumplan que su test de relación con "target_col" es significativo para el nivel 1-pvalue 
    de significación estadística
    
    """

    # Verificaciones iniciales

    # Eliminamos la variable 'target_col' si se encuentra en la lista de columnas
    if target_col in columns:
        columns.remove(target_col)

    if not isinstance(df, pd.DataFrame):
        print("Error: El argumento 'df' no es un DataFrame.")
        return None
    if target_col not in df.columns:
        print(f"Error: La columna '{target_col}' no está en el DataFrame.")
        return None
    if not np.issubdtype(df[target_col].dtype, np.number):
        print("Error: La variable objetivo debe ser numérica.")
        return None
    if not (0 < pvalue < 1):
        print("Error: El valor de 'pvalue' debe estar entre 0 y 1.")
        return None
    if not isinstance(with_individual_plot, bool):
        print("Error: 'with_individual_plot' debe ser booleano.")
        return None

    # Solo se deben graficar aquellas variables categóricas que tengan una relación estadísticamente significativa 
    # con target_col, según el test adecuado (Mann-Whitney, ANOVA o Kruskal-Wallis), con un nivel de significación de p < pvalue
    # Si la variable categórica tiene 2 niveles:
    #    - Se aplica el test U de Mann-Whitney.
    # Si tiene más de 2 niveles:
    #    - Se evalúa:
    #        - Normalidad en cada grupo con el test de Shapiro-Wilk.
    #        - Homogeneidad de varianzas con el test de Levene.
    #    - Si ambos se cumplen → ANOVA.
    #    - Si alguno no se cumple → Kruskal-Wallis.
    def test_statistical_relation(temp_df, cat_col, target_col):
        groups = [group[target_col].values for _, group in temp_df.groupby(cat_col)]
        if len(groups) == 2:
            try:
                _, p = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                return p
            except Exception:
                return 1.0
        elif len(groups) > 2:
            try:
                normal = all(shapiro(g)[1] > 0.05 for g in groups if len(g) >= 3)
                equal_var = levene(*groups)[1] > 0.05
                if normal and equal_var:
                    _, p = f_oneway(*groups)
                else:
                    _, p = kruskal(*groups)
                return p
            except Exception:
                return 1.0
        return 1.0

    # Verificamos que la lista no está vacía
    # Si la lista está vacía, entonces la función igualará "columns" analiza las variables numéricas del dataframe 
    related = []
    if columns is None or len(columns) == 0:
        num_cols = df.select_dtypes(include=[np.number]).columns.drop(target_col, errors='ignore')
        high_cardinality = [col for col in num_cols if df[col].nunique() / len(df) > 0.01]
        for col in high_cardinality:
            try:
                stat, p = ttest_ind(df[target_col].dropna(), df[col].dropna(), equal_var=False)
                if p < pvalue:
                    related.append(col)
            except Exception:
                continue

    else:
        # Evaluamos la relación estadística para cada columna categórica
        for col in columns:
            if col in df.columns:
                p = test_statistical_relation(df, col, target_col)
                if p < pvalue:
                    related.append(col)

    # Graficamos los histrogramas teniendo en cuenta el argumento with_individual_plot
    if related:
        if with_individual_plot:
            for col in related:
                plt.figure(figsize=(8, 5))
                sns.histplot(data=df, x=target_col, hue=col, multiple="stack", kde=False)
                plt.title(f"Distribución de {target_col} según {col}")
                plt.xlabel(target_col)
                plt.ylabel("Frecuencia")
                plt.legend(title=col)
                plt.tight_layout()
                plt.show()
        else:
            fig, axes = plt.subplots(len(related), 1, figsize=(10, 5 * len(related)))
            if len(related) == 1:
                axes = [axes]
            for ax, col in zip(axes, related):
                df_plot = df[[target_col, col]].dropna().copy()
                df_plot["__encoded__"] = df_plot[col].astype("category").cat.codes
                cmap = sns.color_palette("viridis", df_plot["__encoded__"].nunique())

                sns.histplot(data=df_plot, x=target_col, hue="__encoded__", palette=cmap, multiple="stack", kde=False, ax=ax)
                ax.set_title(f"Distribución de {target_col} según {col}")
                ax.set_xlabel(target_col)
                ax.set_ylabel("Frecuencia")

                categories = df_plot[col].astype("category").cat.categories
                handles = [plt.Rectangle((0,0),1,1, color=cmap[i]) for i in range(min(10, len(categories)))]
                # Limitamos la leyenda a las primeras 10 categorías para evitar saturación
                labels = [str(categories[i]) for i in range(min(10, len(categories)))]
                ax.legend(handles=handles, labels=labels, title=col)

            plt.tight_layout()
            plt.show()

        return related
    else:
        print("No se encontraron variables relacionadas con el target.")
