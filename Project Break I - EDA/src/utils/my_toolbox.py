
from itertools import combinations
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


#----------------------------------------------------------------------------------------------------------------
# Función card_tipo
#----------------------------------------------------------------------------------------------------------------

def card_tipo(df):
    n_rows = len(df)

    # Umbrales dinámicos
    if n_rows <= 500:
        umbral_categoria = 10
        umbral_continua = 50
    elif n_rows <= 5000:
        umbral_categoria = 15
        umbral_continua = 25
    else:
        umbral_categoria = 20
        umbral_continua = 15

    df_temp = pd.DataFrame({
        "Card": df.nunique(),
        "%_Card": df.nunique() / len(df) * 100,
        "Tipo": df.dtypes
    })
    df_temp.loc[df_temp.Card == 1, "%_Card"] = 0.00

    df_temp["tipo_sugerido"] = "Categorica"  # Valor por defecto

    # Reglas de clasificación
    df_temp.loc[df_temp["Card"] == 2, "tipo_sugerido"] = "Binaria"
    df_temp.loc[(df_temp["Card"] > 2) & (df_temp["Card"] < umbral_categoria), "tipo_sugerido"] = "Categorica"
    df_temp.loc[(df_temp["Card"] >= umbral_categoria) & (df_temp["%_Card"] < umbral_continua), "tipo_sugerido"] = "Numerica discreta"
    df_temp.loc[(df_temp["%_Card"] >= umbral_continua) & (df_temp["Tipo"].isin(['int64', 'float64'])), "tipo_sugerido"] = "Numerica continua"

    # Reglas para 'por clasificar'
    condiciones_por_clasificar = (
        ((df_temp["%_Card"] >= umbral_continua) & (~df_temp["Tipo"].isin(['int64', 'float64']))) |
        ((df_temp["tipo_sugerido"] == "Numerica discreta") & (~df_temp["Tipo"].isin(['int64', 'float64'])))
    )
    df_temp.loc[condiciones_por_clasificar, "tipo_sugerido"] = "Por clasificar"

    # Crear listas agrupadas
    list_categoricas = df_temp[df_temp["tipo_sugerido"].isin(["Categorica", "Binaria"])].index.tolist()
    list_numericas = df_temp[df_temp["tipo_sugerido"].isin(["Numerica continua", "Numerica discreta"])].index.tolist()
    list_por_clasificar = df_temp[df_temp["tipo_sugerido"] == "Por clasificar"].index.tolist()

    return df_temp, list_categoricas, list_numericas, list_por_clasificar


#----------------------------------------------------------------------------------------------------------------
# Función analysis_uni
#----------------------------------------------------------------------------------------------------------------

def analysis_uni(df, display_units=False):
    """
    Análisis univariante de variables numéricas y categóricas en un DataFrame de pandas.

    Autor: MS
    Fecha: Julio 2025

    Descripción:
    -------------
    Esta función realiza un análisis univariante detallado de un DataFrame, clasificando automáticamente
    las variables en numéricas, categóricas, binarias, discretas, continuas o por clasificar, utilizando
    la función `card_tipo(df)`.

    Para cada variable:
    - Numéricas: se calculan estadísticas descriptivas y se generan histogramas, boxplots y violin plots.
    - Categóricas: se muestran frecuencias absolutas y relativas, y se visualizan con gráficos de barras y pastel.

    Parámetros:
    ------------
    - df : pd.DataFrame
        El DataFrame a analizar.
    - display_units : bool, opcional (default=False)
        Si es True, se muestran etiquetas y unidades en los gráficos.

    Requisitos:
    ------------
    - La función `card_tipo(df)` debe estar definida en el entorno.
    - Librerías necesarias: numpy, pandas, matplotlib, seaborn
    """

    df.info()

    # Clasificación de variables
    df_tipo, categoricas, numericas, por_clasificar = card_tipo(df)

    linea = '-' * 100
    print(linea)
    print("Propuesta de categorización del dataset:")
    print(linea)
    print(df_tipo)

    # Análisis de variables numéricas
    for col in numericas:
        data = df[col].dropna()

        # Si es datetime, convertir a timestamp numérico
        if np.issubdtype(data.dtype, np.datetime64):
            data = data.astype('int64') / 1e9  # convertir a segundos desde epoch

        print(linea)
        print(f"Análisis univariante para la variable numérica: {col}")
        print(linea)

        media = data.mean()
        mediana = data.median()
        moda = data.mode().values
        cuartiles = np.quantile(data, [0.25, 0.5, 0.75])
        percentiles = np.percentile(data, [10, 25, 50, 75, 90])
        varianza = data.var()
        desviacion = data.std()
        rango = data.max() - data.min()
        minimo = data.min()
        maximo = data.max()
        asimetria = data.skew()
        curtosis = data.kurt()

        print("\nEstadísticos de centralidad:")
        print(f"\t{'Media:':<25}{media:.4f}")
        print(f"\t{'Mediana:':<25}{mediana:.4f}")
        print(f"\t{'Moda:':<25}{moda}")
        print(f"\t{'Cuartiles(25,50,75):':<25}{cuartiles}")
        print(f"\t{'Percent.(10,25,50,75,90):':<25}{percentiles}")

        print("\nEstadísticos de dispersión:")
        print(f"\t{'Varianza:':<25}{varianza:.4f}")
        print(f"\t{'Desviación estándar:':<25}{desviacion:.4f}")
        print(f"\t{'Rango:':<25}{rango:.4f}")
        print(f"\t{'Mínimo:':<25}{minimo:.4f}")
        print(f"\t{'Máximo:':<25}{maximo:.4f}")
        print(f"\t{'Asimetría:':<25}{asimetria:.4f}")
        print(f"\t{'Curtosis:':<25}{curtosis:.4f}")

        q1, q3 = cuartiles[0], cuartiles[2]
        iqr = q3 - q1
        lower_whisker = max(data.min(), q1 - 1.5 * iqr)
        upper_whisker = min(data.max(), q3 + 1.5 * iqr)

        fig = plt.figure(constrained_layout=True, figsize=(12, 4))
        gs = fig.add_gridspec(1, 4)

        ax1 = fig.add_subplot(gs[0, 0])
        sns.histplot(data, kde=True, ax=ax1)
        ax1.set_title(f"Histograma + KDE: {col}")
        if display_units:
            ax1.set_xlabel(col)

        ax2 = fig.add_subplot(gs[0, 1:3])
        sns.boxplot(x=data, ax=ax2)
        ax2.set_title(f"Boxplot: {col}")
        ax2.axvline(q1, color='orange', linestyle='--', label='Q1')
        ax2.axvline(q3, color='orange', linestyle='--', label='Q3')
        ax2.axvline(lower_whisker, color='red', linestyle=':', label='Bigote inferior')
        ax2.axvline(upper_whisker, color='red', linestyle=':', label='Bigote superior')
        ax2.legend()

        ax3 = fig

    # Análisis de variables categóricas
    for col in categoricas:
        data = df[col].astype(str).fillna("Desconocido")
        print(linea)
        print(f"Análisis univariante para la variable categórica: {col}")
        print(linea)

        freq_abs = data.value_counts()
        if len(freq_abs) > 10:
            top_9 = freq_abs.nlargest(9)
            otros = freq_abs.iloc[9:].sum()
            freq_abs = pd.concat([top_9, pd.Series({'Otros': otros})])


        freq_rel = freq_abs / freq_abs.sum() * 100

        tabla = pd.DataFrame({'Frecuencia absoluta': freq_abs, 'Frecuencia relativa (%)': freq_rel.round(2)})
        print(tabla)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        freq_abs.plot(kind='bar', ax=axes[0])
        axes[0].set_title(f"Frecuencia absoluta: {col}")
        if display_units:
            axes[0].set_ylabel("Frecuencia")

        freq_rel.plot(kind='pie', ax=axes[1], autopct='%1.1f%%' if display_units else None)
        axes[1].set_title(f"Frecuencia relativa (%): {col}")
        axes[1].set_ylabel("")

        plt.suptitle(f"Análisis univariante de: {col}", fontsize=14)
        plt.tight_layout()
        plt.show()



#----------------------------------------------------------------------------------------------------------------
# Función contingency_combined_categorical
#----------------------------------------------------------------------------------------------------------------

def contingency_combined_categorical(df, target):
    """
    Análisis bivariante entre variables categóricas y una variable objetivo categórica en un DataFrame.

    Autor: MS
    Fecha: Julio 2025

    Descripción:
    -------------
    Esta función realiza un análisis bivariante entre una variable objetivo categórica (`target`)
    y todas las demás variables categóricas del DataFrame. Para cada variable categórica se generan:

    - Una tabla de contingencia con frecuencias absolutas.
    - Una tabla de proporciones relativas por fila.
    - Un multiplot con:
        1. Diagrama de barras apiladas.
        2. Mapa de calor de proporciones.
        3. Tabla de contingencia como texto.

    Este análisis permite explorar visualmente la relación entre las categorías de cada variable
    y los valores del target, facilitando la detección de patrones o asociaciones.

    Parámetros:
    ------------
    - df : pd.DataFrame
        El DataFrame que contiene los datos a analizar.
    - target : str
        Nombre de la variable objetivo categórica frente a la cual se analizarán las demás variables categóricas.

    Requisitos:
    ------------
    - La función `card_tipo(df)` debe estar definida y devolver:
        (df_tipo, list_categoricas, list_numericas, list_por_clasificar)
    - Librerías necesarias: pandas, matplotlib, seaborn
    """

    df_tipo, list_categoricas, list_numericas, list_por_clasificar = card_tipo(df)
    list_categoricas = [col for col in list_categoricas if col != target]

    for cat_var in list_categoricas:
        tabla = pd.crosstab(df[cat_var], df[target])
        tabla_prop = pd.crosstab(df[cat_var], df[target], normalize='index')

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Diagrama de barras apiladas
        tabla.plot(kind='bar', stacked=True, ax=axes[0], colormap='tab20c')
        axes[0].set_title('Barras apiladas')
        axes[0].set_xlabel(cat_var)
        axes[0].set_ylabel('Frecuencia')
        axes[0].legend(title=target)

        # 2. Mapa de calor de proporciones por fila
        sns.heatmap(tabla_prop, annot=True, cmap='Blues', fmt=".2f", ax=axes[1])
        axes[1].set_title('Mapa de calor de proporciones')
        axes[1].set_xlabel(target)
        axes[1].set_ylabel(cat_var)

        # 3. Tabla de contingencia como texto
        axes[2].axis('off')
        table_text = tabla.to_string()
        axes[2].text(0, 0.5, table_text, fontsize=10, va='center', ha='left')
        axes[2].set_title('Tabla de contingencia')

        plt.tight_layout()
        plt.show()



#----------------------------------------------------------------------------------------------------------------
# Función multivariable_categorical_analysis
#----------------------------------------------------------------------------------------------------------------

def multivariable_categorical_analysis(df, target):
    """
    Análisis multivariable entre variables categóricas y una variable objetivo categórica en un DataFrame.

    Para cada combinación de dos variables categóricas predictoras, se analiza su relación conjunta con la variable
    objetivo categórica mediante:
    - Tabla de contingencia tridimensional.
    - Tabla de proporciones.
    - Mapa de calor de proporciones.

    Parámetros:
    ------------
    - df : pd.DataFrame
        El DataFrame que contiene los datos a analizar.
    - target : str
        Nombre de la variable objetivo categórica.

    Requisitos:
    ------------
    - La función `card_tipo(df)` debe estar definida y devolver:
        (df_tipo, list_categoricas, list_numericas, list_por_clasificar)
    - Librerías necesarias: pandas, matplotlib, seaborn
    """

    df_tipo, list_categoricas, list_numericas, list_por_clasificar = card_tipo(df)

    for var1, var2 in combinations(list_categoricas, 2):
        tabla = pd.crosstab(index=[df[var1], df[var2]], columns=df[target])
        tabla_prop = pd.crosstab(index=[df[var1], df[var2]], columns=df[target], normalize='index')

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Mapa de calor de proporciones
        tabla_prop_reset = tabla_prop.reset_index()
        tabla_melted = tabla_prop_reset.melt(id_vars=[var1, var2], var_name=target, value_name='Proporcion')

        pivot_table = tabla_melted.pivot_table(index=var1, columns=var2, values='Proporcion', aggfunc='mean')
        sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt=".2f", ax=axes[0])
        axes[0].set_title(f'Mapa de calor de proporciones promedio\n({var1} vs {var2})')
        axes[0].set_xlabel(var2)
        axes[0].set_ylabel(var1)

        # Tabla de contingencia como texto
        axes[1].axis('off')
        table_text = tabla.to_string()
        axes[1].text(0, 0.5, table_text, fontsize=9, va='center', ha='left')
        axes[1].set_title(f'Tabla de contingencia\n({var1} vs {var2} vs {target})')

        plt.tight_layout()
        plt.show()





def multiplot_dispersion_con_correlacion(df, lista_columnas_x, columna_y, tamano_puntos=50, mostrar_correlacion=False):
    """
    Crea un multiplot de diagramas de dispersión entre varias columnas y una columna Y común.

    Args:
    df (pandas.DataFrame): DataFrame que contiene los datos.
    lista_columnas_x (list): Lista de nombres de columnas para el eje X.
    columna_y (str): Nombre de la columna para el eje Y.
    tamano_puntos (int, opcional): Tamaño de los puntos en el gráfico. Por defecto es 50.
    mostrar_correlacion (bool, opcional): Si es True, muestra la correlación en los gráficos.
    """
    
    n_cols = 4
    n_vars = len(lista_columnas_x)
    n_rows = math.ceil(n_vars / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    axes = axes.flatten()
    
    for i, columna_x in enumerate(lista_columnas_x):
        ax = axes[i]
        sns.scatterplot(data=df, x=columna_x, y=columna_y, s=tamano_puntos, ax=ax)

        if mostrar_correlacion:
            correlacion = df[[columna_x, columna_y]].corr().iloc[0, 1]
            ax.set_title(f'{columna_x} vs {columna_y}\nCorr: {correlacion:.2f}')
        else:
            ax.set_title(f'{columna_x} vs {columna_y}')
        
        ax.set_xlabel(columna_x)
        ax.set_ylabel(columna_y)
        ax.grid(True)

    # Eliminar subplots vacíos
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def multiplot_grouped_boxplots(df, cat_col, lista_numericas, group_size=5):
    """
    Crea un multiplot de boxplots para múltiples columnas numéricas agrupadas por una variable categórica.
    
    Args:
    df (pandas.DataFrame): El DataFrame con los datos.
    cat_col (str): La columna categórica por la que agrupar.
    lista_numericas (list): Lista de columnas numéricas a graficar.
    group_size (int): Número de categorías por gráfico (default = 5).
    """
    
    unique_cats = df[cat_col].unique()
    num_cats = len(unique_cats)
    num_grupos = math.ceil(num_cats / group_size)
    total_plots = num_grupos * len(lista_numericas)

    n_cols = 4
    n_rows = math.ceil(total_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    plot_idx = 0
    for num_col in lista_numericas:
        for i in range(0, num_cats, group_size):
            subset_cats = unique_cats[i:i+group_size]
            subset_df = df[df[cat_col].isin(subset_cats)]

            ax = axes[plot_idx]
            sns.boxplot(x=cat_col, y=num_col, data=subset_df, ax=ax)
            ax.set_title(f'{num_col} por {cat_col} (Grupo {i//group_size + 1})')
            ax.tick_params(axis='x', rotation=45)
            plot_idx += 1

    # Eliminar subplots vacíos
    for j in range(plot_idx, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def multiplot_grouped_histograms(df, cat_col, lista_numericas, group_size=5):
    """
    Crea un multiplot de histogramas para múltiples columnas numéricas agrupadas por una variable categórica.
    
    Args:
    df (pandas.DataFrame): El DataFrame con los datos.
    cat_col (str): La columna categórica por la que agrupar.
    lista_numericas (list): Lista de columnas numéricas a graficar.
    group_size (int): Número de categorías por gráfico (default = 5).
    """

    unique_cats = df[cat_col].unique()
    num_cats = len(unique_cats)
    num_grupos = math.ceil(num_cats / group_size)
    total_plots = len(lista_numericas) * num_grupos

    n_cols = 4
    n_rows = math.ceil(total_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    plot_idx = 0
    for num_col in lista_numericas:
        for i in range(0, num_cats, group_size):
            subset_cats = unique_cats[i:i + group_size]
            subset_df = df[df[cat_col].isin(subset_cats)]

            ax = axes[plot_idx]
            for cat in subset_cats:
                sns.histplot(
                    subset_df[subset_df[cat_col] == cat][num_col],
                    kde=True,
                    label=str(cat),
                    ax=ax,
                    element='step'
                )

            ax.set_title(f'{num_col} por {cat_col} (Grupo {i // group_size + 1})')
            ax.set_xlabel(num_col)
            ax.set_ylabel('Frecuencia')
            ax.legend()
            plot_idx += 1

    # Eliminar subplots vacíos
    for j in range(plot_idx, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_combined_graphs_plus(df, columns, whisker_width=1.5, bins=None):
    num_cols = len(columns)
    if num_cols:
        fig, axes = plt.subplots(num_cols, 2, figsize=(14, 5 * num_cols))

        for i, column in enumerate(columns):
            if df[column].dtype in ['int64', 'float64']:
                ax_hist = axes[i, 0] if num_cols > 1 else axes[0]
                ax_box = axes[i, 1] if num_cols > 1 else axes[1]

                # Histograma y KDE
                sns.histplot(df[column], kde=True, ax=ax_hist, bins="auto" if not bins else bins)
                ax_hist.set_title(f'Histograma y KDE de {column}')

                # Boxplot
                sns.boxplot(x=df[column], ax=ax_box, whis=whisker_width, showmeans=True)
                ax_box.set_title(f'Boxplot de {column} (whis={whisker_width})')

                # Calcular estadísticas
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                median = df[column].median()
                IQR = Q3 - Q1
                lower_whisker = max(df[column].min(), Q1 - whisker_width * IQR)
                upper_whisker = min(df[column].max(), Q3 + whisker_width * IQR)

                # Añadir etiquetas al boxplot
                ax_box.text(lower_whisker, 0.1, f'Min: {lower_whisker:.2f}', ha='center', va='bottom', color='purple', fontsize=10, transform=ax_box.get_xaxis_transform())
                ax_box.text(Q1, 0.2, f'Q1: {Q1:.2f}', ha='center', va='bottom', color='blue', fontsize=10, transform=ax_box.get_xaxis_transform())
                ax_box.text(median, 0.3, f'Mediana: {median:.2f}', ha='center', va='bottom', color='green', fontsize=10, transform=ax_box.get_xaxis_transform())
                ax_box.text(Q3, 0.2, f'Q3: {Q3:.2f}', ha='center', va='bottom', color='red', fontsize=10, transform=ax_box.get_xaxis_transform())
                ax_box.text(upper_whisker, 0.1, f'Max: {upper_whisker:.2f}', ha='center', va='bottom', color='orange', fontsize=10, transform=ax_box.get_xaxis_transform())

        plt.tight_layout()
        plt.show()

