# Libs

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans

st.write("# Iris Dataset  \n")

st.divider()

# Load Data


@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names


@st.cache_data
def model_fit():
    X = df[["petal length (cm)", "petal width (cm)"]].values
    y = df["species"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    softmax_reg = LogisticRegression(C=30, random_state=42)
    softmax_reg.fit(X_train, y_train)

    return softmax_reg, X_train, X_test, y_train, y_test


def proba_graf(softmax_reg):

    X = df[["petal length (cm)", "petal width (cm)"]].values
    y = df["species"]
    custom_cmap = ListedColormap(["#fafab0", "#9898ff", "#a0faa0"])

    x0, x1 = np.meshgrid(np.linspace(0, 8, 500).reshape(-1, 1),
                         np.linspace(0, 3.5, 200).reshape(-1, 1))
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_proba = softmax_reg.predict_proba(X_new)
    y_predict = softmax_reg.predict(X_new)

    zz1 = y_proba[:, 1].reshape(x0.shape)
    zz = y_predict.reshape(x0.shape)

    fig = plt.figure(figsize=(12, 8))
    plt.plot(X[y == 2, 0], X[y == 2, 1], "g^", label="Iris virginica")
    plt.plot(X[y == 1, 0], X[y == 1, 1], "bs", label="Iris versicolor")
    plt.plot(X[y == 0, 0], X[y == 0, 1], "yo", label="Iris setosa")

    plt.contourf(x0, x1, zz, cmap=custom_cmap)
    contour = plt.contour(x0, x1, zz1, cmap="hot")
    plt.clabel(contour, inline=1)
    plt.xlabel("Petal length")
    plt.ylabel("Petal width")
    plt.legend(loc="center left")
    plt.axis([0.5, 7, 0, 3.5])
    plt.grid()
    plt.show()
    return fig


def update_proba(softmax_reg):
    input_data = [[st.session_state.petal_length,
                   st.session_state.petal_width]]
    prediction = softmax_reg.predict(input_data)[0]
    probabilities = softmax_reg.predict_proba(input_data)[0]
    st.session_state.result = target_name[prediction]
    st.session_state.proba = dict(zip(target_name, probabilities.round(2)))


@st.cache_data
def kmeans_fit(k=3):
    X = df_cluster[["petal length (cm)", "petal width (cm)"]].values
    X_train_c, X_test_c = train_test_split(
        X, random_state=42)

    k = k
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    y_pred = kmeans.fit_predict(X)
    return kmeans, y_pred, X_train_c, X_test_c


def cluster_fig(df_cluster):

    fig_scatter, ax_scatter = plt.subplots()
    ax_scatter.scatter(df_cluster['petal length (cm)'],
                       df_cluster['petal width (cm)'], color='blue')
    ax_scatter.set_xlabel("Petal length (cm)")
    ax_scatter.set_ylabel("Petal width (cm)")
    ax_scatter.set_title("Scatter Plot para Clusterizacão")
    return fig_scatter


def cluster_pred_fig():
    # Cria a figura e o eixo
    fig_scatter, ax_scatter = plt.subplots()

    # Plota os pontos do dataset com cores de acordo com o cluster (y_pred)
    scatter = ax_scatter.scatter(
        df_cluster["petal length (cm)"],
        df_cluster["petal width (cm)"],
        c=y_pred,
        cmap='viridis',
        alpha=0.6,
        label='Valores'
    )

    # Plota os centróides com um marcador "X" vermelho e tamanho maior
    centers = kmeans.cluster_centers_
    ax_scatter.scatter(
        centers[:, 0],
        centers[:, 1],
        marker='X',
        c='red',
        s=200,
        label='Centroides'
    )

    ax_scatter.set_xlabel("Petal length (cm)")
    ax_scatter.set_ylabel("Petal width (cm)")
    ax_scatter.set_title("Clusters com Centroides")
    ax_scatter.legend()
    ax_scatter.grid()

    return fig_scatter


# exploratory data analysis

st.write('''
## Análise exploratória dos dados. \n
''')

df, target_name = load_data()

st.write('''
### Método `describe()`: \n
A tabela abaixo apresenta algumas métricas que, posteriormente, serão apresentadas em gráficos.         
''')

describe = df.iloc[:, :-1].describe()
st.dataframe(describe)

st.write('''
- `count`: todos iguais a 150, mostra que o conjuto de dados esta igualmente representado; 
- `mean`: demonstra que a média dos valores de cada caracterísstica estão bem distantes, com excessão das características: "sepal width" e "petal legth";
- `desvio padrão`: baixo nos indica que os dados estão concentrados próximos a média.       
''')

# 1. Pairplot

st.write("\n ### Pairplot \n")
pair_plot = sns.pairplot(df, hue=df.columns[-1])
# O pairplot retorna um objeto PairGrid; usamos o atributo 'fig' para acessar a figura.
st.pyplot(pair_plot.fig)

st.write('''
Este gráfico tem o propósito de demonstrar a relação dois a dois das características, porém desta vez rotuladas de acordo com a espécie. \n
Em um conjunto de dados 2D, em muitos casos, alguns pares de características não irão se diferenciar tanto. \n
- No gráfico `sepal width x sepal length` fica muito difícil separar a espécie 1 da espécie 2. \n
- Já no gráfico `petal length x petal width` temos as característica mais separadas. \n
A diagonal principal demonstra a distribuicão normal dos conjuntos de dados. \n
- As características `petal length` e `petal width` estão menos sobrepostas, sendo esse um forte indicativo para que estas sejam 
as melhores características para um classificador.\n

''')

col1, col2, col3 = st.columns(3)

# 2. Histograma
fig_hist, ax_hist = plt.subplots()
df.hist(ax=ax_hist)
col1.pyplot(fig_hist)

# 3. Boxenplot
fig_boxen, ax_boxen = plt.subplots()
sns.boxenplot(data=df.iloc[:, :-1], ax=ax_boxen)
col2.pyplot(fig_boxen)

# 4. Violinplot
fig_violin, ax_violin = plt.subplots()
sns.violinplot(data=df.iloc[:, :-1], orient="h", ax=ax_violin)
col3.pyplot(fig_violin)

# 5. Correlação (heatmap)

st.write("\n ### Heatmap \n ")
fig_corr, ax_corr = plt.subplots()
sns.heatmap(df.iloc[:, :-1].corr(), cmap='coolwarm', annot=True, ax=ax_corr)
st.pyplot(fig_corr)

st.write('''
Ratificando a linha de pensamento anteriormente contruida, vemos que a matriz de correlação das característica quando relacionadas com as características correlacionadas a
`petal length` e `petal width`.

''')

st.divider()

st.write("## Modelo LogisticRegression \n")

# modelo softmax
softmax_reg, X_train, X_test, y_train, y_test = model_fit()

st.pyplot(proba_graf(softmax_reg))

st.write('''
Foi utilizado o modelo `LogisticRegression` devido o seu médoto `predict_proba`. Desta maneira podemos verificar, dada as caracteristicas, a probabilidade de pertencer a cada classe.\n
''')

st.write("\n #### Medindo a acuracia utilizando `Cross-Validation` \n")
cross_val_df = pd.DataFrame(cross_val_score(
    softmax_reg, X_train, y_train, cv=3, scoring="accuracy"), columns=["Accuracy"])
cross_val_df.index = ["setosa accuracy",
                      "versicolor accuracy", "virginica accuracy"]

st.dataframe(cross_val_df)


st.write("\n ### Playground \n")

# Sliders com armazenamento na session_state via chave (key)
petal_length = st.slider(
    "Petal length",
    float(df['petal length (cm)'].min()),
    float(df['petal length (cm)'].max()),
    key="petal_length"
)
petal_width = st.slider(
    "Petal width",
    float(df['petal width (cm)'].min()),
    float(df['petal width (cm)'].max()),
    key="petal_width"
)

# Inicializando as variáveis na session_state
if 'result' not in st.session_state:
    st.session_state.result = None
if 'proba' not in st.session_state:
    st.session_state.proba = {}  # Inicializa como dicionário vazio

st.write("#### Probabilidade")


if st.button("Set"):
    update_proba(softmax_reg)

if st.session_state.result is not None:
    st.write(f"A espécie classificada foi: {st.session_state.result}")
    if st.session_state.proba:  # Verifica se o dicionário não está vazio
        for specie, proba in st.session_state.proba.items():
            st.write(f"{specie}: {proba}")

st.divider()

st.write("## Clusterizacão")

st.write('''
Para o projeto de clusterização é necessário retitar os rótulos, para que o algotimo possa encontra-los de forma não supervisionada. \n 
''')

df_cluster = df[['petal length (cm)', 'petal width (cm)']].copy()


st.pyplot(cluster_fig(df_cluster))

kmeans, y_pred, X_train_c, X_test_c = kmeans_fit()

st.write('''
\n Ao setar o parametro `n_clusters`, numero esperado de clusters, o algoritmo se encarrega de posicionar controide no local onde haverá mais pontos proximos a ele. 
Desta forma, a classificação ocorre ao definir os pontos que estao da vizinhanca desses centroides. \n
''')

st.write("\n #### Posição dos centroides \n")

df_center = pd.DataFrame(kmeans.cluster_centers_, columns=[
                         "Center Petal width", "Center Petal length"])
df_center.index = ["Cluster 1", "Cluster 2", "Cluster 3"]

st.dataframe(df_center)


st.pyplot(cluster_pred_fig())

st.write("\n #### Medindo a `inercia` \n")
st.write('''
Como o modelo Kmeans não recebe os rotulos de saída, não tem como mediar a acurácia do modelo.\n
Desta forma é utilizado a `inercia` como métrica, esta mede a distancia entre cada intancia e os centroides.
- É a soma dos quadrados das distâncias entre cada instância de treinamento e o centroide mais próximo.\n
''')

inercia = kmeans.inertia_

col4, col5, col6 = st.columns(3)

col5.metric("Inércia", round(inercia, 2), border=True)
