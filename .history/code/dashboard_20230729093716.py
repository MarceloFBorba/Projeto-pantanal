import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling  import BorderlineSMOTE

import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu

import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import Layout
from plotly.subplots import make_subplots

from numerize.numerize import numerize

from PIL import Image

import pickle

import xgboost as xgb


st.set_page_config(page_title='Pantanal.dev', 
                   page_icon='imagens/LogoFraudWatchdog.png',
                   layout='wide',
                   initial_sidebar_state='auto'
                   )

# file_path = 'https://www.dropbox.com/s/b44o3t3ehmnx2b7/creditcard.csv?dl=1'
file_path = 'creditcard.csv'

df = pd.read_csv(file_path)

model_path = 'xgboost_model.pkl'

with open(model_path, 'rb') as arquivo_pkl:
    modelo_carregado = pickle.load(arquivo_pkl)

X = df.drop('Class', axis = 1)
y = df['Class']

borderLineSMOTE = BorderlineSMOTE(sampling_strategy= 0.1, random_state=42)
X_over,y_over = borderLineSMOTE.fit_resample(X, y)

rus = RandomUnderSampler(random_state=42)
X_under, y_under = rus.fit_resample(X_over, y_over)

X_train, X_test, y_train, y_test = train_test_split(X_under, y_under, test_size=0.2, shuffle=True, random_state=42)

scaler = StandardScaler()

X_test['std_amount'] = scaler.fit_transform(X_test['Amount'].values.reshape(-1, 1))
X_test['std_time'] = scaler.fit_transform(X_test['Time'].values.reshape(-1, 1))

amount = X_test["Amount"]

X_test.drop(['Time', 'Amount'], axis=1, inplace=True)
X_train.drop(['Time', 'Amount'], axis=1, inplace=True)

y_pred_xgb = modelo_carregado.predict(X_test)

X_test['Amount'] = amount
df_resultados = pd.DataFrame({'Transacao': range(len(y_test)),
        'Previsao': y_pred_xgb,
        'Rotulo_verdadeiro':y_test,
        'Amount':X_test['Amount']})

falsos_positivos = df_resultados[(df_resultados['Previsao'] == 0) & (df_resultados['Rotulo_verdadeiro'] == 1)]
falsos_negativos = df_resultados[(df_resultados['Previsao'] == 1) & (df_resultados['Rotulo_verdadeiro'] == 0)]

valor_falso_positivo = falsos_positivos['Amount'].sum()

with st.sidebar:
    st.sidebar.image('imagens/LogoFraudWatchdog.png', width=150)
    selected2 = option_menu("Menu",["Home", "Dados Usados", "Gráficos", "Sobre"], 
    icons=['house', 'database', 'graph-up', 'info-circle'], 
    menu_icon="menu-app", default_index=0,
    styles={
        "nav-link-selected": {"background-color": "#0378A6"}
    }
    )

# Header da página
if (selected2 != "Gráficos"):
    header_left, header_mid, header_right = st.columns([1, 2, 1], gap='large')
    with header_left:
        image = Image.open("imagens/logo-pantanal.png")
        # image = Image.open("/Projeto-pantanal/imagens/logo-pantanal.png")
        # Exibindo a imagem
        st.image(image, width=260)
    with header_mid:
        st.title('Detecção de fraudes em cartões de crédito')

    with header_right:
        image = Image.open("imagens/ufms_logo_negativo_rgb.png")
        #image = Image.open("imagens/ufms_logo_negativo_rgb.png")
        st.image(image, width=130)

# pagina Home
if (selected2 == "Home"):
    with st.empty():
        st.header(':blue[Introdução]')
    with st.empty():
        st.write('###### Anualmente, as perdas globais totais devidas a fraudes financeiras têm estado na faixa de bilhões de dólares, com algumas estimativas sugerindo um custo anual para os Estados Unidos acima de 400 bilhões de dólares, segundo Waleed Hilal, S. Andrew Gadsden e John Yawney, no artigo entitulado “Financial Fraud: A Review of Anomaly Detection Techniques and Recent Advances”.\
            \n\n ###### Entre essas fraudes, aquelas envolvendo cartões de crédito são de grande relevância, uma vez que a sua não-detecção acarreta em prejuízos consideráveis, tanto para o consumidor quanto para a instituição financeira. Por todos esses motivos, o investimento na área de detecção de fraudes por meio de Inteligência Artificial vem crescendo a cada ano.\
            \n\n ###### Portanto o desenvolvimento de algoritmos e modelos eficazes de detecção de fraudes é essencial. O conjunto de dados utilizado para o desenvolvimento de um modelo para esse propósito é o de transações de cartões de crédito datada de setembro de 2013, realizado por titulares de cartões europeus. Este conjunto de dados contém 284.807 transações, das quais 492 foram rotuladas como fraudes. A classe de fraudes representa apenas 0,172% de todas as transações, tornando o conjunto de dados altamente desequilibrado, sendo necessário sua manipulação.')

# pagina Dados usados
if (selected2 == "Dados Usados"):
    st.header(":blue[Dados Usados]")
    st.write("###### As variáveis de entrada neste conjunto de dados são numéricas e foram obtidas através de uma transformação PCA, exceto pelas características 'Time' (Tempo) e 'Amount' (Valor). A característica 'Time' representa os segundos decorridos entre cada transação e a primeira transação no conjunto de dados, enquanto 'Amount' é o valor da transação. A característica 'Class' é a variável de resposta, assumindo o valor 1 em caso de fraude e 0 caso contrário.\
        \n\n#### Processo de manipulação dos dados\
        \n\n###### O conjunto de dados foi dividido em conjuntos de treino e teste, e precauções foram tomadas para evitar que dados de treino se misturassem com dados de teste, utilizando a função **_drop_duplicates_**. Além disso, para combater o desequilíbrio entre transações normais e fraudulentas, o oversampling foi aplicado às transações fraudulentas, criando dados sintéticos que representam 10% do número de transações normais. Em seguida, o undersampling foi utilizado para reduzir o número de transações normais e equilibrar os dados.\
        \n\n###### Diferentes algoritmos foram empregados para treinar o modelo de detecção de fraudes, incluindo a regressão logística, que atribui probabilidades de classificação, a árvore de decisão, um modelo de aprendizado supervisionado que divide os dados em subconjuntos puros, e o XGBoost, que combina vários modelos de árvore de decisão para criar um modelo mais poderoso.\
        \n\n#### Conclusão\
        \n\n###### A árvore de decisão se destacou na interpretação dos resultados, fornecendo uma visão clara das características mais importantes que levaram à classificação das transações como fraudulentas. Isso pode ser especialmente útil para projetos com dados não anonimizados, permitindo a identificação e o monitoramento de variáveis sensíveis relacionadas às transações possivelmente fraudulentas.\
        \n\n###### Por outro lado, o XGBoost apresentou os melhores resultados em todas as métricas, especialmente em relação aos falsos positivos, demonstrando uma melhora na predição com pequenas alterações nos parâmetros padrões. Além disso, o XGBoost forneceu informações sobre as variáveis mais influentes em suas decisões.\
        \n\n###### Em suma, a combinação de diferentes técnicas de tratamento de dados e a utilização de algoritmos variados contribuíram para a criação de um modelo de detecção de fraudes eficiente e robusto, que pode ser valioso tanto para a prevenção quanto para a compreensão de atividades fraudulentas em transações de cartões de crédito.\
        \n\n#### Bibliografia\
        \n\n###### Devido a questões de confidencialidade, as características originais e informações de fundo sobre os dados não podem ser fornecidas. No entanto, pesquisadores têm trabalhado nesse campo e disponibilizaram um simulador de dados de transações para auxiliar no desenvolvimento de metodologias de detecção de fraudes em cartões de crédito, e também pisquisadores aperfeiçoando cada vez mais o campo de Data Science.\
        \n\n###### Dentre os trabalhos relevantes sobre o tema, destacam-se:\
        \n\n###### 1 - Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. [Calibrating Probability with Undersampling for Unbalanced Classification](https://www.researchgate.net/publication/283349138_Calibrating_Probability_with_Undersampling_for_Unbalanced_Classification). Simpósio de Inteligência Computacional e Mineração de Dados (CIDM), IEEE, 2015\
        \n\n###### 2 - Dal Pozzolo, Andrea; Caelen, Olivier; Le Borgne, Yann-Ael; Waterschoot, Serge; Bontempi, Gianluca. [Learned lessons in credit card fraud detection from a practitioner perspective](https://www.researchgate.net/publication/260837261_Learned_lessons_in_credit_card_fraud_detection_from_a_practitioner_perspective), Sistemas especialistas e suas aplicações,41,10,4915-4928,2014, Pergamon\
        \n\n###### 3 - Dal Pozzolo, Andrea; Boracchi, Giacomo; Caelen, Olivier; Alippi, Cesare; Bontempi, Gianluca. [Credit card fraud detection: a realistic modeling and a novel learning strategy](https://www.researchgate.net/publication/319867396_Credit_Card_Fraud_Detection_A_Realistic_Modeling_and_a_Novel_Learning_Strategy), IEEE transações em redes neurais e sistemas de aprendizagem,29,8,3784-3797,2018,IEEE\
        \n\n###### 4 - Dal Pozzolo, Andrea [Adaptive Machine learning for credit card fraud detection](https://di.ulb.ac.be/map/adalpozz/pdf/Dalpozzolo2015PhD.pdf), tese de ULB MLG PhD (supervisionado por G. Bontempi)\
        \n\n###### 5 - Carcillo, Fabrizio; Dal Pozzolo, Andrea; Le Borgne, Yann-Aël; Caelen, Olivier; Mazzer, Yannis; Bontempi, Gianluca. [Scarff: a scalable framework for streaming credit card fraud detection with Spark](https://www.researchgate.net/publication/319616537_SCARFF_a_Scalable_Framework_for_Streaming_Credit_Card_Fraud_Detection_with_Spark), Information fusion,41, 182-194,2018,Elsevier\
        \n\n###### 6 - Carcillo, Fabrizio; Le Borgne, Yann-Aël; Caelen, Olivier; Bontempi, Gianluca. [Streaming active learning strategies for real-life credit card fraud detection: assessment and visualization](https://www.researchgate.net/publication/332180999_Deep-Learning_Domain_Adaptation_Techniques_for_Credit_Cards_Fraud_Detection), Jornal Internacional de Ciência de Dados e Análise, 5,4,285-300,2018,Springer International Publishing\
        \n\n###### 7 - Bertrand Lebichot, Yann-Aël Le Borgne, Liyun He, Frederic Oblé, Gianluca Bontempi [Deep-Learning Domain Adaptation Techniques for Credit Cards Fraud Detection](https://www.researchgate.net/publication/332180999_Deep-Learning_Domain_Adaptation_Techniques_for_Credit_Cards_Fraud_Detection), INNSBDDL 2019: Avanços recentes em Big Data e Deep Learning, pp 78-88, 2019\
        \n\n###### 8 - Fabrizio Carcillo, Yann-Aël Le Borgne, Olivier Caelen, Frederic Oblé, Gianluca Bontempi [Combining Unsupervised and Supervised Learning in Credit Card Fraud Detection](https://www.researchgate.net/publication/333143698_Combining_Unsupervised_and_Supervised_Learning_in_Credit_Card_Fraud_Detection) Information Sciences, 2019\
        \n\n###### 9 - Yann-Aël Le Borgne, Gianluca Bontempi [Reproducible machine Learning for Credit Card Fraud Detection - Practical Handbook](https://www.researchgate.net/publication/351283764_Reproducible_Machine_Learning_for_Credit_Card_Fraud_Detection_-_Practical_Handbook)\
        \n\n###### 10 - Bertrand Lebichot, Gianmarco Paldino, Wissam Siblini, Liyun He, Frederic Oblé, Gianluca Bontempi [Incremental learning strategies for credit cards fraud detection](https://www.researchgate.net/publication/352275169_Incremental_learning_strategies_for_credit_cards_fraud_detection), Jornal Internacional de Ciência de Dados e Análise\
        \n\n###### 11 - Max Tingle, [Preventing Data Leakage in Your Machine Learning Model](https://towardsdatascience.com/preventing-data-leakage-in-your-machine-learning-model-9ae54b3cd1fb)")
  
# pagina Graficos
if (selected2 == "Gráficos"):
    st.title(':blue[Gráficos]')
    
    # total1, total2, total3, total4, total5 = st.columns(5, gap='large')
    
    total1, total2, total3, total4, total5 = st.columns(5, gap='large')
# Resultados resumidos
    with total1:
        image = Image.open('imagens/dinheiro-total-cortado.png')
        # Exibindo a imagem
        total = df['Amount'].sum()
        st.image(image, use_column_width='Auto')
        st.metric(label='##### Valores totais ($)', value=numerize(total))

    with total2:
        image = Image.open('imagens/dinheiro-fraudado.png')
        #image = Image.open('imagens/sem-dinheiro.png')
        # Exibindo a imagem
        totalPerdas = df.Amount[df['Class'] == 1].sum()
        st.image(image, width=125)
        st.metric(label='##### Perdas com fraudes ($)', value=numerize(totalPerdas))

    with total3:
        image = Image.open('imagens/roubo.png')
        # Exibindo a imagem
        
        st.image(image, use_column_width=125)
        st.metric(label='##### Com o modelo ($)', value=numerize(valor_falso_positivo))

    with total4:
        image = Image.open('imagens/sem-dinheiro.png')
        #image = Image.open('imagens/sem-dinheiro.png')
        # Exibindo a imagem

        # Carregar o modelo salvo em formato .pkl
        media = df.Amount[df['Class'] == 1].mean()
        # porcentagem = df.Amount[df['Class'] == 1].sum() / df.Amount.sum() * 100
        
        st.image(image, use_column_width='Auto')
        st.metric(label='##### Média dos valores fraudados ($)', value=numerize(media))
        
    with total5:
        image = Image.open('imagens/dinheiro-repetido.png')
        # Exibindo a imagem
        moda = df.Amount[df['Class'] == 1].mode().values[0]
        
        st.image(image, use_column_width=125)
        st.metric(label='##### Moda do valor em fraudes ($)', value=numerize(moda))

    Q1, Q2 = st.columns(2)

# Distribuição das transações
    with Q1:
        st.header(':blue[Distribuição das transações]')
        st.write('###### Neste gráfico podemos observar a quantidade de transações fraudulentas em comparação com as transações normais.')
       
        class_counts = df['Class'].value_counts()
        class_counts.rename({'count': 'Quantidade'}, inplace=True)

        colors = ['#0C3559', '#F2F2F2']
        my_layout = Layout(hoverlabel = dict(bgcolor = '#FFFFFF'), template='simple_white')

        fig = go.Figure(layout = my_layout)

        fig.add_trace(go.Bar(
            x=class_counts.index,
            y=class_counts.values,
            text=class_counts.values,
            textposition='outside',
            marker_color=colors,
        ))

        fig.update_layout(
            height = 500,
            xaxis=dict(
                tickvals=[0, 1],
            ticktext=['Normal', 'Fraude',])
        )
            
        st.plotly_chart(fig, use_container_width=True)
    
# Resumo estatístico das transações
    with Q2:
        st.header(':blue[Resumo estatístico das transações]')
        my_layout = Layout(hoverlabel = dict(bgcolor = '#FFFFFF'), template='simple_white')
        st.write('###### Neste gráfico podemos observar o resumo estatístico das transações normais e fraudulentas.')

        fig = go.Figure(layout = my_layout)
        fig.add_trace(go.Box(
            y=df.Amount[df['Class'] == 0],
            name='Transações normais',
            marker_color='#0C3559',
            boxmean=True,
            boxpoints='outliers',
            hovertext='casa'
        ))
        fig.add_trace(go.Box(
            y=df.Amount[df['Class'] == 1],
            name='Transações fradulentas',
            marker_color='#3698BF',
            boxmean=True
        ))

        fig.update_layout(
            height = 500,
            title={
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                },
            title_text= '',
            yaxis_range=[-10,500],
            showlegend=False,
            hoverlabel=dict(bgcolor='#FFFFFF'))
        st.plotly_chart(fig, use_container_width=True)

# Transações normais
    with st.container():    
        st.header(':blue[Transações normais]')
        st.write('###### As transações normais têm seus valores mais comuns entre \$1,00 e \$15,00 apenas.')
        
        valoresTransacoesNormais = df[['Amount', 'Class']]
        valoresTransacoesNormais = valoresTransacoesNormais[valoresTransacoesNormais['Class'] == 0]
        valoresTransacoesNormais = valoresTransacoesNormais[valoresTransacoesNormais['Amount'] != 0.0]
        valoresTransacoesNormais['count'] = valoresTransacoesNormais.groupby('Amount')['Amount'].transform('count')
        valoresTransacoesNormais = valoresTransacoesNormais.sort_values(by=['count', 'Amount'], ascending=False)
        valoresTransacoesNormais = valoresTransacoesNormais.drop_duplicates()

        valoresTransacoesNormais = valoresTransacoesNormais.reset_index().drop(columns=['index'], axis = 1)

        x = valoresTransacoesNormais['Amount'][:10]
        y = valoresTransacoesNormais['count'][:10]

        colors = ['#0C3559', '#033F73', '#033E8C', '#0378A6', '#049DBF',
            '#3698BF', '#A6ACE6', '#A0C9D9 ', '#DEE0FC', '#F2F2F2']

        my_layout = Layout(hoverlabel=dict(
                    bgcolor='#FFFFFF'),
                    template='simple_white')

        fig = go.Figure(data=[
                go.Bar(name='', x=x.index, y=y, hovertemplate=' ',
                    text=y,
                    textposition='outside',
                    marker_color=colors, showlegend=False,),],
                layout=my_layout)

        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis_range=[0, 5000],
            yaxis=dict(
                tickvals=[0, 1000, 2000, 3000, 4000,
                    5000, 6000, 7000, 8000, 9000,
                    10000, 11000, 12000,
                    13000, 14000, 15000],
            ticktext=['0', '1 mil', '2 mil', '3 mil', '4 mil', '5 mil', '6 mil', '7 mil', '8 mil',
                    '9 mil', '10 mil', '11 mil', '12 mil', '13 mil', '14 mil', '15 mil']))
        fig.update_layout(
            title={
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                },
            title_text='Valores mais comuns das transações normais',
            # title_font_color='#0C3559',
            # title_font_size=20,
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis_range=[0,15000],
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                tickvals=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                ticktext=[f"$ {percent:.2f}" for percent in x]),
            yaxis=dict(showgrid=False, zeroline=False))

        fig.update_xaxes(title_text='Valores')
        fig.update_yaxes(title_text='Ocorrências')

        st.plotly_chart(fig, use_container_width=True)
        
# Transações fraudulentas
    with st.container():
        st.header(':blue[Transações fraudulentas]')
        st.write('###### Aqui se vê quais os valores mais comuns das transações fraudulentas. Uma observação interessante é que a maior parte delas tem valor de $1,00, por ser um valor baixo e pouco provável de ser barrado.Outra observação é que as transações fraudulentas, em sua maioria, são de valores baixos.')
            
        valoresTransacoesFraudulentas = df[['Amount', 'Class']]
        valoresTransacoesFraudulentas = valoresTransacoesFraudulentas[valoresTransacoesFraudulentas['Class'] == 1]
        valoresTransacoesFraudulentas = valoresTransacoesFraudulentas[valoresTransacoesFraudulentas['Amount'] != 0.0]
        valoresTransacoesFraudulentas['count'] = valoresTransacoesFraudulentas.groupby('Amount')['Amount'].transform('count')
        valoresTransacoesFraudulentas = valoresTransacoesFraudulentas.sort_values(by=['count', 'Amount'], ascending=False)
        valoresTransacoesFraudulentas = valoresTransacoesFraudulentas.drop_duplicates()

        valoresTransacoesFraudulentas = valoresTransacoesFraudulentas.reset_index().drop(columns=['index'], axis = 1)

        x = valoresTransacoesFraudulentas['Amount'][:10]
        y = valoresTransacoesFraudulentas['count'][:10]

        colors = ['#0C3559', '#033F73', '#033E8C', '#0378A6', '#049DBF', '#3698BF',
                '#A0C9D9', '#A6ACE6', '#DEE0FC', '#F2F2F2']

        my_layout = Layout(hoverlabel=dict(
                        bgcolor='#FFFFFF'),
                        template='simple_white')

        fig = go.Figure(data=[
                    go.Bar(name='', x=x.index, y=y, hovertemplate=' ',
                        text=y,
                        textposition='outside',
                        marker_color=colors, showlegend=False,),],
                    layout=my_layout)

        fig.update_layout(
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis_range=[0, 150],
        yaxis=dict(
            tickvals=[0, 10, 20, 30, 40,
                        50, 60, 70, 80, 90,
                        100, 110, 125]))
            # ticktext=['0', '1 mil', '2 mil', '3 mil', '4 mil', '5 mil', '6 mil', '7 mil', '8 mil',
                        # '9 mil', '10 mil', '11 mil', '12 mil', '13 mil', '14 mil', '15 mil']))
        fig.update_layout(
            title={
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                },
            title_text='Valores mais comuns das transações fraudulentas',
            # title_font_color='#0C3559',
            # title_font_size=20,
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis_range=[0,125],
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                tickvals=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                ticktext=[f"$ {percent:.2f}" for percent in x]),
            yaxis=dict(showgrid=False, zeroline=False))

        fig.update_xaxes(title_text='Valores')
        fig.update_yaxes(title_text='Ocorrências')

        st.plotly_chart(fig, use_container_width=True)
        
# Transações por tempo        
    with st.container():

        st.header(':blue[Transações por tempo]')
        st.write('###### No gráfico de transações normais podemos observar que grande parte ocorre no tempo entre 40000s e 80000s e entre 120000s e 160000s.\
            \n ###### Já no gráfico de trasações fraudulentas, as mesmas possuem uma grande quantidade de transações realizadas no tempo de 40000s e 90000s.')    
        # fig, ax = plt.subplots(nrows=2, ncols = 1, figsize=(17, 8))

        # ax[0].hist(df.Time[df.Class == 0], bins = 30, color = '#0088B7', rwidth= 0.9)

        # ax[0].text(df.Time[df.Class == 0].min(), 18000, "Transações normais",
        #         fontsize = 15,
        #         color = '#3f3f4e',
        #         fontweight= 'bold')

        # ax[0].set_xlabel('Tempo(s)', fontsize = 12, color= '#000000')
        # ax[0].set_ylabel('Transações', fontsize = 12, color= '#000000')

        # ax[0].spines['top'].set_visible(False)
        # ax[0].spines['right'].set_visible(False)

        # ax[0].margins(x=0)
                
        # ax[1].hist(df.Time[df.Class == 1], bins = 30, color= '#5FC2D9', rwidth= 0.9)

        # ax[1].text(df.Time[df.Class == 1].min(), 55, "Transações fraudulentas",
        #         fontsize = 15,
        #         color = '#3f3f4e',
        #         fontweight= 'bold')

        # ax[1].set_xlabel('Tempo(s)', fontsize = 12, color= '#000000')
        # ax[1].set_ylabel('Transações', fontsize = 12, color= '#000000')
        # ax[1].spines['top'].set_visible(False)
        # ax[1].spines['right'].set_visible(False)

        # ax[1].margins(x=0)
        

        # plt.tight_layout(pad = 3.0)

        # st.pyplot(fig, use_container_width=True)
        
        # Criar subplots com compartilhamento de eixo x e espaço vertical entre as figuras
        fig = make_subplots(rows=2, cols=1)

        # Calcular os valores de frequência para cada hora
        counts_class_0, edges_class_0 = np.histogram(df[df.Class == 0].Time.div(3600), bins=48)
        counts_class_1, edges_class_1 = np.histogram(df[df.Class == 1].Time.div(3600), bins=48)

        # Adicionar histograma para a classe 0 com cor personalizada (vermelho) e hovertext com a quantidade de transações
        fig.add_trace(
            go.Histogram(x=df[df.Class == 0].Time.div(3600), 
                         nbinsx=48, hovertext='', name='Transações normais',
                         hovertemplate=[f"{percent:.0f}" for percent in counts_class_0]),
            row=1, col=1
        )

        # Adicionar histograma para a classe 1 com cor personalizada (azul) e hovertext com a quantidade de transações
        fig.add_trace(
            go.Histogram(x=df[df.Class == 1].Time.div(3600), 
                         nbinsx=48, hovertext='', name='Transações fraudulentas',
                         hovertemplate=[f"{percent:.0f}" for percent in counts_class_1]),
            row=2, col=1
        )

        # Atualizar rótulos dos eixos
        fig.update_xaxes(title_text="Tempo (horas)", row=1, col=1)
        fig.update_xaxes(title_text="Tempo (horas)", row=2, col=1)
        fig.update_yaxes(title_text="Frequência", row=1, col=1)
        fig.update_yaxes(title_text="Frequência", row=2, col=1)

        # Definir os valores do eixo y para cada subplot
        y_values_subplot1 = [0, 2500, 5000, 7500, 10000, ]  # Valores personalizados para o subplot 1
        y_values_subplot2 = [0, 50, 100, 150]  # Valores personalizados para o subplot 2
        fig.update_yaxes(tickvals=y_values_subplot1, 
                         showgrid=False, row=1, col=1)
        fig.update_yaxes(tickvals=y_values_subplot2, 
                         showgrid=False, row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)
        
# Transações por valor
    with st.container():
        st.header(':blue[Transações por valor]')
        st.write('###### Nos gráficos de transações por valor podemos observar as transações normais são todos abaixo de $3000. \
            \n ###### Já no gráfico de transações Fraudulentas podemos obsevar que grande parte das transações são de 0 até 750 dólares, tendo o seu grande volume em valores abaixo de $50.')  
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(17, 8))

        ax[0].hist(df.Amount[df.Class == 0], bins = 30, color = '#0088B7', rwidth= 0.9)

        ax[0].text(df.Amount[df.Class == 0].min(), 310000, "Transações normais",
                fontsize = 15,
                color = '#3f3f4e',
                fontweight= 'bold')

        ax[0].set_xlabel('Valor($)', fontsize = 12, color= '#000000')
        ax[0].set_ylabel('Transações', fontsize = 12, color= '#000000')

        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)

        ax[0].margins(x=0)

        ax[1].hist(df.Amount[df.Class == 1], bins = 30, color = '#5FC2D9', rwidth= 0.9)

        ax[1].text(df.Amount[df.Class == 1].min(), 350, "Transações fraudulentas",
                fontsize = 15,
                color = '#3f3f4e',
                fontweight= 'bold')

        ax[1].set_xlabel('Valor($)', fontsize = 12, color= '#000000')
        ax[1].set_ylabel('Transações', fontsize = 12, color= '#000000')

        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)

        ax[1].margins(x=0)
        
        plt.tight_layout(pad = 3.0)
        
        st.pyplot(fig, use_container_width=True)

#----------------------------------------------------------------------------
# Modelo xgboost

    with st.container():
        st.header('Modelo')

# modelo XGBoost funcionando

    #from imblearn.under_sampling import RandomUnderSampler
    #from sklearn.model_selection import train_test_split
    #import xgboost as xgb
    #from imblearn.over_sampling  import BorderlineSMOTE

    #from sklearn.model_selection import train_test_split
    #from sklearn.preprocessing   import StandardScaler

    Q3, Q4 = st.columns(2)

# possivel modelo XGboost
    with Q3:
        # st.header('Matriz de confusão do XGBoost')
        # st.write('###### Melhor resultado obtido pelos modelos treinados')
        
        #df = df.drop_duplicates()
        
        #borderLineSMOTE = BorderlineSMOTE(sampling_strategy= 0.1, random_state=42)
        
        #X_over,y_over = borderLineSMOTE.fit_resample(X, y)
        
        #rus = RandomUnderSampler(random_state=42)
        #X_under, y_under = rus.fit_resample(X_over, y_over)
    
    
        #scaler = StandardScaler()

        #X_train['std_amount'] = scaler.fit_transform(X_train['Amount'].values.reshape(-1, 1))
        #X_train['std_time'] = scaler.fit_transform(X_train['Time'].values.reshape(-1, 1))

        #X_test['std_amount'] = scaler.fit_transform(X_test['Amount'].values.reshape(-1, 1))
        #X_test['std_time'] = scaler.fit_transform(X_test['Time'].values.reshape(-1, 1))

        #X_train.drop(['Time', 'Amount'], axis=1, inplace=True)
        #X_test.drop(['Time', 'Amount'], axis=1, inplace=True)
        
        #plt.figure(figsize=(2, 2))

        #modelXGB = xgb.XGBClassifier(n_estimators     = 100,
        #                        max_depth        = 4,
        #                        learning_rate    = 0.3,
        #                        subsample        = 1,
        #                        colsample_bytree = 1,
        #                        reg_alpha        = 0,
        #                        reg_lambda       = 0,
        #                        scale_pos_weight = 1,
        #                        random_state     = 42,)
        
        #modelXGB.fit(X_train, y_train)

        plt.figure(figsize=(2, 2))
        
        matriz = confusion_matrix(y_test, y_pred_xgb)
        sns.heatmap(matriz, square=True, annot=True, cbar=False, cmap= 'Blues', fmt='.0f')
        
        plt.title('Matriz de confusão do XGB',
                fontsize = 8,
                color = '#000000',
                pad= 3)
    
        plt.xlabel('Previsão',fontsize = 4, color= '#000000')
        plt.ylabel('Valor real'  ,fontsize = 4, color= '#000000')
        # plt.show()
        #st.plotly_chart(plt, use_container_width=True)

        st.pyplot(plt, use_container_width=True)
        
        # df_resultados = pd.DataFrame({'Transacao': range(len(y_test)),
        #         'Previsao': y_pred_xgb,
        #         'Rotulo_verdadeiro':y_test,
        #         'Amount':X_test['Amount']})
        
        # falsos_positivos = df_resultados[(df_resultados['Previsao'] == 1) & (df_resultados['Rotulo_verdadeiro'] == 0)]
        
        st.dataframe(falsos_positivos)
        
    with Q4:
        st.header('Identificação das colunas mais influêntes')
        st.write('###### Neste gráfico podemos observar as colunas com maior influencia para a classificação.')
        
        xgb.plot_importance(modelo_carregado, grid=False, ylabel='Colunas',
            max_num_features=15,
            title='15 colunas mais importantes para classificação')
        # plt.show()
        st.pyplot(plt, use_container_width=True)
            
# pagina sobre
if (selected2 == "Sobre"):    
    st.header(":blue[Sobre]")
    
    st.write('###### O Pantanal.Dev é um programa de capacitação imersiva em tecnologias inovadoras. No Módulo Onça Pintada, abordamos o campo de Data Science com diversas técnicas de mineração de dados, pré-processamento, transformações de dados, aprendizado de máquina e agrupamento de dados, com foco especial na Detecção de Fraude em Cartões de Crédito. \
        \n ###### Este projeto foi totalmente feito com a Linguagem Python, com toda a sua flexibilidade, vasta biblioteca e comunidade ativa de desenvolvedores permitiram uma abordagem eficaz e eficiente na criação das soluções necessárias e foi utilizado o Streamit para a criação de um Web App interativo, facilitando a visualização dos dados. \
        \n ###### O projeto foi desenvolvido por alunos da  :blue[Universidade Federal de Mato Grosso do Sul (UFMS)], em parceria com as empresas :blue[B3, PDtec, BLK e Neoway], com foco na aplicação prática de conhecimentos em Data Science para detecção de fraudes em transações de cartões de crédito.')
    
    st.subheader(":blue[Projeto realizado por:]")    
    
    perfil1, perfil2, perfil3, perfil4= st.columns(4, gap= "large")
    
    with perfil1:
                  
        st.image("imagens/rodrigo1.png", width=200)
        #st.image("imagens/rodrigo1.png", width=200)
        st.write('#### **_Wallynson Rodrigo H. da Silva_** \n\n Curso: Sistemas de informação \n\n Email: w.rodrigo@ufms.br', use_column_width=True)
        
        url = "https://github.com/wrodrigohs"
        url2= "https://www.linkedin.com/in/wrodrigohs/"
    
        # link1, link2 = st.columns([1, 2])
        # with link1:

        # if st.button(":violet[GitHub]",url):
        #     webbrowser.open_new_tab(url) 
        # # with link2:           
        # if st.button( ":blue[Linkedin]",url2):
        #     webbrowser.open_new_tab(url2)

        # st.markdown(f'<a href="{url}" target="_blank">:violet[GitHub]</a>', unsafe_allow_html=True)
        # st.markdown(f'<a href="{url2}" target="_blank">:blue[LinkedIn]</a>', unsafe_allow_html=True)

        st.write(f'<a href="{url}" target="_blank" style="text-decoration: none;"><button style="background-color: #000000; color: white; padding: 8px 15px; border: none; border-radius: 5px; cursor: pointer;">GitHub</button></a>', unsafe_allow_html=True)

        st.write(f'<a href="{url2}" target="_blank" style="text-decoration: none;"><button style="background-color: #4682b4; color: white; padding: 8px 15px; border: none; border-radius: 5px; cursor: pointer;">LinkedIn</button></a>', unsafe_allow_html=True)


    with perfil2:
        st.image("imagens/vitor2.png", width=200)
        #st.image("imagens/vitor2.png", width=200)
        st.write('#### **_Vitor de Sousa Santos_** \n\n Curso: Engenharia da computação \n\n Email: vi.ssantos2000@gmail.com \n\n  ', use_column_width=True )
        
        #links para GitHub e linkedin
        url = "https://github.com/VitorSousaS"
        url2= "https://www.linkedin.com/in/vitor-de-sousa-santos/"

        # link1, link2 = st.columns([1, 3])
        # with link1:
        # if st.button(":violet[GitHub]",url):
        #     webbrowser.open_new_tab(url) 
        # with link2:           
        # if st.button( ":blue[Linkedin]",url2):
        #     webbrowser.open_new_tab(url2)

        st.write(f'<a href="{url}" target="_blank" style="text-decoration: none;"><button style="background-color: #000000; color: white; padding: 8px 15px; border: none; border-radius: 5px; cursor: pointer;">GitHub</button></a>', unsafe_allow_html=True)

        st.write(f'<a href="{url2}" target="_blank" style="text-decoration: none;"><button style="background-color: #4682b4; color: white; padding: 8px 15px; border: none; border-radius: 5px; cursor: pointer;">LinkedIn</button></a>', unsafe_allow_html=True)
        
    with perfil3:
        st.image("imagens/icaro3.png", width=200)
        #st.image("imagens/icaro3.png", width=200)
        st.write('#### **_Ícaro de Paula F. Coêlho_** \n\nCurso: Engenharia da computação \n\n Email:  icarogga@gmail.com \n\n', use_column_width=True)
        
        url = "https://github.com/icarogga"
        url2= "https://www.linkedin.com/in/ícaro-coelho-3a5b60206/"
        
        # link1, link2 = st.columns([1, 3])
        # with link1:
        # if st.button(":violet[GitHub]",url):
        #     webbrowser.open_new_tab(url) 
        # with link2:           
        # if st.button( ":blue[Linkedin]",url2):
        #     webbrowser.open_new_tab(url2)

        st.write(f'<a href="{url}" target="_blank" style="text-decoration: none;"><button style="background-color: #000000; color: white; padding: 8px 15px; border: none; border-radius: 5px; cursor: pointer;">GitHub</button></a>', unsafe_allow_html=True)

        st.write(f'<a href="{url2}" target="_blank" style="text-decoration: none;"><button style="background-color: #4682b4; color: white; padding: 8px 15px; border: none; border-radius: 5px; cursor: pointer;">LinkedIn</button></a>', unsafe_allow_html=True)
        
    with perfil4:
        st.image("imagens/marcelo4.png", width=200)
        #st.image("imagens/marcelo4.png", width=200)
        st.write('#### **_Marcelo Ferreira Borba_** \nCurso: Sistemas de informação \n\n Email: m.ferreira@ufms.br \n', use_column_width=True)
        
        url = "https://github.com/MarceloFBorba"
        url2= "https://www.linkedin.com/in/marcelo-ferreira-dev/"
        
        # link1, link2 = st.columns([1, 3])
        # with link1:
        # if st.button(":violet[GitHub]",url):
        #     webbrowser.open_new_tab(url) 

        # with link2:           
        # if st.button( ":blue[Linkedin]",url2):
        #     webbrowser.open_new_tab(url2)

        st.write(f'<a href="{url}" target="_blank" style="text-decoration: none;"><button style="background-color: #000000; color: white; padding: 8px 15px; border: none; border-radius: 5px; cursor: pointer;">GitHub</button></a>', unsafe_allow_html=True)

        st.write(f'<a href="{url2}" target="_blank" style="text-decoration: none;"><button style="background-color: #4682b4; color: white; padding: 8px 15px; border: none; border-radius: 5px; cursor: pointer;">LinkedIn</button></a>', unsafe_allow_html=True)
        
    st.write(" #### Mentor do Projeto:")
    
    perfil5= st.container()
    
    with perfil5:
        st.image("imagens/titos5.png", width=200)
        #st.image("imagens/titos5.png", width=200)
        st.write(" #### **_Bruno Laureano Titos Moreno_** \n\n Coordernador de Tecnologia na B3\n\n Email: bruno.moreno@b3.com.br")

        url = "https://www.linkedin.com/in/bruno-titos-8b537abb/"
        
        # if st.button( ":blue[Linkedin]",url):
        #     webbrowser.open_new_tab(url) 

        st.write(f'<a href="{url}" target="_blank" style="text-decoration: none;"><button style="background-color: #4682b4; color: white; padding: 8px 15px; border: none; border-radius: 5px; cursor: pointer;">LinkedIn</button></a>', unsafe_allow_html=True)

