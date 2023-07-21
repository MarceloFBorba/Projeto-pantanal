import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
from streamlit_folium import st_folium
import streamlit.components.v1 as components

import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import Layout

from numerize.numerize import numerize
from streamlit_option_menu import option_menu

from PIL import Image

st.set_page_config(page_title='Pantanal.dev',
                   page_icon='logo-pantanal.png',
                   layout='wide',
                   initial_sidebar_state='collapsed'
                   )

# Instalar sidebar   
# pip install streamlit-option-menu

selected2 = option_menu(None, ["Home", "Dados Usados", "Gráficos", "Sobre"], 
    icons=['house', 'database', 'graph-up', 'info-circle'], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "nav-link-selected": {"background-color": "#0378A6"}
    }
    )

# @st.cache_data
# def get_data(dados):
#     if (dados == 'Estadual - 1º turno'):
#         df = pd.read_csv('Dashboard/data/df_estados_1turno_2020.csv')
#     elif (dados == 'Estadual - 2º turno'):
#         df = pd.read_csv('Dashboard/data/df_estados_2turno_2020.csv')
#     elif (dados == 'Municipal - 1º turno'):
#         df = pd.read_csv('Dashboard/data/df_municipios_1turno_2020.csv')
#     else:
#         df = pd.read_csv('Dashboard/data/df_municipios_2turno_2020.csv')
#     return df

# dados = st.selectbox(label='Selecione a eleição',
#                      options=['Estadual - 1º turno', 'Estadual - 2º turno',
#                               'Municipal - 1º turno', 'Municipal - 2º turno'])

# df = get_data(dados)

# total_eleitores = float(df['aptos'].sum())
# total_eleitores_feminino = float(df['eleitorado_feminino'].sum())
# total_eleitores_masculino = float(df['eleitorado_masculino'].sum())
# comparecimento_percentual = float(df['comparecimento_percentual(%)'].mean())
# abstencao_percentual = float(df['abstencao_percentual(%)'].mean())

header_left, header_mid, header_right = st.columns([1, 2, 1], gap='large')
with header_left:
    image = Image.open('logo-pantanal.png')
    # Exibindo a imagem
    st.image(image, width=260)
with header_mid:
    st.title('')
    st.title('Detecção de fraudes em cartões de crédito')

with header_right:
    image = Image.open('ufms_logo_negativo_rgb.png')
    st.image(image, width=150)

# pagina Home
if (selected2 == "Home"):
    with st.empty():
        st.title('')
    with st.empty():
        st.write('Anualmente, as perdas globais totais devidas a fraudes financeiras têm estado na faixa de bilhões de dólares, com algumas estimativas sugerindo um custo anual para os Estados Unidos acima de 400 bilhões de dólares, segundo Waleed Hilal, S. Andrew Gadsden e John Yawney, no artigo entitulado “Financial Fraud: A Review of Anomaly Detection Techniques and Recent Advances”.\
             \n\nEntre essas fraudes, aquelas envolvendo cartões de crédito são de grande relevância, uma vez que a sua não-detecção acarreta em prejuízos consideráveis, tanto para o consumidor quanto para a instituição financeira. Por todos esses motivos, o investimento na área de detecção de fraudes por meio de Inteligência Artificial vem crescendo a cada ano.')    

# pagina Dados usados
if (selected2 == "Dados Usados"):
    st.write("pagina DADOS")
   
# pagina Graficos
if (selected2 == "Gráficos"):
    file_path = 'https://www.dropbox.com/s/b44o3t3ehmnx2b7/creditcard.csv?dl=1'
    # file_path = "creditcard.csv"
    df = pd.read_csv(file_path)
    st.header("Gráficos")
    
    total1, total2, total3, total4, total5 = st.columns(5, gap='large')
    with total1:
        image = Image.open('sem-dinheiro.png')
        # Exibindo a imagem
        total = 500
        st.image(image, use_column_width='Auto')
        st.metric(label='Perdas com fraudes (R$)', value=numerize(total))

    with total2:
        image = Image.open('sem-dinheiro.png')
        # Exibindo a imagem
        total = 500
        st.image(image, use_column_width='Auto')
        st.metric(label='Perdas com fraudes (R$)', value=numerize(total))

    with total3:
        image = Image.open('sem-dinheiro.png')
        # Exibindo a imagem
        total = 500
        st.image(image, use_column_width='Auto')
        st.metric(label='Perdas com fraudes (R$)', value=numerize(total))

    with total4:
        image = Image.open('sem-dinheiro.png')
        # Exibindo a imagem
        total = 500
        st.image(image, use_column_width='Auto')
        st.metric(label='Perdas com fraudes (R$)', value=numerize(total))

    with total5:
        image = Image.open('sem-dinheiro.png')
        # Exibindo a imagem
        total = 500
        st.image(image, use_column_width='Auto')
        st.metric(label='Perdas com fraudes (R$)', value=numerize(total))
        
    Q1, Q2 = st.columns(2)

    with Q1:
        # plotar gráfico de barras para as Classes
        fig, ax = plt.subplots(figsize=(6,4))

        sns.countplot(data=df, x='Class', palette=['#2A8BF0', '#F03131'])

        ax.set_frame_on(False)

        ax.text(-0.4, df.Class.value_counts()[0] + 45000, 'Distribuição das transações', fontsize = 20, color = '#3f3f4e')

        ax.get_yaxis().set_visible(False)
        ax.set_xticklabels(['Normal', 'Fraude'], fontsize=16, color='#3f3f4e')
        ax.set_xlabel('')

        for i in ax.patches:
            ax.text(i.get_x() + i.get_width() / 2,
                i.get_height() + 5000,
                '{} ({:0,.2f}%)'.format(int(i.get_height()), (i.get_height()/int(df.Class.value_counts()[0] + df.Class.value_counts()[1])) * 100).replace('.',','),
                ha = 'center',
                fontsize=14, color='#3f3f4e')
            
        st.plotly_chart(fig, use_container_width=True)

    with Q2:
        my_layout = Layout(hoverlabel = dict(bgcolor = '#FFFFFF'), template='simple_white')

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
            title_text='Gráfico do resumo estatístico das transações',
            title_font_color='#0C3559',
            title_font_size=20,
            yaxis_range=[-10,500],
            showlegend=False,
            hoverlabel=dict(bgcolor='#FFFFFF'))
        st.plotly_chart(fig, use_container_width=True)

        # plotly.offline.plot(fig, filename = 'filename.html', auto_open=False)
        # st.pyplot(fig)

        # fig.show()


    with st.container():    
        st.header('Transações normais')
        st.write('Transações normais')
        
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
            '#3698BF', '#A0C9D9', '#A6ACE6', '#DEE0FC', '#F2F2F2']

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
            title_font_color='#0C3559',
            title_font_size=20,
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
        

    with st.container():
        st.header('Transações fraudulentas')
        st.write('##### Aqui se observa quais os valores mais comuns das transações fraudulentas. Uma observação (talvez curiosa) é que a maior parte delas tem valor de $1,00, talvez por ser um valor baixo e pouco provável de ser barrado.Outra observação é que as transações fraudulentas, em sua maioria, são de valores baixos.')
            
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
            title_font_color='#0C3559',
            title_font_size=20,
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
        

    with st.container():

        st.header('Transações por tempo')
        st.write('##### texto')    
        fig, ax = plt.subplots(nrows=2, ncols = 1, figsize=(10, 8))

        ax[0].hist(df.Time[df.Class == 0], bins = 30, color = '#2A8BF0', rwidth= 0.9)

        ax[0].text(df.Time[df.Class == 0].min(), 18000, "Transações normais",
                fontsize = 20,
                color = '#3f3f4e',
                fontweight= 'bold')

        ax[0].set_xlabel('Tempo(s)', fontsize = 12, color= '#000000')
        ax[0].set_ylabel('Transações', fontsize = 12, color= '#000000')

        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)

        ax[0].margins(x=0)

        ax[1].hist(df.Time[df.Class == 1], bins = 30, color= '#F03131', rwidth= 0.9)

        ax[1].text(df.Time[df.Class == 1].min(), 55, "Transações fraudulentas",
                fontsize = 20,
                color = '#3f3f4e',
                fontweight= 'bold')

        ax[1].set_xlabel('Tempo(s)', fontsize = 12, color= '#000000')
        ax[1].set_ylabel('Transações', fontsize = 12, color= '#000000')

        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)

        ax[1].margins(x=0)

        st.write(plt.tight_layout(pad = 3.0))
        
        
        
    with st.container():
        st.header('Transações por valor')
        st.write('##### texto')  
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

        ax[0].hist(df.Amount[df.Class == 0], bins = 30, color = '#2A8BF0', rwidth= 0.9)

        ax[0].text(df.Amount[df.Class == 0].min(), 310000, "Transações normais",
                fontsize = 20,
                color = '#3f3f4e',
                fontweight= 'bold')

        ax[0].set_xlabel('Valor($)', fontsize = 12, color= '#000000')
        ax[0].set_ylabel('Transações', fontsize = 12, color= '#000000')

        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)

        ax[0].margins(x=0)

        ax[1].hist(df.Amount[df.Class == 1], bins = 30, color = '#F03131', rwidth= 0.9)

        ax[1].text(df.Amount[df.Class == 1].min(), 350, "Transações fraudulentas",
                fontsize = 20,
                color = '#3f3f4e',
                fontweight= 'bold')

        ax[1].set_xlabel('Valor($)', fontsize = 12, color= '#000000')
        ax[1].set_ylabel('Transações', fontsize = 12, color= '#000000')

        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)

        ax[1].margins(x=0)

        st.write(plt.tight_layout(pad = 3.0))

    from imblearn.under_sampling import RandomUnderSampler
    import sklearn.metrics as metrics
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    from imblearn.over_sampling  import BorderlineSMOTE

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing   import StandardScaler
    from sklearn.metrics         import confusion_matrix


    Q3, Q4 = st.columns(2)

    with Q3:
        st.header('Matriz de confusão do XGBoost')
        st.write('##### texto')
        
        df = df.drop_duplicates()
        X = df.drop('Class', axis = 1)
        y = df['Class']
        
        borderLineSMOTE = BorderlineSMOTE(sampling_strategy= 0.1, random_state=42)
        
        X_over,y_over = borderLineSMOTE.fit_resample(X, y)
        
        rus = RandomUnderSampler()
        X_under, y_under = rus.fit_resample(X_over, y_over)
        
        X_train, X_test, y_train, y_test = train_test_split(X_under, y_under, test_size=0.2, shuffle=True)
        scaler = StandardScaler()

        X_train['std_amount'] = scaler.fit_transform(X_train['Amount'].values.reshape(-1, 1))
        X_train['std_time'] = scaler.fit_transform(X_train['Time'].values.reshape(-1, 1))

        X_test['std_amount'] = scaler.fit_transform(X_test['Amount'].values.reshape(-1, 1))
        X_test['std_time'] = scaler.fit_transform(X_test['Time'].values.reshape(-1, 1))

        X_train.drop(['Time', 'Amount'], axis=1, inplace=True)
        X_test.drop(['Time', 'Amount'], axis=1, inplace=True)
        
        plt.figure(figsize=(2, 2))

        modelXGB = xgb.XGBClassifier(n_estimators     = 125,
                                max_depth        = 6,
                                learning_rate    = 0.3,
                                subsample        = 1,
                                colsample_bytree = 1,
                                reg_alpha        = 0,
                                reg_lambda       = 0,
                                scale_pos_weight = 1,)
        
        modelXGB.fit(X_train, y_train)
        y_pred_xgb = modelXGB.predict(X_test)
        matriz = confusion_matrix(y_test, y_pred_xgb)
        sns.heatmap(matriz, square=True, annot=True, cbar=False, cmap= 'Blues', fmt='.0f')


        plt.title('Matriz de confusão do XGBoost',
                fontsize = 6,
                color = '#000000',
                pad= 5,
                fontweight= 'bold')

        plt.xlabel('Previsão',fontsize = 2, color= '#000000')
        plt.ylabel('Valor real'  ,fontsize = 2, color= '#000000')


        #plt.show()
        #st.plotly_chart(plt, use_container_width=True)
        
        st.pyplot(plt, use_container_width=False)
        
        
    st.header('Matriz de confusão do XGBoost')
    st.write('##### texto')
    
    df = df.drop_duplicates()
    X = df.drop('Class', axis = 1)
    y = df['Class']
    
    borderLineSMOTE = BorderlineSMOTE(sampling_strategy= 0.1, random_state=42)
    
    X_over,y_over = borderLineSMOTE.fit_resample(X, y)
    
    rus = RandomUnderSampler()
    X_under, y_under = rus.fit_resample(X_over, y_over)
    
    X_train, X_test, y_train, y_test = train_test_split(X_under, y_under, test_size=0.2, shuffle=True)
    scaler = StandardScaler()

    X_train['std_amount'] = scaler.fit_transform(X_train['Amount'].values.reshape(-1, 1))
    X_train['std_time'] = scaler.fit_transform(X_train['Time'].values.reshape(-1, 1))

    X_test['std_amount'] = scaler.fit_transform(X_test['Amount'].values.reshape(-1, 1))
    X_test['std_time'] = scaler.fit_transform(X_test['Time'].values.reshape(-1, 1))

    X_train.drop(['Time', 'Amount'], axis=1, inplace=True)
    X_test.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    plt.figure(figsize=(2, 2))

    modelXGB = xgb.XGBClassifier(n_estimators     = 125,
                             max_depth        = 6,
                             learning_rate    = 0.3,
                             subsample        = 1,
                             colsample_bytree = 1,
                             reg_alpha        = 0,
                             reg_lambda       = 0,
                             scale_pos_weight = 1,)
    
    modelXGB.fit(X_train, y_train)
    y_pred_xgb = modelXGB.predict(X_test)
    matriz = confusion_matrix(y_test, y_pred_xgb)
    sns.heatmap(matriz, square=True, annot=True, cbar=False, cmap= 'Blues', fmt='.0f')


    plt.title('Matriz de confusão do XGBoost',
            fontsize = 6,
            color = '#000000',
            pad= 5,
            fontweight= 'bold')

    plt.xlabel('Previsão',fontsize = 2, color= '#000000')
    plt.ylabel('Valor real'  ,fontsize = 2, color= '#000000')


    #plt.show()
    #st.plotly_chart(plt, use_container_width=True)
    
    st.pyplot(plt, use_container_width=False)

# pagina sobre
if (selected2 == "Sobre"):
    st.write("pagina sobre")

    

#         homens = df['eleitorado_masculino_percentual(%)']
#         mulheres = df['eleitorado_feminino_percentual(%)']

#         homens = homens.drop_duplicates()
#         mulheres = mulheres.drop_duplicates()

#         estados = df['estado'].drop_duplicates()

#         fig = go.Figure()

#         fig.add_trace(go.Bar(y=estados, x=homens,
#                              name='Homens',
#                              hovertemplate='%{y} %{x:.2f}%',
#                              marker_color='#355070',
#                              orientation='h'))

#         fig.add_trace(go.Bar(y=estados,
#                              x=mulheres,
#                              hovertemplate='%{y} %{x:.2f}%',
#                              marker_color='#FCC202',
#                              name='Mulheres',
#                              orientation='h'))

#         fig.update_layout(title='', plot_bgcolor="rgba(0,0,0,0)",
#                           title_font_size=22, barmode='relative',
#                           hoverlabel=dict(bgcolor='#FFFFFF'),
#                           template='simple_white',
#                           bargap=0, bargroupgap=0,
#                           margin=dict(l=1, r=1, t=60, b=1),
#                           xaxis_range=[0, 100],
#                           xaxis=dict(tickvals=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
#                                      ticktext=['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']))

#         fig.update_traces(width=0.5)

#         st.plotly_chart(fig, use_container_width=True)
    
# with Q4:
#     st.write('Divisão por sexo no estado selecionado')
#         def plot_chart(estadoIndex, df):
#             estado = format_func_estado(estadoIndex)
#             nomeEstado = [df.loc[(df['estado'] == estado, 'estado')].values[0]]
            
#             homens = [df.loc[(df['estado'] == estado), 
#                 'eleitorado_masculino_percentual(%)'].values[0]]
            
#             mulheres = [df.loc[(df['estado'] == estado), 
#                 'eleitorado_feminino_percentual(%)'].values[0]]

#             my_layout = Layout(hoverlabel=dict(
#                 bgcolor='#FFFFFF'), template='simple_white')

#             fig_sexoEstado = go.Figure()
#             fig_sexoEstado.add_trace(go.Bar(y=nomeEstado, x=homens,
#                                                name='',
#                                                hovertemplate='Homens: %{x:.2f}%',
#                                                marker_color='#355070',
#                                                orientation='h'))

#             fig_sexoEstado.add_trace(go.Bar(y=nomeEstado, x=mulheres,
#                                                hovertemplate='Mulheres: %{x:.2f}%',
#                                                marker_color='#FCC202',
#                                                name='',
#                                                orientation='h'))

#             fig_sexoEstado.update_layout(barmode='relative',
#                                             hoverlabel=dict(bgcolor='#FFFFFF'),
#                                             template='simple_white',
#                                             bargap=0, bargroupgap=0,
#                                             margin=dict(l=1, r=1, t=60, b=1),
#                                             xaxis_range=[0, 100],
#                                             xaxis=dict(
#                                                 tickvals=[0, 10, 20, 30, 40,
#                                                           50, 60, 70, 80, 90, 100],
#                                                 ticktext=['0%', '10%', '20%', '30%', '40%', '50%',
#                                                           '60%', '70%', '80%', '90%', '100%']))

#             fig_sexoEstado.update_traces(width=0.5)
#             fig_sexoEstado.update_xaxes(ticksuffix="")
#             fig_sexoEstado.update_yaxes(ticksuffix="")
#             st.plotly_chart(fig_sexoEstado, use_container_width=True)

#         plot_chart(estado, df)

#     st.write('Divisão por escolaridade por estado')
#     Q5, Q6 = st.columns(2)

#     with Q5:
#         escolaridade_percentual = df[['estado', 'analfabeto_percentual(%)', 'le_escreve_percentual(%)',
#         'fundamental_incompleto_percentual(%)', 'fundamental_completo_percentual(%)', 'medio_incompleto_percentual(%)',
#         'medio_completo_percentual(%)', 
#         'superior_incompleto_percentual(%)', 'superior_completo_percentual(%)']].sort_values(by = 'superior_completo_percentual(%)', ascending = False)[:10]
        
#         x1 = escolaridade_percentual['estado'] 
#         analfabeto = escolaridade_percentual['analfabeto_percentual(%)'] 
#         le_escreve = escolaridade_percentual['le_escreve_percentual(%)']
#         fundamental_incompleto = escolaridade_percentual['fundamental_incompleto_percentual(%)']
#         fundamental_completo = escolaridade_percentual['fundamental_completo_percentual(%)']
#         medio_incompleto = escolaridade_percentual['medio_incompleto_percentual(%)']
#         medio_completo = escolaridade_percentual['medio_completo_percentual(%)']
#         superior_incompleto = escolaridade_percentual['superior_incompleto_percentual(%)']
#         superior_completo = escolaridade_percentual['superior_completo_percentual(%)']

#         my_layout = Layout(hoverlabel = dict(bgcolor = '#FFFFFF'), template = 'simple_white')

#         fig = go.Figure(data=[
#             go.Bar(name='',x= x1, y=analfabeto, hovertemplate = 'Analfabeto: %{y:.2f}%', marker_color='#355070', showlegend = False),
#             go.Bar(name='', x=x1, y=le_escreve, hovertemplate = 'Lê e escreve: %{y:.2f}%', marker_color= '#597092', showlegend = False),
#             go.Bar(name='', x= x1, y=fundamental_incompleto, hovertemplate = 'Fundamental incompleto %{y:.2f}%', marker_color='#7179E6', showlegend = False),
#             go.Bar(name='', x=x1, y=fundamental_completo, hovertemplate = 'Fundamental completo %{y:.2f}%', marker_color= '#DEE0FC', showlegend = False),
#             go.Bar(name='', x= x1, y=medio_incompleto, hovertemplate = 'Médio incompleto: %{y:.2f}%', marker_color='#E9DEFC', showlegend = False),
#             go.Bar(name='', x=x1, y=medio_completo, hovertemplate = 'Médio completo: %{y:.2f}%', marker_color= '#FEE592', showlegend = False),
#             go.Bar(name='', x= x1, y=superior_incompleto, hovertemplate = 'Superior incompleto: %{y:.2f}%', marker_color='#E6DD39', showlegend = False),
#             go.Bar(name='', x=x1, y=superior_completo, hovertemplate = 'Superior completo: %{y:.2f}%', marker_color= '#FCC202', showlegend = False)
#         ], layout = my_layout)

#         fig.update_layout(
#         xaxis=dict(
#             rangeslider=dict(
#             visible=True
#             ),
#         ),
#         barmode='stack',
#         yaxis_range=[0,100],
#         yaxis = dict(
#             tickvals = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
#             ticktext = ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
#         )

#         fig.update_xaxes(ticksuffix = "")
#         fig.update_yaxes(ticksuffix = "")

#         st.plotly_chart(fig, use_container_width=True)

#         #if (dados == 'Estadual - 1º turno'):
#         #    htmlFile = open(
#         #        "D:\\UFMS\\TCC\\Dashboard\\data\\charts\\estados_1turno\\estadosEscolaridade_1turno.html", 'r', encoding='utf-8')
#         #elif (dados == 'Estadual - 2º turno'):
#         #    htmlFile = open(
#         #        "D:\\UFMS\\TCC\\Dashboard\\data\\charts\\estados_2turno\\estadosEscolaridade_2turno.html", 'r', encoding='utf-8')
#         #elif (dados == 'Municipal - 1º turno'):
#         #    htmlFile = open(
#         #        "D:\\UFMS\\TCC\\Dashboard\\data\\charts\\municipios_1turno\\municipiosEscolaridade_1turno.html", 'r', encoding='utf-8')
#         #else:
#         #    htmlFile = open(
#         #        "D:\\UFMS\\TCC\\Dashboard\\data\\charts\\municipios_2turno\\municipiosEscolaridade_2turno.html", 'r', encoding='utf-8')
#         #source_code = htmlFile.read()
#         #components.html(source_code, height=480)

#         with Q6:
#             #ESCOLARIDADE POR ESTADO
#             # st.write('Divisão por escolaridade por estado')

#             def plot_chart(estadoIndex, df):

#                 estado = format_func_estado(estadoIndex)
#                 nomeEstado = [df.loc[(df['estado'] == estado, 'estado')].values[0]]

#                 analfabeto = [df.loc[(df['estado'] == estado), 'analfabeto_percentual(%)'].values[0]]
#                 le_escreve = [df.loc[(df['estado'] == estado), 'le_escreve_percentual(%)'].values[0]]
#                 fundamental_incompleto = [df.loc[(df['estado'] == estado), 
#                     'fundamental_incompleto_percentual(%)'].values[0]]
#                 fundamental_completo = [df.loc[(df['estado'] == estado), 
#                     'fundamental_completo_percentual(%)'].values[0]]
#                 medio_incompleto = [df.loc[(df['estado'] == estado), 'medio_incompleto_percentual(%)'].values[0]]
#                 medio_completo = [df.loc[(df['estado'] == estado), 
#                     'medio_completo_percentual(%)'].values[0]]
#                 superior_incompleto = [df.loc[(df['estado'] == estado), 
#                     'superior_incompleto_percentual(%)'].values[0]]
#                 superior_completo = [df.loc[(df['estado'] == estado), 
#                     'superior_completo_percentual(%)'].values[0]]

#                 my_layout = Layout(hoverlabel=dict(
#                     bgcolor='#FFFFFF'), template='simple_white')

#                 fig = go.Figure(data=[
#                     go.Bar(name='', x=nomeEstado, y=analfabeto,
#                         hovertemplate='Analfabeto: {}%'.format(
#                         str(analfabeto[0]).replace('.', ',')), marker_color='#355070', showlegend=False),
#                     go.Bar(name='', x=nomeEstado, y=le_escreve,
#                         hovertemplate='Lê e escreve: {}%'.format(
#                     str(le_escreve[0]).replace('.', ',')), marker_color='#597092', showlegend=False),
#                     go.Bar(name='', x=nomeEstado, y=fundamental_incompleto,
#                         hovertemplate='Fundamental incompleto: {}%'.format(
#                     str(fundamental_incompleto[0]).replace('.', ',')), marker_color='#7179E6', showlegend=False),
#                     go.Bar(name='', x=nomeEstado, y=fundamental_completo,
#                         hovertemplate='Fundamental completo: {}%'.format(
#                     str(fundamental_completo[0]).replace('.', ',')), marker_color='#DEE0FC', showlegend=False),
#                     go.Bar(name='', x=nomeEstado, y=medio_incompleto,
#                         hovertemplate='Médio incompleto: {}%'.format(
#                     str(medio_incompleto[0]).replace('.', ',')), marker_color='#E9DEFC', showlegend=False),
#                     go.Bar(name='', x=nomeEstado, y=medio_completo,
#                         hovertemplate='Médio completo: {}%'.format(
#                     str(medio_completo[0]).replace('.', ',')), marker_color='#FEE592', showlegend=False),
#                     go.Bar(name='', x=nomeEstado, y=superior_incompleto,
#                         hovertemplate='Superior incompleto: {}%'.format(
#                     str(superior_incompleto[0]).replace('.', ',')), marker_color='#E6DD39', showlegend=False),
#                     go.Bar(name='', x=nomeEstado, y=superior_completo,
#                         hovertemplate='Superior completo: {}%'.format(
#                     str(superior_completo[0]).replace('.', ',')), marker_color='#FCC202', showlegend=False)
#                 ], layout=my_layout)

#                 fig.update_layout(
#                     barmode='stack',
#                     bargap=0, bargroupgap=0,
#                     margin=dict(l=1, r=1, t=1, b=1),
#                     yaxis_range=[0, 100],
#                     yaxis=dict(
#                         tickvals=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
#                         ticktext=['0%', '10%', '20%', '30%', '40%', '50%',
#                                 '60%', '70%', '80%', '90%', '100%']
#                     )
#                 )

#                 fig.update_traces(width=0.5)
#                 fig.update_xaxes(ticksuffix="")
#                 fig.update_yaxes(ticksuffix="")
#                 st.plotly_chart(fig, use_container_width=True)

#             plot_chart(estado, df)

#     st.write('Estados com mais eleitores analfabetos')
#     Q7, Q8 = st.columns(2)

#     with Q7:
#         top_10_analfabetos = df[['estado', 'analfabeto']].sort_values(
#             by='analfabeto', ascending=False)[:10]

#         x1 = top_10_analfabetos['estado']
#         analfabeto = top_10_analfabetos['analfabeto']

#         colors = ['#FCC202', '#E6DD39', '#FEE592', '#FEE592', '#E1E0C7',
#                   '#DEE0FC', '#A6ACE6', '#7179E6', '#597092', '#355070']

#         my_layout = Layout(hoverlabel=dict(
#             bgcolor='#FFFFFF'), template='simple_white')

#         fig_analfabetos = go.Figure(data=[
#             go.Bar(name='', x=x1, y=analfabeto, hovertemplate=' ',
#                    text=analfabeto, 
#                    textposition='outside',
#                    marker_color=colors, showlegend=False)],
#             layout=my_layout)
#         if (dados == 'Estadual - 1º turno'):
#             fig_analfabetos.update_layout(
#                 plot_bgcolor='rgba(0,0,0,0)',
#                 yaxis_range=[0, 850000],
#                 yaxis=dict(
#                     tickvals=[0, 100000, 200000, 300000, 400000,
#                               500000, 600000, 700000, 800000, ],
#                     ticktext=['0', '100 mil', '200 mil', '300 mil', '400 mil', '500 mil', '600 mil', '700 mil', '800 mil'])
#             )
#         else:
#             fig_analfabetos.update_layout(
#                 plot_bgcolor='rgba(0,0,0,0)',
#                 yaxis_range=[0, 300000],
#                 yaxis=dict(
#                     tickvals=[0, 50000, 100000, 150000,
#                               200000, 250000, 300000],
#                     ticktext=['0', '50 mil', '100 mil', '150 mil',
#                               '200 mil', '250 mil', '300 mil'])
#             )

#         st.plotly_chart(fig_analfabetos, use_container_width=True)

#     with Q8:
#         def plot_chart(estadoIndex, df):
#             estado = format_func_estado(estadoIndex)
#             valor = df.loc[(df['estado'] == estado), 'analfabeto'].values[0]
            
#             fig = go.Figure(
#                 go.Indicator(
#                     value=valor,
#                     title={'text': f"Número de eleitores analfabetos em {estado}"},
#                     number={'font_color': '#355070',
#                             'font_size': 80, "valueformat": ".0f"},
#                     align='center'))

#             st.plotly_chart(fig, use_container_width=True)

#         plot_chart(estado, df)
    
#     st.write('Estados com maior percentual de eleitores analfabetos')
#     Q9, Q10 = st.columns(2)
#     with Q9:
#         top_10_analfabetos_percentual = df[['estado', 'analfabeto_percentual(%)']].sort_values(
#             by='analfabeto_percentual(%)', ascending=False)[:10]

#         x1 = top_10_analfabetos_percentual['analfabeto_percentual(%)']
#         analfabeto = top_10_analfabetos_percentual['estado']

#         colors = ['#FCC202', '#E6DD39', '#FEE592', '#FEE592', '#E1E0C7',
#                 '#DEE0FC', '#A6ACE6', '#7179E6', '#597092', '#355070']

#         my_layout = Layout(hoverlabel=dict(
#             bgcolor='#FFFFFF'), template='simple_white')

#         fig_percentual_analfabetos = go.Figure(data=[
#             go.Bar(name='', x=x1, y=analfabeto, 
#                 hovertemplate=[f"{percent:.2f}%" for percent in x1],
#                 text=[f"{percent:.2f}%" for percent in x1], 
#                 textposition='outside', 
#                 marker_color=colors, showlegend=False,
#                 orientation='h',)], layout=my_layout)

#         fig_percentual_analfabetos.update_layout(
#             xaxis_range=[0, 100]
#         )

#         fig_percentual_analfabetos.update_layout(
#             plot_bgcolor='rgba(0,0,0,0)',
#             xaxis=dict(
#                 showgrid=False,
#                 zeroline=False,
#                 tickvals=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
#                 ticktext=['0%', '10%', '20%', '30%', '40%', '50%', 
#                     '60%', '70%', '80%', '90%', '100%']),
#             yaxis=dict(showgrid=False, zeroline=False, autorange="reversed"))

#         st.plotly_chart(fig_percentual_analfabetos, use_container_width=True)
#     with Q10:
#         def plot_chart(estadoIndex, df):
#             estado = format_func_estado(estadoIndex)
#             valor = df.loc[(df['estado'] == estado), 'analfabeto_percentual(%)'].values[0]
            
#             fig = go.Figure(
#                 go.Indicator(
#                     value=valor,
#                     title={'text': f"Percentual de eleitores analfabetos em {estado}"},
#                     number={'font_color': '#355070', 'suffix':'%',
#                             'font_size': 80, "valueformat": ".2f"},
#                     align='center'))

#             st.plotly_chart(fig, use_container_width=True)

#         plot_chart(estado, df)
    
#     st.write('Estados com mais eleitores com deficiência')
#     Q11, Q12 = st.columns(2)

#     with Q11:
#         top_10_deficiencia = df[['estado', 'eleitores_deficiencia']].sort_values(
#             by='eleitores_deficiencia', ascending=False)[:10].reset_index()

#         colors = ['#FCC202', '#E6DD39', '#FEE592', '#FEE592', '#E1E0C7',
#                   '#DEE0FC', '#A6ACE6', '#7179E6', '#597092', '#355070']

#         my_layout = Layout(hoverlabel=dict(
#             bgcolor='#FFFFFF'), template='simple_white')

#         fig_deficientes = go.Figure()
#         fig_deficientes.update_xaxes(title_text=' ')
#         fig_deficientes.update_yaxes(title_text=' ')
#         fig_deficientes.update_layout(
#             plot_bgcolor='rgba(0,0,0,0)',
#             xaxis=dict(
#                 showline=True,
#                 showgrid=False,
#                 zeroline=False,
#             ),
#             yaxis=dict(
#                 mirror=False,
#                 showline=True,
#                 showgrid=False,
#                 zeroline=False,
#                 tickvals=[0, 50000, 100000, 150000, 200000,
#                           250000, 300000, 350000, 400000, 450000],
#                 ticktext=['0', '50 mil', '100 mil', '150 mil',
#                           '200 mil', '250 mil', '300 mil', '350 mil', '400 mil', '450 mil']
#             ))

#         # pontos
#         fig_deficientes.add_trace(
#             go.Scatter(
#                 x=top_10_deficiencia["estado"],
#                 y=top_10_deficiencia["eleitores_deficiencia"],
#                 mode='markers+text',
#                 name='',
#                 text=top_10_deficiencia["eleitores_deficiencia"],
#                 textposition='top center',
#                 hovertemplate='%{y}',
#                 marker_color=colors,
#                 marker_size=25))
#         # linhas
#         for i, v in top_10_deficiencia["eleitores_deficiencia"].items():
#             fig_deficientes.add_shape(
#                 type='line',
#                 x0=i, y0=0,
#                 x1=i,
#                 y1=v,
#                 line=dict(color=colors[i], width=10))

#         st.plotly_chart(fig_deficientes, use_container_width=True)

#     with Q12:
#         def plot_chart(estadoIndex, df):
#             estado = format_func_estado(estadoIndex)
#             valor = df.loc[(df['estado'] == estado), 'eleitores_deficiencia'].values[0]
            
#             fig = go.Figure(
#                 go.Indicator(
#                     value=valor,
#                     title={'text': f"Número de eleitores com deficiência em {estado}"},
#                     number={'font_color': '#355070', 
#                             'font_size': 80, "valueformat": ".0f"},
#                     align='center'))

#             st.plotly_chart(fig, use_container_width=True)

#         plot_chart(estado, df)

#     st.write('Estados com maior percentual de eleitores com deficiência')
#     Q13, Q14 = st.columns(2)

#     with Q13:
#         top_10_deficiencia_percentual = df[['estado', 'eleitores_deficiencia_percentual(%)']].sort_values(
#             by='eleitores_deficiencia_percentual(%)', ascending=False)[:10].reset_index()

#         fig_percentual_deficientes = go.Figure()
#         fig_percentual_deficientes.update_layout(
#             plot_bgcolor='rgba(0,0,0,0)',
#             xaxis_range=[0, 100],
#             xaxis=dict(
#                 ticks="outside",
#                 mirror=False,
#                 showline=True,
#                 showgrid=False,
#                 zeroline=False,
#                 tickvals=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
#                 ticktext=['0%', '10%', '20%', '30%', '40%', '50%',
#                         '60%', '70%', '80%', '90%', '100%']),
#             yaxis=dict(
#                 ticks="outside",
#                 mirror=False,
#                 showline=True,
#                 showgrid=False,
#                 zeroline=False,
#                 autorange="reversed"))

#         # pontos
#         fig_percentual_deficientes.add_trace(
#             go.Scatter(
#                 x=top_10_deficiencia_percentual["eleitores_deficiencia_percentual(%)"],
#                 y=top_10_deficiencia_percentual["estado"],
#                 mode='markers+text',
#                 name='',
#                 text=[f"{percent:.2f}%" for percent in top_10_deficiencia_percentual["eleitores_deficiencia_percentual(%)"]],
#                 textposition='middle right',
#                 hovertemplate=[
#                     f"{percent:.2f}%" for percent in top_10_deficiencia_percentual['eleitores_deficiencia_percentual(%)']],
#                 marker_color=colors,
#                 marker_size=20))
#         # linhas
#         for i, v in top_10_deficiencia_percentual['eleitores_deficiencia_percentual(%)'].items():
#             fig_percentual_deficientes.add_shape(
#                 type='line',
#                 x0=0, y0=i,
#                 x1=v,
#                 y1=i,
#                 line=dict(color=colors[i], width=8))

#         st.plotly_chart(fig_percentual_deficientes, use_container_width=True)

#     with Q14:
#         def plot_chart(estadoIndex, df):
#             estado = format_func_estado(estadoIndex)
#             valor = df.loc[(df['estado'] == estado), 'eleitores_deficiencia_percentual(%)'].values[0]
            
#             fig = go.Figure(
#                 go.Indicator(
#                     value=valor,
#                     title={'text': f"Percentual de eleitores com deficiência em {estado}"},
#                     number={'font_color': '#355070', 'suffix': '%', 
#                             'font_size': 80, "valueformat": ".2f"},
#                     align='center'))

#             st.plotly_chart(fig, use_container_width=True)

#         plot_chart(estado, df)
#         # st.write('Estados com maior percentual de eleitores com deficiência')
#         # 

#     st.write('Eleitorado facultativo por estado')

#     with st.empty():
#         facultativos = df[['estado', 'eleitorado_facultativo_percentual(%)', '16_anos_percentual(%)', '17_anos_percentual(%)',
#                            '65_69_anos_percentual(%)', '70_74_anos_percentual(%)', '75_79_anos_percentual(%)',
#                            '80_84_anos_percentual(%)', '85_89_anos_percentual(%)', '90_94_anos_percentual(%)',
#                            '95_99_anos_percentual(%)', '100_anos_percentual(%)']].sort_values(by='eleitorado_facultativo_percentual(%)', ascending=False)

#         jovens = facultativos['16_anos_percentual(%)'] + \
#             facultativos['17_anos_percentual(%)']
#         idosos = facultativos['65_69_anos_percentual(%)'] + facultativos['70_74_anos_percentual(%)'] + facultativos['75_79_anos_percentual(%)'] + facultativos['80_84_anos_percentual(%)'] + \
#             facultativos['85_89_anos_percentual(%)'] + facultativos['90_94_anos_percentual(%)'] + \
#             facultativos['95_99_anos_percentual(%)'] + \
#             facultativos['100_anos_percentual(%)']

#         colors = ['#FCC202', '#355070']

#         df = facultativos
#         df['jovens'] = jovens
#         df['idosos'] = idosos
#         df['total'] = jovens + idosos

#         my_layout = Layout(hoverlabel=dict(
#             bgcolor='#FFFFFF'), template='simple_white')

#         fig_facultativo = go.Figure(layout=my_layout)

#         fig_facultativo.add_trace(go.Scatter(
#             x=facultativos['estado'],
#             y=df['idosos'],
#             hovertemplate=[f"{percent:.2f}%" for percent in df['idosos']],
#             marker=dict(color="#FCC202"),
#             name='Idosos',
#             showlegend=True))

#         fig_facultativo.add_trace(go.Scatter(
#             x=df['estado'],
#             y=df['jovens'],
#             hovertemplate=[f"{percent:.2f}%" for percent in df['jovens']],
#             marker=dict(color="#355070"),
#             name='Jovens',
#             showlegend=True))

#         fig_facultativo.add_trace(go.Scatter(
#             x=facultativos['estado'],
#             y=df['total'],
#             hovertemplate=[f"{percent:.2f}%" for percent in df['total']],
#             marker=dict(color="#296B21"),
#             name='Total',
#             showlegend=True))

#         fig_facultativo.update_layout(hovermode="x unified", yaxis_range=[0, 100], plot_bgcolor='rgba(255,255,255,255)',
#                                       xaxis=dict(showgrid=False,
#                                                  zeroline=False),
#                                       yaxis=dict(showgrid=False,
#                                                  zeroline=False,
#                                                  tickvals=[
#                                                      0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
#                                                  ticktext=['0%', '10%', '20%', '30%', '40%',
#                                                            '50%', '60%', '70%', '80%', '90%', '100%']),
#                                       xaxis_title="",
#                                       yaxis_title="",
#                                       legend_title='Legenda',)

#         st.plotly_chart(fig_facultativo, use_container_width=True)
# else:
#     ESTADOS = (df['estado'].drop_duplicates())

#     def format_func_estado(option):
#         return ESTADOS[option]

#     estado = st.selectbox(
#         "", options=list(ESTADOS.keys()), 
#         format_func=format_func_estado,
#         key='estadosAnalfabetismoPercentualMunicipal')
#     municipios = list(df.loc[df['estado'] == str(
#         format_func_estado(estado)), 'municipio'])

#     indexMunicipio = st.selectbox(
#         "",
#         range(len(municipios)),
#         format_func=lambda x: municipios[x],
#         key='analfabetismoPercentualMunicipal')
            
#     # M U N I C Í P I O S
#     st.write('Comparecimento percentual por município')
#     if (dados == 'Municipal - 1º turno'):
#         Q1, Q2, Q3 = st.columns(3)

#         with Q1:
#             htmlFile = open(
#                 "Dashboard/data/charts/municipios_1turno/mapa1_municipios_1turno.html", 'r', encoding='utf-8')
#             source_code = htmlFile.read()
#             components.html(source_code, height=480)

#         with Q2:
#             htmlFile = open(
#                 "Dashboard/data/charts/municipios_1turno/mapa2_municipios_1turno.html", 'r', encoding='utf-8')
#             source_code = htmlFile.read()
#             components.html(source_code, height=480)

#         with Q3:
#             htmlFile = open(
#                 "Dashboard/data/charts/municipios_1turno/mapa3_municipios_1turno.html", 'r', encoding='utf-8')
#             source_code = htmlFile.read()
#             components.html(source_code, height=480)

#     else:
#         with st.empty():
#             htmlFile = open(
#                 "Dashboard/data/charts/municipios_2turno/mapa_municipios_2turno.html", 'r', encoding='utf-8')
#             source_code = htmlFile.read()
#             components.html(source_code, height=480)
    
#     Q4, Q5, Q55 = st.columns(3)

#     with Q4:
#         st.write('Divisão de sexos por município')

#         def plot_chart(estadoIndex, municipios, index, df):
#             estado = format_func_estado(estadoIndex)
#             cidade = [municipios[index]]

#             homens = [df.loc[(df['estado'] == estado)
#                 & (df['municipio'] == cidade[0]), 
#                 'eleitorado_masculino_percentual(%)'].values[0]]
            
#             mulheres = [df.loc[(df['estado'] == estado)
#                 & (df['municipio'] == cidade[0]), 
#                 'eleitorado_feminino_percentual(%)'].values[0]]

#             my_layout = Layout(hoverlabel=dict(
#                 bgcolor='#FFFFFF'), template='simple_white')

#             fig_sexoMunicipio = go.Figure()
#             fig_sexoMunicipio.add_trace(go.Bar(y=cidade, x=homens,
#                                                name='',
#                                                hovertemplate='Homens: %{x:.2f}%',
#                                                marker_color='#355070',
#                                                orientation='h'))

#             fig_sexoMunicipio.add_trace(go.Bar(y=cidade, x=mulheres,
#                                                hovertemplate='Mulheres: %{x:.2f}%',
#                                                marker_color='#FCC202',
#                                                name='',
#                                                orientation='h'))

#             fig_sexoMunicipio.update_layout(barmode='relative',
#                                             hoverlabel=dict(bgcolor='#FFFFFF'),
#                                             template='simple_white',
#                                             bargap=0, bargroupgap=0,
#                                             margin=dict(l=1, r=1, t=60, b=1),
#                                             xaxis_range=[0, 100],
#                                             xaxis=dict(
#                                                 tickvals=[0, 10, 20, 30, 40,
#                                                           50, 60, 70, 80, 90, 100],
#                                                 ticktext=['0%', '10%', '20%', '30%', '40%', '50%',
#                                                           '60%', '70%', '80%', '90%', '100%']))

#             fig_sexoMunicipio.update_traces(width=0.5)
#             fig_sexoMunicipio.update_xaxes(ticksuffix="")
#             fig_sexoMunicipio.update_yaxes(ticksuffix="")
#             st.plotly_chart(fig_sexoMunicipio, use_container_width=True)

#         plot_chart(estado, municipios, indexMunicipio, df)

#     with Q5:
#         #ESCOLARIDADE POR MUNICÍPIO
#         st.write('Divisão por escolaridade por município')

#         def plot_chart(estadoIndex, municipios, index, df):

#             estado = format_func_estado(estadoIndex)
#             cidade = [municipios[index]]
#             analfabeto = [df.loc[(df['estado'] == estado)
#                            & (df['municipio'] == cidade[0]), 'analfabeto_percentual(%)'].values[0]]
#             le_escreve = [df.loc[(df['estado'] == estado)
#                            & (df['municipio'] == cidade[0]), 'le_escreve_percentual(%)'].values[0]]
#             fundamental_incompleto = [df.loc[(df['estado'] == estado)
#                 & (df['municipio'] == cidade[0]), 
#                 'fundamental_incompleto_percentual(%)'].values[0]]
#             fundamental_completo = [df.loc[(df['estado'] == estado)
#                 & (df['municipio'] == cidade[0]), 
#                 'fundamental_completo_percentual(%)'].values[0]]
#             medio_incompleto = [df.loc[(df['estado'] == estado)
#                 & (df['municipio'] == cidade[0]), 'medio_incompleto_percentual(%)'].values[0]]
#             medio_completo = [df.loc[(df['estado'] == estado)
#                 & (df['municipio'] == cidade[0]), 
#                 'medio_completo_percentual(%)'].values[0]]
#             superior_incompleto = [df.loc[(df['estado'] == estado)
#                 & (df['municipio'] == cidade[0]), 
#                 'superior_incompleto_percentual(%)'].values[0]]
#             superior_completo = [df.loc[(df['estado'] == estado)
#                 & (df['municipio'] == cidade[0]), 
#                 'superior_completo_percentual(%)'].values[0]]

#             my_layout = Layout(hoverlabel=dict(
#                 bgcolor='#FFFFFF'), template='simple_white')

#             fig = go.Figure(data=[
#                 go.Bar(name='', x=cidade, y=analfabeto,
#                        hovertemplate='Analfabeto: {}%'.format(
#                        str(analfabeto[0]).replace('.', ',')), marker_color='#355070', showlegend=False),
#                 go.Bar(name='', x=cidade, y=le_escreve,
#                        hovertemplate='Lê e escreve: {}%'.format(
#                 str(le_escreve[0]).replace('.', ',')), marker_color='#597092', showlegend=False),
#                 go.Bar(name='', x=cidade, y=fundamental_incompleto,
#                        hovertemplate='Fundamental incompleto: {}%'.format(
#                 str(fundamental_incompleto[0]).replace('.', ',')), marker_color='#7179E6', showlegend=False),
#                 go.Bar(name='', x=cidade, y=fundamental_completo,
#                        hovertemplate='Fundamental completo: {}%'.format(
#                 str(fundamental_completo[0]).replace('.', ',')), marker_color='#DEE0FC', showlegend=False),
#                 go.Bar(name='', x=cidade, y=medio_incompleto,
#                        hovertemplate='Médio incompleto: {}%'.format(
#                 str(medio_incompleto[0]).replace('.', ',')), marker_color='#E9DEFC', showlegend=False),
#                 go.Bar(name='', x=cidade, y=medio_completo,
#                        hovertemplate='Médio completo: {}%'.format(
#                 str(medio_completo[0]).replace('.', ',')), marker_color='#FEE592', showlegend=False),
#                 go.Bar(name='', x=cidade, y=superior_incompleto,
#                        hovertemplate='Superior incompleto: {}%'.format(
#                 str(superior_incompleto[0]).replace('.', ',')), marker_color='#E6DD39', showlegend=False),
#                 go.Bar(name='', x=cidade, y=superior_completo,
#                        hovertemplate='Superior completo: {}%'.format(
#                 str(superior_completo[0]).replace('.', ',')), marker_color='#FCC202', showlegend=False)
#             ], layout=my_layout)

#             fig.update_layout(
#                 barmode='stack',
#                 bargap=0, bargroupgap=0,
#                 margin=dict(l=1, r=1, t=1, b=1),
#                 yaxis_range=[0, 100],
#                 yaxis=dict(
#                     tickvals=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
#                     ticktext=['0%', '10%', '20%', '30%', '40%', '50%',
#                               '60%', '70%', '80%', '90%', '100%']
#                 )
#             )

#             fig.update_traces(width=0.5)
#             fig.update_xaxes(ticksuffix="")
#             fig.update_yaxes(ticksuffix="")
#             st.plotly_chart(fig, use_container_width=True)

#         plot_chart(estado, municipios, indexMunicipio, df)
    
#     with Q55:
#         def plot_chart(estadoIndex, municipios, index, df):
#             estado = format_func_estado(estadoIndex)
#             cidade = municipios[index]

#             valor = df.loc[(df['estado'] == estado)
#                            & (df['municipio'] == cidade), 'comparecimento_percentual(%)'].values[0]
#             fig = go.Figure(
#                 go.Indicator(
#                     value=valor,
#                     title={'text': f"Comparecimento percentual em {cidade}"},
#                     number={'font_color': '#355070', "suffix":"%",
#                             'font_size': 70, "valueformat": ".2f"},
#                     align='center'))

#             st.plotly_chart(fig, use_container_width=True)

#         plot_chart(estado, municipios, indexMunicipio, df)

#     Q6, Q7 = st.columns(2)
#     with Q6:
#         st.write('Municípios com mais eleitores analfabetos')
#         top_10_analfabetos = df[['municipio', 'analfabeto']].sort_values(
#             by='analfabeto', ascending=False)[:10]

#         x1 = top_10_analfabetos['municipio']
#         analfabeto = top_10_analfabetos['analfabeto']

#         colors = ['#FCC202', '#E6DD39', '#FEE592', '#FEE592', '#E1E0C7',
#                   '#DEE0FC', '#A6ACE6', '#7179E6', '#597092', '#355070']

#         my_layout = Layout(hoverlabel=dict(
#             bgcolor='#FFFFFF'), template='simple_white')

#         fig_analfabetos = go.Figure(data=[
#             go.Bar(name='', x=x1, y=analfabeto, hovertemplate=' ',
#                    text=analfabeto, textposition='outside',
#                    marker_color=colors, showlegend=False)],
#             layout=my_layout)
#         if (dados == 'Municipal - 1º turno'):
#             fig_analfabetos.update_layout(
#                 plot_bgcolor='rgba(0,0,0,0)',
#                 yaxis_range=[0, 200000],
#                 yaxis=dict(
#                     tickvals=[0, 50000, 100000, 150000, 200000],
#                     ticktext=['0', '50 mil', '100 mil', '150 mil', '200 mil'])
#             )
#         elif (dados == 'Municipal - 2º turno'):
#             fig_analfabetos.update_layout(
#                 plot_bgcolor='rgba(0,0,0,0)',
#                 yaxis_range=[0, 200000],
#                 yaxis=dict(
#                     tickvals=[0, 50000, 100000, 150000,
#                               200000, 250000, 300000],
#                     ticktext=['0', '50 mil', '100 mil', '150 mil',
#                               '200 mil', '250 mil', '300 mil'])
#             )

#         st.plotly_chart(fig_analfabetos, use_container_width=True)

#     with Q7:
#         def plot_chart(estadoIndex, municipios, index, df):
#             estado = format_func_estado(estadoIndex)
#             cidade = municipios[index]

#             valor = df.loc[(df['estado'] == estado)
#                            & (df['municipio'] == cidade), 'analfabeto'].values[0]
#             fig = go.Figure(
#                 go.Indicator(
#                     value=valor,
#                     title={'text': f"Número de eleitores analfabetos em {cidade}"},
#                     number={'font_color': '#355070',
#                             'font_size': 80, "valueformat": ".0f"},
#                     align='center'))

#             st.plotly_chart(fig, use_container_width=True)

#         plot_chart(estado, municipios, indexMunicipio, df)
    
#     Q8, Q9 = st.columns(2)
#     with Q8:
#         st.write('Municípios com maior percentual de eleitores analfabetos')
#         top_10_analfabetos = df[['municipio', 'analfabeto_percentual(%)']].sort_values(
#             by='analfabeto_percentual(%)', ascending=False)[:10]

#         x1 = top_10_analfabetos['municipio']
#         analfabeto = top_10_analfabetos['analfabeto_percentual(%)']

#         colors = ['#FCC202', '#E6DD39', '#FEE592', '#FEE592', '#E1E0C7',
#                   '#DEE0FC', '#A6ACE6', '#7179E6', '#597092', '#355070']

#         my_layout = Layout(hoverlabel=dict(
#             bgcolor='#FFFFFF'), template='simple_white')

#         fig_analfabetos_percentual = go.Figure(data=[
#             go.Bar(name='', x=analfabeto, y=x1, hovertemplate=' ',
#                    text=[f"{percent:.2f}%" for percent in analfabeto], textposition='outside',
#                    marker_color=colors, showlegend=False, orientation='h')],
#             layout=my_layout)
#         fig_analfabetos_percentual.update_layout(
#                 plot_bgcolor='rgba(0,0,0,0)',
#                 yaxis=dict(autorange="reversed"),
#                 xaxis_range=[0, 100],
#                 xaxis=dict(
#                     tickvals=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
#                     ticktext=['0%', '10%', '20%', '30%', '40%', '50%',
#                     '60%', '70%', '80%', '90%', '100%'])
#             )
#         st.plotly_chart(fig_analfabetos_percentual, use_container_width=True)

#     with Q9:
#         #ANALFABETISMO PERCENTUAL MUNICIPAL
#         def plot_chart(estadoIndex, municipios, index, df):
#             estado = format_func_estado(estadoIndex)
#             cidade = municipios[index]

#             valor = df.loc[(df['estado'] == estado)
#                            & (df['municipio'] == cidade), 'analfabeto_percentual(%)'].values[0]
#             fig = go.Figure(
#                 go.Indicator(
#                     value=valor,
#                     title={'text': f"Percentual de eleitores analfabetos em {cidade}"},
#                     number={'font_color': '#355070', "suffix": "%",
#                             'font_size': 80, "valueformat": ".2f"},
#                     align='center'))

#             st.plotly_chart(fig, use_container_width=True)

#         plot_chart(estado, municipios, indexMunicipio, df)

#     Q10, Q11 = st.columns(2)
#     with Q10:
#         #ELEITORADO COM DEFICIÊNCIA
#         st.write('Municípios com mais eleitores com deficiência')
#         top_10_deficiencia = df[['municipio', 'eleitores_deficiencia']].drop_duplicates().sort_values(by = 'eleitores_deficiencia', ascending = False)[:10].reset_index()

#         colors = ['#FCC202', '#E6DD39', '#FEE592', '#FEE592', '#E1E0C7', '#DEE0FC', '#A6ACE6', '#7179E6', '#597092', '#355070']

#         my_layout = Layout(hoverlabel = dict(bgcolor = '#FFFFFF'), template='simple_white')

#         fig_deficientes = go.Figure()
#         fig_deficientes.update_xaxes(title_text=' ')
#         fig_deficientes.update_yaxes(title_text=' ')
#         fig_deficientes.update_layout(
#             plot_bgcolor='rgba(0,0,0,0)',
#             xaxis=dict(
#                 mirror=False,
#                 showline=True,
#                 showgrid=False,
#                 zeroline=False,),
#             yaxis=dict(
#                 showgrid=False, 
#                 zeroline=False,
#                 mirror=False,
#                 showline=True,
#                 tickvals = [0, 50000, 100000, 150000, 200000],
#                 ticktext = ['0', '50 mil', '100 mil', '150 mil', '200 mil']
#             ))

#         # pontos
#         fig_deficientes.add_trace(
#             go.Scatter(
#                 x = top_10_deficiencia["municipio"],
#                 y = top_10_deficiencia["eleitores_deficiencia"], 
#                 mode = 'markers+text',
#                 name='',
#                 text=top_10_deficiencia["eleitores_deficiencia"],
#                 textposition='top center',
#                 hovertemplate = '%{y}',
#                 marker_color =colors,
#                 marker_size  = 25))
#         # linhas
#         for i, v in top_10_deficiencia["eleitores_deficiencia"].items():
#             fig_deficientes.add_shape(
#                 type='line',
#                 x0 = i, y0 = 0,
#                 x1 = i,
#                 y1 = v,
#                 line=dict(color=colors[i], width = 10))
#         if (dados == 'Municipal - 1º turno'):
#             fig_deficientes.update_layout(
#             plot_bgcolor='rgba(0,0,0,0)',
#             yaxis_range=[0, 200000],
#             yaxis=dict(
#                 tickvals=[0, 25000, 50000, 75000, 
#                 100000, 125000, 150000, 175000, 200000],
#                 ticktext=['0', '25 mil', '50 mil', '75 mil', 
#                 '100 mil', '125 mil', '150 mil', '175 mil', '200 mil'])
#             )
#         elif (dados == 'Municipal - 2º turno'):
#             fig_deficientes.update_layout(
#                 plot_bgcolor='rgba(0,0,0,0)',
#                 yaxis_range=[0, 200000],
#                 yaxis=dict(
#                     tickvals=[0, 25000, 50000, 75000,
#                         100000, 125000,
#                         150000, 175000, 200000],
#                     ticktext=['0', '25 mil', '50 mil', '75 mil',
#                         '100 mil', '125 mil', '150 mil',
#                         '175 mil', '200 mil'])
#             )

#         st.plotly_chart(fig_deficientes, use_container_width=True)

#     with Q11:
#         #ELEITORADO MUNICIPAL COM DEFICIÊNCIA
#         def plot_chart(estadoIndex, municipios, index, df):
#             estado = format_func_estado(estadoIndex)
#             cidade = municipios[index]

#             valor = df.loc[(df['estado'] == estado)
#                            & (df['municipio'] == cidade), 'eleitores_deficiencia'].values[0]
#             fig = go.Figure(
#                 go.Indicator(
#                     value=valor,
#                     title={'text': f"Número de eleitores com deficiência em {cidade}"},
#                     number={'font_color': '#355070',
#                             'font_size': 80, "valueformat": ".0f"},
#                     align='center'))

#             st.plotly_chart(fig, use_container_width=True)

#         plot_chart(estado, municipios, indexMunicipio, df)

#     Q12, Q13 = st.columns(2)
#     with Q12:
#         st.write('Municípios com maior percentual de eleitores com deficiência')
#         top_10_deficiencia_percentual = df[['municipio', 'eleitores_deficiencia_percentual(%)']].sort_values(
#             by='eleitores_deficiencia_percentual(%)', ascending=False)[:10].reset_index()

#         fig_percentual_deficientes = go.Figure()
#         fig_percentual_deficientes.update_layout(
#             plot_bgcolor='rgba(0,0,0,0)',
#             xaxis_range=[0, 100],
#             xaxis=dict(
#                 ticks="outside",
#                 mirror=False,
#                 showline=True,
#                 showgrid=False,
#                 zeroline=False,
#                 tickvals=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
#                 ticktext=['0%', '10%', '20%', '30%', '40%', '50%',
#                 '60%', '70%', '80%', '90%', '100%']),
#             yaxis=dict(
#                 ticks="outside",
#                 mirror=False,
#                 showline=True,
#                 showgrid=False,
#                 zeroline=False,
#                 autorange="reversed"))

#         # pontos
#         fig_percentual_deficientes.add_trace(
#             go.Scatter(
#                 x=top_10_deficiencia_percentual["eleitores_deficiencia_percentual(%)"],
#                 y=top_10_deficiencia_percentual["municipio"],
#                 mode='markers+text',
#                 name='',
#                 text=[f"{percent:.2f}%" for percent in top_10_deficiencia_percentual["eleitores_deficiencia_percentual(%)"]],
#                 textposition='middle right',
#                 hovertemplate=[
#                     f"{percent:.2f}%" for percent in top_10_deficiencia_percentual['eleitores_deficiencia_percentual(%)']],
#                 marker_color=colors,
#                 marker_size=20))
#         # linhas
#         for i, v in top_10_deficiencia_percentual['eleitores_deficiencia_percentual(%)'].items():
#             fig_percentual_deficientes.add_shape(
#                 type='line',
#                 x0=0, y0=i,
#                 x1=v,
#                 y1=i,
#                 line=dict(color=colors[i], width=8))

#         st.plotly_chart(fig_percentual_deficientes, use_container_width=True)

#     with Q13:
#         def plot_chart(estadoIndex, municipios, index, df):
#             estado = format_func_estado(estadoIndex)
#             cidade = municipios[index]

#             valor = df.loc[(df['estado'] == estado)
#                            & (df['municipio'] == cidade), 'eleitores_deficiencia_percentual(%)'].values[0]
#             fig = go.Figure(
#                 go.Indicator(
#                     value=valor,
#                     title={'text': f"Percentual de eleitores com deficiência em {cidade}"},
#                     number={'font_color': '#355070', "suffix": "%",
#                             'font_size': 80, "valueformat": ".2f"},
#                     align='center'))

#             st.plotly_chart(fig, use_container_width=True)

#         plot_chart(estado, municipios, indexMunicipio, df)

#     Q14, Q15 = st.columns(2)

#     facultativos = df[['estado', 'municipio', 'eleitorado_facultativo_percentual(%)','16_anos_percentual(%)', '17_anos_percentual(%)', 
#       '65_69_anos_percentual(%)', '70_74_anos_percentual(%)', '75_79_anos_percentual(%)',
#        '80_84_anos_percentual(%)', '85_89_anos_percentual(%)', '90_94_anos_percentual(%)',
#        '95_99_anos_percentual(%)', '100_anos_percentual(%)']].sort_values(by = 'eleitorado_facultativo_percentual(%)', ascending = False)
    
#     with Q14:
#         st.write('Municípios com maior eleitorado facultativo')
#         jovens = facultativos['16_anos_percentual(%)'] + facultativos['17_anos_percentual(%)']
#         idosos = facultativos['65_69_anos_percentual(%)'] + facultativos['70_74_anos_percentual(%)'] + facultativos['75_79_anos_percentual(%)'] + facultativos['80_84_anos_percentual(%)'] + facultativos['85_89_anos_percentual(%)'] + facultativos['90_94_anos_percentual(%)'] + facultativos['95_99_anos_percentual(%)'] + facultativos['100_anos_percentual(%)']

#         colors = ['#FCC202', '#355070']

#         facultativos['jovens'] = jovens
#         facultativos['idosos'] = idosos
#         facultativos['total'] = jovens + idosos

#         my_layout = Layout(hoverlabel = dict(bgcolor = '#FFFFFF'), template='simple_white')

#         fig_facultativo = go.Figure(layout=my_layout)

#         fig_facultativo.add_trace(go.Scatter(
#             x = facultativos['municipio'][:10],
#             y = facultativos['idosos'][:10],
#             hovertemplate = [f"{percent:.2f}%" for percent in facultativos['idosos']],
#             marker=dict(color="#FCC202"),
#             name='Idosos',
#             showlegend = True))

#         fig_facultativo.add_trace(go.Scatter(
#             x = facultativos['municipio'][:10],
#             y = facultativos['jovens'][:10],
#             hovertemplate = [f"{percent:.2f}%" for percent in facultativos['jovens']],
#             marker=dict(color="#355070"),
#             name='Jovens',
#             showlegend=True))

#         fig_facultativo.add_trace(go.Scatter(
#             x = facultativos['municipio'][:10],
#             y = facultativos['total'][:10],
#             hovertemplate = [f"{percent:.2f}%" for percent in facultativos['total']],
#             marker=dict(color="#296B21"),
#             name='Total',
#             showlegend = True))

#         fig_facultativo.update_layout(hovermode="x unified", yaxis_range=[0,100], plot_bgcolor='rgba(255,255,255,255)',
#                         xaxis=dict(showgrid=False, 
#                                     zeroline=False), 
#                         yaxis=dict(showgrid=False, 
#                                     zeroline=False,
#                                     tickvals = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
#                                     ticktext = ['0%', '10%', '20%', '30%', '40%', 
#                                     '50%', '60%', '70%', '80%', '90%', '100%']),
#                         xaxis_title="",
#                         yaxis_title="",
#                         legend_title='Legenda',)

#         st.plotly_chart(fig_facultativo, use_container_width=True)
    
#     with Q15:
#         def plot_chart(estadoIndex, municipios, index, df):
#             estado = format_func_estado(estadoIndex)
#             cidade = municipios[index]

#             valor = facultativos.loc[(facultativos['estado'] == estado)
#                 & (facultativos['municipio'] == cidade), 'total'].values[0]
#             fig = go.Figure(
#                 go.Indicator(
#                     value=valor,
#                     title={'text': f"Percentual do eleitorado facultativo em {cidade}"},
#                     number={'font_color': '#355070', "suffix": "%",
#                             'font_size': 80, "valueformat": ".2f"},
#                     align='center'))

#             st.plotly_chart(fig, use_container_width=True)

#         plot_chart(estado, municipios, indexMunicipio, df)

    
