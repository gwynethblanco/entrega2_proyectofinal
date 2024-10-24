import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf, pacf
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera
from dash import dash_table

app = dash.Dash(__name__)
server = app.server

# Leer los datos
pm10 = pd.read_csv("https://raw.githubusercontent.com/nicollF/datos_dash/refs/heads/main/pm10Series.csv")
ermita17_19 = pd.read_csv("https://raw.githubusercontent.com/nicollF/datos_dash/refs/heads/main/Serie1719.csv")
newseries = pd.read_csv('https://raw.githubusercontent.com/nicollF/datos_dash/refs/heads/main/seriefinalpm10.csv')
data_pm10 = pd.read_csv('https://raw.githubusercontent.com/nicollF/datos_dash/refs/heads/main/seriepm10.csv')


#MODELO KNN
data_pm10['pm10_lag1'] = data_pm10['pm10'].shift(1)  # Un retraso
data_pm10['pm10_lag2'] = data_pm10['pm10'].shift(2)  # Dos retrasos
data_pm10['pm10_lag3'] = data_pm10['pm10'].shift(3)
data_pm10['pm10_lag4'] = data_pm10['pm10'].shift(4) 
data_pm10['pm10_lag5'] = data_pm10['pm10'].shift(5)  
data_pm10 = data_pm10.dropna()  # Eliminar las filas con valores NaN causados por los shifts

X = data_pm10[['pm10_lag1', 'pm10_lag2','pm10_lag3','pm10_lag4','pm10_lag5']]
y = data_pm10['pm10']

pipeline = Pipeline([
    ('scaler', StandardScaler()),   
    ('knn', KNeighborsRegressor()) 
])
tscv = TimeSeriesSplit(n_splits=5)
param_grid = {
    'knn__n_neighbors': [3,5,7,10,15,20,30]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='r2')
grid_search.fit(X, y)
predicc_knn = grid_search.predict(X)
#residuos
residuos_knn = y - predicc_knn

# MODELO LASSO
pipeline = Pipeline([
    ('scaler', StandardScaler()),   
    ('lasso', Lasso()) 
])
tscv = TimeSeriesSplit(n_splits=5)
param_grid = {
    'lasso__alpha': [0.001, 0.01, 0.1, 1.0, 10.0] 
}
grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='r2')
grid_search.fit(X, y)
predicc_lasso = grid_search.predict(X)
# Calcular los residuos
residuos_lasso = y - predicc_lasso


#MODELO RIGDE


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge()) 
])
tscv = TimeSeriesSplit(n_splits=5)
param_grid = {
    'ridge__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]  
}
grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='r2')
grid_search.fit(X, y)
predicc_ridge = grid_search.predict(X)
# Calcular los residuos
residuos_ridge = y - predicc_ridge

#-------------------------------------------------------------------------------

# Series de tiempo sin valores nulos
serie_pm10_wna = ermita17_19.dropna(subset=['pm10']).reset_index(drop=True)

# Crear las figuras de las series de tiempo
fig = px.line(pm10, x='fecha', y='pm10', title='Serie de tiempo PM10 2017-2022')
fig.update_traces(line_color='#473C8B')

fig2 = px.line(newseries, x='fecha', y='pm10', title='Serie de tiempo PM10 2017-2022 imputada')
fig2.update_traces(line_color='#473C8B')


#GRAFICAS DE LOS MODELOS 

# KNN --------------------------------------------------------
knnfig = px.line(data_pm10, x='fecha', y=[y, predicc_knn], 
              labels={'value': 'PM10', 'variable': 'Leyenda'})

# Personalizar el estilo de las líneas
knnfig.update_traces(line=dict(width=2))  # Grosor de las líneas
knnfig.update_traces(line_color='#473C8B', selector=dict(name='pm10_observado'))  # Color de la línea observada
knnfig.update_traces(line=dict(dash='dash'), selector=dict(name='pm10_predicho'))  # Estilo de línea discontinua para las predicciones

# Personalizar el layout
knnfig.update_layout(
    title='Predicción vs Observación de PM10 con el modelo K-NN',
    xaxis_title='Fecha',
    yaxis_title='Concentración de PM10',
    legend_title='Leyenda',
    template='plotly_white',  # Cambia el estilo del fondo
    hovermode='x unified'  # Muestra información de todas las líneas al pasar el ratón
)

lassofig = px.line(data_pm10, x='fecha', y=[y, predicc_lasso], 
              labels={'value': 'PM10', 'variable': 'Leyenda'})


# LASSO -----------------------------------------------
# Personalizar el estilo de las líneas
lassofig.update_traces(line=dict(width=2))  # Grosor de las líneas
lassofig.update_traces(line_color='#473C8B', selector=dict(name='pm10_observado'))  # Color de la línea observada
lassofig.update_traces(line=dict(dash='dash'), selector=dict(name='pm10_predicho'))  # Estilo de línea discontinua para las predicciones

# Personalizar el layout
lassofig.update_layout(
    title='Predicción vs Observación de PM10 con el modelo Lasso',
    xaxis_title='Fecha',
    yaxis_title='Concentración de PM10',
    legend_title='Leyenda',
    template='plotly_white',  # Cambia el estilo del fondo
    hovermode='x unified'  # Muestra información de todas las líneas al pasar el ratón
)

 
# RIGDE --------------------------------------------------------
ridgefig = px.line(data_pm10, x='fecha', y=[y, predicc_ridge], 
              labels={'value': 'PM10', 'variable': 'Leyenda'})

# Personalizar el estilo de las líneas
ridgefig.update_traces(line=dict(width=2))  # Grosor de las líneas
ridgefig.update_traces(line_color='#473C8B', selector=dict(name='pm10_observado'))  # Color de la línea observada
ridgefig.update_traces(line=dict(dash='dash'), selector=dict(name='pm10_predicho'))  # Estilo de línea discontinua para las predicciones

# Personalizar el layout
ridgefig.update_layout(
    title='Predicción vs Observación de PM10 con el modelo Ridge',
    xaxis_title='Fecha',
    yaxis_title='Concentración de PM10',
    legend_title='Leyenda',
    template='plotly_white',  # Cambia el estilo del fondo
    hovermode='x unified'  # Muestra información de todas las líneas al pasar el ratón
)






# Graficar autocorrelación (ACF) serie original
autocorr_values = acf(serie_pm10_wna['pm10'], nlags=100)
acf_plot = go.Figure()

acf_plot.add_trace(go.Bar(
    x=np.arange(len(autocorr_values)), 
    y=autocorr_values, 
    marker=dict(color='#4B0082')
))

acf_plot.update_layout(
    title="Autocorrelation Function (ACF) of PM10",
    xaxis_title="Lags",
    yaxis_title="ACF",
    height=600,
    width=700
)

# Graficar autocorrelación parcial (PACF) serie original hasta 50 rezagos
pacf_values = pacf(serie_pm10_wna['pm10'], nlags=100)
pacf_plot = go.Figure()

pacf_plot.add_trace(go.Bar(
    x=np.arange(len(pacf_values)), 
    y=pacf_values, 
    marker=dict(color='#483D8B')
))

pacf_plot.update_layout(
    title="Partial Autocorrelation Function (PACF) of PM10",
    xaxis_title="Lags",
    yaxis_title="PACF",
    height=600,
    width=700
)

# ACF y PACF para la serie imputada
autocorr_values_imputada = acf(newseries['pm10'], nlags=100)
acf_plot_imputada = go.Figure()

acf_plot_imputada.add_trace(go.Bar(
    x=np.arange(len(autocorr_values_imputada)), 
    y=autocorr_values_imputada, 
    marker=dict(color='#4B0082')
))

acf_plot_imputada.update_layout(
    title="Autocorrelation Function (ACF) of Imputed PM10 (2017-2023)",
    xaxis_title="Lags",
    yaxis_title="ACF",
    height=600,
    width=700
)

# Graficar autocorrelación parcial (PACF) serie imputada hasta 50 rezagos
pacf_values_imputada = pacf(newseries['pm10'], nlags=100)
pacf_plot_imputada = go.Figure()

pacf_plot_imputada.add_trace(go.Bar(
    x=np.arange(len(pacf_values_imputada)), 
    y=pacf_values_imputada, 
    marker=dict(color='#483D8B')
))

pacf_plot_imputada.update_layout(
    title="Partial Autocorrelation Function (PACF) of Imputed PM10 (2017-2023)",
    xaxis_title="Lags",
    yaxis_title="PACF",
    height=600,
    width=700
)



#cards
cards = [
    dbc.Card(
        [
            html.H2("Contexto:", className="card-title", style={"fontSize": "15px", "textAlign": "left"}),
            html.P("Se conoce como PM10 a aquellas partículas sólidas o líquidas de diferente composición que se encuentran dispersas en la atmósfera. Estas pueden ser causadas por consecuencias humanas o naturales. Las características más importantes de estas partículas tienen un diámetro aerodinámico menor que 10 u/m. Al ser estas partículas tan pequeñas, estas pueden ser inhaladas por nuestro sistema respiratorio, por esto se les conoce a estas partículas como fracción respirable. Es de interés el estudio de la presencia de estas partículas, puesto que generan efectos adversos sobre la salud, como asma agravada, función pulmonar reducida, irritación en las vías respiratorias, hasta muerte prematura en personas con enfermedades cardíacas o pulmonares. En este proyecto serán usados datos que han sido recolectados en las estaciones de muestreo de la calidad del aire por el DAGMA (Departamento Administrativo de Gestión del Medio Ambiente) de la ciudad de Cali.", className="card-text", style={"fontSize": "14px", "textAlign": "left"}),
        ],
        body=True,
        color="light",
        style={"border": "2px solid #6a0dad", "backgroundColor": "#e6e6fa", "borderRadius": "10px", "margin": "10px", "padding": "20px"}
    ),
]



# Datos de la tabla
tabla_data = pd.DataFrame({
    'Modelo': ['K-NN', 'Lasso', 'Ridge'],
    'MAE': [9.717867, 10.068336, 10.067056],
    'MSE': [186.206472, 203.764545, 203.763002],
    'RMSE': [13.645749, 14.274612, 14.274558],
    'r2': [0.598803, 0.560972, 0.560976],
    'Ljung-Box': [4.894530e-34, 3.478560e-01, 3.891170e-01],
    'Jarque-Bera': [0.0, 0.0, 0.0]
})

# Diseño de la tabla
tabla_modelos = dash_table.DataTable(
    data=tabla_data.to_dict('records'),
    columns=[{'name': col, 'id': col} for col in tabla_data.columns],
    style_cell={'textAlign': 'center', 'padding': '5px'},
    style_header={
        'backgroundColor': '#473C8B',
        'fontWeight': 'bold',
        'color': 'white'
    },
    style_data={
        'backgroundColor': '#f0f0f0',
        'color': '#000000'
    },
    style_table={'margin-top': '20px', 'width': '90%'} 
)



# Diseño de la aplicación con pestañas
app.layout = html.Div(style ={'backgroundColor': '#f0f0f0'}, children=[
    html.H1('Análisis de Series de Tiempo de PM10', style={'textAlign': 'center', 'color': '#473C8B'}),
    
    dcc.Tabs(className='custom-tabs', children=[
        dcc.Tab(label='EDA', className='tab', selected_className='tab--selected', children=[
            html.Div(style={'padding': '50px'}, children=[
                html.H3('Exploratory Data Analysis', style={'color': '#473C8B'}),
                
                html.Div(style={'display': 'flex', 'justifyContent': 'space-around', 'margin-bottom': '20px'}, children=cards),
    
                # Dropdown de opciones
                html.Div(style={'display': 'flex', 'flex-direction': 'row', 'align-items': 'start'}, children=[
                    dcc.Dropdown(
                        id='serie-dropdown',
                        options=[
                            {'label': 'Serie Original', 'value': 'original'},
                            {'label': 'Serie Imputada', 'value': 'imputada'},
                        ],  
                        value='original',  # Valor por defecto
                        placeholder='Selecciona una opción',
                        style={'width' : "40%"}

                    ),
                    html.Label('serie pm10',
                        style={
                        'margin-top': '10px',
                        'padding': '10px',  # Espaciado interno
                        'background-color': '#f0f0f0',  # Fondo
                        'border': '2px solid #9370DB',  # Borde
                        'border-radius': '5px',  # Bordes redondeados
                        'color': '#9370DB',  # Color del texto
                        'font-weight': 'bold',  # Texto en negrita
                        'width': 'fit-content',  # Ajusta el ancho al contenido
                        'box-shadow': '2px 2px 5px rgba(0, 0, 0, 0.2)'  # Sombra para dar profundidad
        }),
                     html.Label(
                            'Total de datos: 473256', 
                            style={
                                'margin-top': '10px',
                                'padding': '10px',
                                'background-color': '#f0f0f0',
                                'border': '2px solid #9370DB',  # Cambié el color del borde
                                'border-radius': '5px',
                                'color': '#9370DB',  # Cambié el color del texto
                                'font-weight': 'bold',
                                'width': 'fit-content',
                                'box-shadow': '2px 2px 5px rgba(0, 0, 0, 0.2)'
                            }
    ),
                  # Tercera etiqueta decorada
                        html.Label(
                            'Estación Ermita', 
                            style={
                                'margin-top': '10px',
                                'padding': '10px',
                                'background-color': '#f0f0f0',
                                'border': '2px solid #9370DB',  # Borde de otro color
                                'border-radius': '5px',
                                'color': '#9370DB',  # Texto en color diferente
                                'font-weight': 'bold',
                                'width': 'fit-content',
                                'box-shadow': '2px 2px 5px rgba(0, 0, 0, 0.2)'
                            }
                        )
                    
                ]),
                    html.Div(id='graphs-container', style={'margin-top': '20px'}) 
            ])
        ]),
        
        dcc.Tab(label='MODELOS', className='tab', selected_className='tab--selected',style={'backgroundColor': '#D3D3D3'}, children=[
            html.Div(style={'padding': '20px'}, children=[
                html.H3('Modelos', style={'color': '#473C8B'}),
                        dcc.Dropdown(
                        id='modelos-dropdown',
                        options=[
                            {'label': 'Modelo K-nn', 'value': 'k-nn'},
                            {'label': 'Modelo Lasso', 'value': 'lasso'},
                            {'label': 'Modelo Ridge', 'value': 'ridge'},
                        ],
                        value='k-nn',  # Valor por defecto
                        placeholder='Selecciona una opción',
                        style={'width' : "40%"}

                    ),
                    tabla_modelos, 
                         
                        html.Div(id='model-graphs-container', style={'margin-top': '20px'})
                        
                 ])            
                        
            ])
        ])
    ])
fig.update_layout(  # Fondo de la "hoja" donde está la gráfica
    font_color='#473C8B',  # Color del texto
    title_font=dict(size=20, color='#473C8B'),
    xaxis=dict(showgrid=True),
    yaxis=dict(showgrid=True)
)

# Callback para actualizar la gráfica según la selección de los dropdowns
@app.callback(
    Output('graphs-container', 'children'),  # Actualiza las gráficas
    Input('serie-dropdown', 'value')    # Se activa al cambiar la selección de la serie  
)
def update_graph(selected_option):
    graphs = []
    
    # Gráficas de las series de tiempo según el dropdown de serie
    if selected_option == 'original':
        graphs.append(dcc.Graph(figure=fig))  # Serie original
        # Colocar ACF y PACF en una fila para la serie original
        graphs.append(html.Div(style={'display': 'flex', 'justifyContent': 'space-between'}, children=[
            dcc.Graph(figure=acf_plot),
            dcc.Graph(figure=pacf_plot)
        ]))
    elif selected_option == 'imputada':
        graphs.append(dcc.Graph(figure=fig2))  # Serie imputada
        # Colocar ACF y PACF en una fila para la serie imputada
        graphs.append(html.Div(style={'display': 'flex', 'justifyContent': 'space-between'}, children=[
            dcc.Graph(figure=acf_plot_imputada),
            dcc.Graph(figure=pacf_plot_imputada)
        ]))

    return graphs

# Callback para actualizar la gráfica según la selección del dropdown de los modelos
@app.callback(
    Output('model-graphs-container', 'children'),  # Actualiza las gráficas de los modelos
    [Input('modelos-dropdown', 'value')]           # Se activa al cambiar la selección del modelo
)
def update_model_graph(selected_model):
    if selected_model == 'k-nn':
        return dcc.Graph(figure=knnfig)  # Gráfica del modelo KNN
    elif selected_model == 'lasso':
        return dcc.Graph(figure=lassofig)  # Gráfica del modelo Lasso
    elif selected_model == 'ridge':
        return dcc.Graph(figure=ridgefig)  # Gráfica del modelo Ridge
    else:
        return html.P("Selecciona un modelo para visualizar su predicción.")



# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server()
    #app.run_server(debug=True, host='0.0.0.0', port=9000)
