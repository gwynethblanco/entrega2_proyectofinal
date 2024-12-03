import pickle
import plotly.graph_objects as go
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera
from scipy.stats import jarque_bera
import pandas as pd
import numpy as np
from dash import dash_table
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
import plotly.graph_objects as go
import plotly.express as px

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

# Leer los datos
pm10 = pd.read_csv("https://raw.githubusercontent.com/nicollF/datos_dash/refs/heads/main/pm10Series.csv")
ermita17_19 = pd.read_csv("https://raw.githubusercontent.com/nicollF/datos_dash/refs/heads/main/Serie1719.csv")
newseries = pd.read_csv('https://raw.githubusercontent.com/nicollF/datos_dash/refs/heads/main/seriefinalpm10.csv')  # Serie imputada del pm10

# Series de tiempo sin valores nulos
serie_pm10_wna = ermita17_19.dropna(subset=['pm10']).reset_index(drop=True)

# Gráfica de la serie de tiempo original
fig_original = px.line(pm10, x='fecha', y='pm10', title='Serie de tiempo PM10 2017-2022')
fig_original.update_traces(line_color='#2E8B57')  # Verde oscuro
fig_original.update_layout(template="plotly_white")

# ACF y PACF para la serie original
autocorr_values = acf(serie_pm10_wna['pm10'], nlags=10)
pacf_values = pacf(serie_pm10_wna['pm10'], nlags=10)

acf_plot = go.Figure()
acf_plot.add_trace(go.Bar(
    x=np.arange(len(autocorr_values)),
    y=autocorr_values,
    marker=dict(color='#2E8B57')
))
acf_plot.update_layout(
    title="(ACF) of PM10",
    xaxis_title="Lags",
    yaxis_title="ACF",
    height=400,
    template="plotly_white" 
)

pacf_plot = go.Figure()
pacf_plot.add_trace(go.Bar(
    x=np.arange(len(pacf_values)),
    y=pacf_values,
    marker=dict(color='#2E8B57')
))
pacf_plot.update_layout(
    title="(PACF) of PM10",
    xaxis_title="Lags",
    yaxis_title="PACF",
    height=400,
    template="plotly_white" 
)

# Gráfica de la serie imputada
fig_imputed = px.line(newseries, x='fecha', y='pm10', title='Serie Imputada PM10 2017-2023')
fig_imputed.update_traces(line_color='#2E8B57')
fig_imputed.update_layout(template="plotly_white")

# ACF y PACF para la serie imputada
autocorr_values_imputada = acf(newseries['pm10'], nlags=10)
pacf_values_imputada = pacf(newseries['pm10'], nlags=10)

acf_plot_imputada = go.Figure()
acf_plot_imputada.add_trace(go.Bar(
    x=np.arange(len(autocorr_values_imputada)),
    y=autocorr_values_imputada,
    marker=dict(color='#2E8B57')
))
acf_plot_imputada.update_layout(
    title="(ACF) of Imputed PM10",
    xaxis_title="Lags",
    yaxis_title="ACF",
    height=400,
    template="plotly_white" 
)

pacf_plot_imputada = go.Figure()
pacf_plot_imputada.add_trace(go.Bar(
    x=np.arange(len(pacf_values_imputada)),
    y=pacf_values_imputada,
    marker=dict(color='#2E8B57')
))
pacf_plot_imputada.update_layout(
    title="(PACF) of Imputed PM10",
    xaxis_title="Lags",
    yaxis_title="PACF",
    height=400,
    template="plotly_white" 
)
 
# -------------------- DISTRIBUCIONES ---------------------------------

def crear_distribucion(data, title):
    distri = go.Figure()

    # Histograma
    distri.add_trace(
        go.Histogram(
            x=data,
            nbinsx=30,
            marker_color='#2E8B57',
            opacity=0.7,
            name=title
        )
    )

    # Aplicar el template predefinido de Plotly
    distri.update_layout(
        title_text=title,  # Título del gráfico
        showlegend=False,   
        template='plotly_white',  
        xaxis_title="Concentración de PM10",
        yaxis_title="Frecuencia"
    )

    return distri
# ---------------------------------MODELOS-----------------------------------

# Cargar el diccionario desde el archivo pickle
with open('resultados_XGB.pkl', 'rb') as file:
    resultados_xgboost = pickle.load(file)

with open('resultados_SVR_final.pkl', 'rb') as file:
    resultados_svr = pickle.load(file)


with open('resultados_rf_final.pkl', 'rb') as file:
    resultados_rf = pickle.load(file)
    
with open('resultados_hibrido_xgb_svr.pkl', 'rb') as file:
    hybridmodel = pickle.load(file)
        
    
# Función para crear los gráficos con Plotly

def graficasmodelos(observed, predicted, training_data, title):
    fig = go.Figure()

    # observado
    fig.add_trace(go.Scatter(
        x=list(range(len(training_data), len(training_data) + len(observed))),
        y=observed,
        mode='lines',
        name='Observado',
        line=dict(color='#6bd425')
    ))

    # predicho
    fig.add_trace(go.Scatter(
        x=list(range(len(training_data), len(training_data) + len(predicted))),
        y=predicted,
        mode='lines',
        name='Predicho',
        line=dict(color='#d5573b', dash='dash')
    ))

    # train
    fig.add_trace(go.Scatter(
        x=list(range(len(training_data))),
        y=training_data,
        mode='lines',
        name='Entrenamiento',
        line=dict(color='#1e3f20')
    ))

    # deco
    fig.update_layout(
        title=title,
        xaxis_title="Índices de tiempo",
        yaxis_title="Valor",
        template="plotly_white",  
    )
    return fig

def graficasresiduos(resultados):
    graphs = []

    # Verificamos si es un modelo sin ventana
    if isinstance(resultados, dict) and len(resultados) == 1:
       
        datos = next(iter(resultados.values()))  
        
        # Acceder correctamente a los residuos
        residuo = datos["residuos"]

        # Crear la figura combinada con subplots en columnas
        fig = make_subplots(
            rows=1, cols=2,  
            shared_yaxes=False,  
            horizontal_spacing=0.2, 
            subplot_titles=("Histograma de residuos", "Autocorrelación")
        )

        # Histograma de residuos 
        fig.add_trace(
            go.Histogram(
                x=residuo,
                nbinsx=30,
                name="Histograma",
                marker=dict(color='#1e3f20'),
                opacity=0.6
            ),
            row=1, col=1
        )

        # Gráfico de autocorrelación 
        fig.add_trace(
            go.Scatter(
                x=list(range(len(sm.tsa.acf(residuo, nlags=7)))),
                y=sm.tsa.acf(residuo, nlags=7),  # ACF de residuos
                mode='markers+lines',
                name="Autocorrelación",
                line=dict(color='#1e3f20'),
                marker=dict(color='#1e3f20')
            ),
            row=1, col=2
        )

        # Ajustar el layout de la figura
        fig.update_layout(
            title="Residuos y Autocorrelación",
            template="plotly_white",
            showlegend=True,
            width=900,  # Ajustar el ancho total de la figura
            height=400  # Ajustar la altura total de la figura
        )

        # Centrar la figura usando un Div con estilo
        graphs.append(
            html.Div(
                dcc.Graph(figure=fig),
                style={
                    "display": "flex",
                    "justify-content": "center",
                    "align-items": "center",
                    "margin": "20px"
                }
            )
        )

    # Si hay más de una ventana, el código funciona como antes
    else:
        for k, datos in resultados.items():
            # Acceder correctamente a los residuos
            residuo = datos["residuos"]

            # Crear la figura combinada con subplots en columnas
            fig = make_subplots(
                rows=1, cols=2,  # Una fila, dos columnas
                shared_yaxes=False,  # No compartir ejes Y
                horizontal_spacing=0.2,  # Espaciado entre columnas
                subplot_titles=(f"Histograma de residuos - ventana {round(k/24)}", f"Autocorrelación - ventana {round(k/24)}")
            )

            # Histograma de residuos (en la primera columna)
            fig.add_trace(
                go.Histogram(
                    x=residuo,
                    nbinsx=30,
                    name="Histograma",
                    marker=dict(color='#1e3f20'),
                    opacity=0.6
                ),
                row=1, col=1
            )

            # Gráfico de autocorrelación 
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(sm.tsa.acf(residuo, nlags=7)))),
                    y=sm.tsa.acf(residuo, nlags=7),  # ACF de residuos
                    mode='markers+lines',
                    name="Autocorrelación",
                    line=dict(color='#1e3f20'),
                    marker=dict(color='#1e3f20')
                ),
                row=1, col=2
            )

            # Ajustar el layout de la figura
            fig.update_layout(
                title=f"Residuos y Autocorrelación - ventana {round(k/24)}",
                template="plotly_white",
                showlegend=True,
                width=900,  # Ajustar el ancho total de la figura
                height=400  # Ajustar la altura total de la figura
            )

            
            graphs.append(
                html.Div(
                    dcc.Graph(figure=fig),
                    style={
                        "display": "flex",
                        "justify-content": "center",
                        "align-items": "center",
                        "margin": "20px"
                    }
                )
            )

    return graphs


# gráficos para los diferentes modelos -------------------------------
# XGBoost
figxg_7 = graficasmodelos(resultados_xgboost[168]["observado"][:500], 
                             resultados_xgboost[168]["predicc"][:500],
                             resultados_xgboost[168]["entrenamiento y"][-100:], 
                             "XGBoost - Modelo 7 días")

figxg_14 = graficasmodelos(resultados_xgboost[336]["observado"][:500],
                                  resultados_xgboost[336]["predicc"][:500],
                                  resultados_xgboost[336]["entrenamiento y"][-100:], 
                                  "XGBoost - Modelo 14 días")

figxg_21 = graficasmodelos(resultados_xgboost[504]["observado"][:500], 
                              resultados_xgboost[504]["predicc"][:500],
                              resultados_xgboost[504]["entrenamiento y"][-100:], 
                              "XGBoost - Modelo 21 días")

figxg_28 = graficasmodelos(resultados_xgboost[672]["observado"][:500], 
                                   resultados_xgboost[672]["predicc"][:500],
                                   resultados_xgboost[672]["entrenamiento y"][-100:], 
                                   "XGBoost - Modelo 28 días")


# SVM GRAFICAS -----------------------------------------
figsvr_7 = graficasmodelos(resultados_svr[168]["observado"][:500], 
                             resultados_svr[168]["predicc"][:500],
                             resultados_svr[168]["entrenamiento y"][-100:], 
                             "SVR - Modelo 7 días")

figsvr_14 = graficasmodelos(resultados_svr[336]["observado"][:500],
                                  resultados_svr[336]["predicc"][:500],
                                  resultados_svr[336]["entrenamiento y"][-100:], 
                                  "SVR - Modelo 14 días")

figsvr_21 = graficasmodelos(resultados_svr[504]["observado"][:500], 
                              resultados_svr[504]["predicc"][:500],
                              resultados_svr[504]["entrenamiento y"][-100:], 
                              "SVR - Modelo 21 días")

figsvr_28 = graficasmodelos(resultados_svr[672]["observado"][:500], 
                                   resultados_svr[672]["predicc"][:500],
                                   resultados_svr[672]["entrenamiento y"][-100:], 
                                   "RF - Modelo 28 días")

# RANDOM FOREST GRAFICAS ----------------------------------------------------
figrf_7 = graficasmodelos(resultados_rf[168]["observado"][:500], 
                             resultados_rf[168]["predicc"][:500],
                             resultados_rf[168]["entrenamiento y"][-100:], 
                             "RF - Modelo 7 días")

figrf_14 = graficasmodelos(resultados_rf[336]["observado"][:500],
                                  resultados_rf[336]["predicc"][:500],
                                  resultados_rf[336]["entrenamiento y"][-100:], 
                                  "RF - Modelo 14 días")

figrf_21 = graficasmodelos(resultados_rf[504]["observado"][:500], 
                              resultados_rf[504]["predicc"][:500],
                              resultados_rf[504]["entrenamiento y"][-100:], 
                              "RF - Modelo 21 días")

figrf_28 = graficasmodelos(resultados_rf[672]["observado"][:500], 
                                   resultados_rf[672]["predicc"][:500],
                                   resultados_rf[672]["entrenamiento y"][-100:], 
                                   "RF - Modelo 28 días")

obs = hybridmodel['result']['observado'][:500]
predict = hybridmodel['result']['predicc'][:500]
train = hybridmodel['result']['entrenamiento y'][-100:]
titulo = "Hybrid - Modelo Original"

hybridgrafica = graficasmodelos(obs, predict, train, titulo)

# ------------------ funcion tabla ----------------------------------------------------------

def tablametricas(resultados, ventana=None, modelo=None):
    tabla = pd.DataFrame()

    # Verificar si el modelo es sin ventana o con ventana
    if isinstance(resultados, dict) and len(resultados) == 1:
        # Si no hay ventanas (solo un conjunto de resultados)
        datos = next(iter(resultados.values()))  # Obtener el primer (y único) valor
        real = datos['observado']
        predic = datos['predicc']
        residuos = datos['residuos']
        r2 = datos['score test']
        parametro = datos.get('parametros', 'N/A')

        # Cálculo de las métricas
        mae = mean_absolute_error(real, predic)
        mse = mean_squared_error(real, predic)
        rmse = np.sqrt(mse)

        # Ljung-Box y Jarque-Bera
        ljung_box_results = acorr_ljungbox(residuos, lags=[10], return_df=True)
        jb_stat, jb_pvalue = jarque_bera(residuos)

        # Crear un DataFrame temporal
        tabla_temp = pd.DataFrame({
            'Ventana': ['Modelo único'],  # Solo hay un modelo
            'parametro': [str(parametro)],
            'MAE': [round(mae,4)],
            'MSE': [round(mse,4)],
            'RMSE': [round(rmse,6)],
            'R2': [round(r2,5)],
            'Ljung-Box p-value': [round(ljung_box_results['lb_pvalue'].values[0],5)],
            'Jarque-Bera p-value': [round(jb_pvalue,5)]
        })

       
        tabla = pd.concat([tabla, tabla_temp], ignore_index=True)

    else:
        # Si hay ventanas, procesamos los resultados por ventana
        for clave, valor in resultados.items():
          
            if ventana is None or clave == ventana:
                real = valor['observado']
                predic = valor['predicc']
                residuos = valor['residuos']
                r2 = valor['score test']
                parametro = valor.get('parametros', 'N/A')

                # Cálculo de las métricas
                mae = mean_absolute_error(real, predic)
                mse = mean_squared_error(real, predic)
                rmse = np.sqrt(mse)

                # Ljung-Box y Jarque-Bera
                ljung_box_results = acorr_ljungbox(residuos, lags=[10], return_df=True)
                jb_stat, jb_pvalue = jarque_bera(residuos)

               
                tabla_temp = pd.DataFrame({
                    'Ventana': [int(clave / 24)],  
                    'parametro': [str(parametro)],
                    'MAE': [round(mae,4)],
                    'MSE': [round(mse,4)],
                    'RMSE': [round(rmse,6)],
                    'R2': [round(r2,5)],
                    'Ljung-Box p-value': [round(ljung_box_results['lb_pvalue'].values[0],5)],
                    'Jarque-Bera p-value': [round(jb_pvalue,5)]
                })

                # Concatenar el DataFrame temporal a la tabla final
                tabla = pd.concat([tabla, tabla_temp], ignore_index=True)

    return tabla



# ------------------------------------------Crear la aplicación Dash -----------------------------------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True)

# Estilos para la barra lateral
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "250px",
    "padding": "20px",
    "background-color": "#3CB371",
    "color": "white",
}

# Estilo para el contenido principal
CONTENT_STYLE = {
    "margin-left": "260px",
    "padding": "20px",
}

# Barra lateral
sidebar = html.Div(
    [
        html.Img(
            src="/assets/logopm10.png",
            style={"width": "100%", "margin-bottom": "20px"},
        ),
        html.H2("Opciones", className="text-center"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("EDA", href="/dash", active="exact"),
                dbc.NavLink("Modelos", href="/modelos", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

# Contenido principal
content = html.Div(id="page-content", style=CONTENT_STYLE)

# Layout de la aplicación
app.layout = html.Div(
    [
        dcc.Location(id="url"),
        sidebar,
        content,
    ]
)

# Callback para actualizar el contenido según la URL
@app.callback(
    dash.Output("page-content", "children"),
    [dash.Input("url", "pathname")],
)
def render_page_content(pathname):
    if pathname == "/dash":
        return html.Div(
            [
                html.H1("Análisis Exploratorio de Datos", style={"color": "#1b5e20"}),
                html.Label("Seleccione una opción:", style={"color": "#1b5e20"}),
                dcc.Dropdown(
                    id="eda-dropdown",
                    options=[{
                        "label": "Serie Original", "value": "original"},
                        {"label": "Serie Imputada", "value": "imputed"},
                    ],
                    value="original",
                    style={"width": "50%", "margin-bottom": "20px"},
                ),
                html.Div(id="eda-content", style={"margin-top": "20px"}),
            ]
        )
    elif pathname == "/modelos":
        return html.Div(
            [
                html.H1("Modelos de Predicción", style={"color": "#1b5e20"}),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Seleccione un modelo:", style={"color": "#1b5e20"}),
                                dcc.Dropdown(
                                    id="model-dropdown",
                                    options=[
                                        {"label": "Random Forest", "value": "rf"},
                                        {"label": "XGBoost", "value": "xgb"},
                                        {"label": "SVR", "value": "svr"},
                                        {"label": "Modelo Original", "value": "original"},
                                    ],
                                    value="rf",  # Valor inicial
                                    style={"width": "90%"},
                                ),
                            ],
                            style={"width": "48%", "margin-right": "2%"},
                        ),
                        html.Div(
                            [
                                html.Label("Seleccione ventana de días:", style={"color": "#1b5e20"}),
                                dcc.Dropdown(
                                    id="window-dropdown",
                                    options=[
                                        {"label": "7 días", "value": "7"},
                                        {"label": "14 días", "value": "14"},
                                        {"label": "21 días", "value": "21"},
                                        {"label": "28 días", "value": "28"},
                                    ],
                                    value=7,  # Valor inicial
                                    style={"width": "90%"},
                                ),
                            ],
                            style={"width": "48%"},
                        ),
                    ],
                    style={"display": "flex", "justify-content": "space-between", "margin-bottom": "20px"},
                ),
                html.Div(id="model-content", style={"margin-top": "20px"}),
            ]
        )
    return html.H1("Trabajo final Machine learning: PM10", style={"color": "#1b5e20"})




# ---------------------------- CALLBACK EDA ------------------------------------
@app.callback(
    dash.Output("eda-content", "children"),
    [dash.Input("eda-dropdown", "value")],
)
def update_eda_content(selected_value):
    if selected_value == "original":
        distri = crear_distribucion(pm10['pm10'], "Distribución Original")
        return html.Div(
            style={"display": "flex", "flex-direction": "column", "align-items": "center"},
            children=[
                # Primera fila con la distribución original y su gráfica
                html.Div(
                    style={"display": "flex", "justify-content": "space-between", "width": "80%", "margin-bottom": "20px"},
                    children=[
                        html.Div(dcc.Graph(figure=fig_original), style={"width": "70%"}),  # Gráfica de serie de tiempo
                        html.Div(dcc.Graph(figure=distri), style={"width": "40%"}),  # Gráfica de distribución
                    ]
                ),
                # Segunda fila con las gráficas de autocorrelación y PACF
                html.Div(
                    style={"display": "flex", "justify-content": "space-between", "width": "80%", "margin-top": "20px"},
                    children=[
                        html.Div(dcc.Graph(figure=acf_plot), style={"width": "48%"}),  # ACF
                        html.Div(dcc.Graph(figure=pacf_plot), style={"width": "48%"}),  # PACF
                    ]
                ),
            ]
        )
    
    elif selected_value == "imputed":
        distri = crear_distribucion(newseries['pm10'], "Distribución Imputada")
        return html.Div(
            style={"display": "flex", "flex-direction": "column", "align-items": "center"},
            children=[
                
                html.Div(
                    style={"display": "flex", "justify-content": "space-between", "width": "80%", "margin-bottom": "20px"},
                    children=[
                        html.Div(dcc.Graph(figure=fig_imputed), style={"width": "70%"}),  # Gráfica de serie de tiempo imputada
                        html.Div(dcc.Graph(figure=distri), style={"width": "40%"}),  # Gráfica de distribución imputada
                    ]
                ),
                # Segunda fila con las gráficas de autocorrelación y PACF
                html.Div(
                    style={"display": "flex", "justify-content": "space-between", "width": "80%", "margin-top": "20px"},
                    children=[
                        html.Div(dcc.Graph(figure=acf_plot_imputada), style={"width": "48%"}),  # ACF imputada
                        html.Div(dcc.Graph(figure=pacf_plot_imputada), style={"width": "48%"}),  # PACF imputada
                    ]
                ),
            ]
        )
    
    return html.Div()  


# ---------------------- CALLBACK MODELOS ---------------------------
@app.callback(
    dash.Output("model-content", "children"),
    [dash.Input("model-dropdown", "value"),  # Modelo seleccionado
     dash.Input("window-dropdown", "value")]  # Ventana seleccionada
)
def update_graph_for_model_and_window(selected_model, selected_window):
    # Asegurar un valor por defecto para selected_window
    if selected_window is None:
        selected_window = "7"

    selected_window = int(selected_window) * 24
    series_figures = []
    residual_graphs = []
    tabla_resultados = []

    # Selección de gráficas según modelo y ventana
    if selected_model == "xgb":
        if selected_window == 168:
            series_figures = [figxg_7]
            residual_graphs = graficasresiduos({168: resultados_xgboost[168]})
            tabla_resultados = tablametricas(resultados_xgboost, 168, "xgb")
        elif selected_window == 336:
            series_figures = [figxg_14]
            residual_graphs = graficasresiduos({336: resultados_xgboost[336]})
            tabla_resultados = tablametricas(resultados_xgboost, 336, "xgb")
        elif selected_window == 504:
            series_figures = [figxg_21]
            residual_graphs = graficasresiduos({504: resultados_xgboost[504]})
            tabla_resultados = tablametricas(resultados_xgboost, 504, "xgb")
        elif selected_window == 672:
            series_figures = [figxg_28]
            residual_graphs = graficasresiduos({672: resultados_xgboost[672]})
            tabla_resultados = tablametricas(resultados_xgboost, 672, "xgb")
    elif selected_model == "svr":
        if selected_window == 168:
            series_figures = [figsvr_7]
            residual_graphs = graficasresiduos({168: resultados_svr[168]})
            tabla_resultados = tablametricas(resultados_svr, 168, "svr")
        elif selected_window == 336:
            series_figures = [figsvr_14]
            residual_graphs = graficasresiduos({336: resultados_svr[336]})
            tabla_resultados = tablametricas(resultados_svr, 336, "svr")
        elif selected_window == 504:
            series_figures = [figsvr_21]
            residual_graphs = graficasresiduos({504: resultados_svr[504]})
            tabla_resultados = tablametricas(resultados_svr, 504, "svr")
        elif selected_window == 672:
            series_figures = [figsvr_28]
            residual_graphs = graficasresiduos({672: resultados_svr[672]})
            tabla_resultados = tablametricas(resultados_svr, 672, "svr")
    elif selected_model == "rf":
        if selected_window == 168:
            series_figures = [figrf_7]
            residual_graphs = graficasresiduos({168: resultados_rf[168]})
            tabla_resultados = tablametricas(resultados_rf, 168, "rf")
        elif selected_window == 336:
            series_figures = [figrf_14]
            residual_graphs = graficasresiduos({336: resultados_rf[336]})
            tabla_resultados = tablametricas(resultados_rf, 336, "rf")
        elif selected_window == 504:
            series_figures = [figrf_21]
            residual_graphs = graficasresiduos({504: resultados_rf[504]})
            tabla_resultados = tablametricas(resultados_rf, 504, "rf")
        elif selected_window == 672:
            series_figures = [figrf_28]
            residual_graphs = graficasresiduos({672: resultados_rf[672]})
            tabla_resultados = tablametricas(resultados_rf, 672, "rf")
    elif selected_model == "original":
        series_figures = [hybridgrafica]
        residual_graphs = graficasresiduos({0: hybridmodel["result"]})
        tabla_resultados = tablametricas({0: hybridmodel["result"]})
 
 

    else:
        return html.Div([html.H3("No hay datos aún.")])
    
    
    # Convertir tabla_resultados a DataFrame si es una lista de registros
    if isinstance(tabla_resultados, list):
        tabla_resultados = pd.DataFrame(tabla_resultados)

# Crear la tabla Dash ---------------------------------------------
        
    tabla_dash = dash_table.DataTable(
        id="model-table",
        columns=[{"name": col, "id": col} for col in tabla_resultados.columns],  # Nombres de las columnas
        data=tabla_resultados.to_dict("records"),  # Datos de la tabla
        
        # Estilo de la tabla (para altura y desplazamiento vertical)
        style_table={
            'height': 'auto',  # Altura ajustada
            'overflowY': 'auto',  # Desplazamiento vertical
            'boxShadow': '0px 4px 8px rgba(0, 0, 0, 0.1)',  # Sombra suave para profundidad
            'borderRadius': '8px',  # Bordes redondeados
        },
        
        # Estilo de las celdas
        style_cell={
            'padding': '10px',  # Espaciado dentro de las celdas
            'textAlign': 'center',  # Alineación de texto al centro
            'fontSize': '14px',  # Tamaño de la fuente
            'border': '1px solid #ddd',  # Bordes suaves para las celdas
            'width': '150px',  # Ancho de las columnas ajustado
            'maxWidth': '200px',  # Ancho máximo de las columnas
            'overflow': 'hidden',  # Evita el desbordamiento del texto
            'textOverflow': 'ellipsis',  # Puntos suspensivos para el texto largo
        },
        
        # Estilo de los encabezados
        style_header={
            'backgroundColor': '#4CAF50',  # Fondo verde claro para los encabezados
            'color': 'white',  # Texto blanco en los encabezados
            'fontWeight': 'bold',  # Texto en negrita
            'textAlign': 'center',  # Alineación al centro
            'border': '1px solid #ddd',  # Bordes suaves
            'fontSize': '16px',  # Tamaño de fuente más grande en los encabezados
            'padding': '12px',  # Espaciado en los encabezados
        },
        
        # Estilo de las filas
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},  # Filas impares
                'backgroundColor': '#f9f9f9',  # Fondo gris claro
            },
            {
                'if': {'row_index': 'even'},  # Filas pares
                'backgroundColor': '#ffffff',  # Fondo blanco
            },
        ],
        
        style_data={
            'whiteSpace': 'normal',  # Permite el salto de línea dentro de las celdas
            'overflow': 'visible',  # Permite el desbordamiento del texto
            'textOverflow': 'clip',  # Evita los puntos suspensivos
        },
    )


    
    # Renderizar las gráficas y la tabla seleccionada
    if selected_model == "original":
        return html.Div([
            html.H3(f"Modelo seleccionado: {selected_model.upper()}"),
            html.Hr(),  # Separador
            html.H4("Métricas:"),
            tabla_dash,  # Tabla de resultados
            html.Div([dcc.Graph(figure=fig) for fig in series_figures]),  # Observados/predichos
            html.Div(residual_graphs),  # Residuos
            html.Hr(),  # Separador
        ])

    # Para los demás modelos, incluir la ventana seleccionada
    return html.Div([
        html.H3(f"Modelo seleccionado: {selected_model.upper()}"),
        html.H3(f"Ventana seleccionada: {selected_window // 24} días"),
        html.Hr(),  # Separador
        html.H4("Métricas:"),
        tabla_dash,  # Tabla de resultados
        html.Div([dcc.Graph(figure=fig) for fig in series_figures]),  # Observados/predichos
        html.Div(residual_graphs),  # Residuos
        html.Hr(),  # Separador
    ])
    
# Ejecutar la app
if __name__ == '__main__':
    app.run_server()
    #app.run_server(debug=True, host='0.0.0.0', port=9000)
