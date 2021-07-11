import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import math
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout

app = dash.Dash()
server = app.server

def calculate_indicator_roc(data, current):
    return ((current[0] - data[19][0]) / (data[19][0])) * 100

def lstm_predict_future(data, modelName, indicatorArr, period):
    # model
    modelFileName = '../MODEL/' + modelName
    sorted(indicatorArr)
    for indicator in indicatorArr:
        if indicator == 'close':
            continue
        modelFileName = modelFileName + '_' + indicator
    modelFileName = modelFileName + '.h5'
    model = load_model(modelFileName)
        
    # data
    data = data[indicatorArr].values
    data = data[-60:]

    # scaled data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaledData = scaler.fit_transform(data)

    # model input
    modelInput = scaledData.reshape(-1, scaledData.shape[0], scaledData.shape[1])

    # predicted scaled value
    predictedScaledValue = model.predict(modelInput)

    # predicted value
    predictedValue = scaler.inverse_transform(np.tile(predictedScaledValue, (1, scaledData.shape[1])))[:, 0]
    
    return predictedValue

df= pd.read_csv("./stock_data.csv")

app.layout = html.Div([
   
    html.H1("Stock Price Analysis", style={"textAlign": "center"}),
   
    dcc.Tabs(id="tabs", children=[
       
        dcc.Tab(label='Stock Data', children=[
            html.Div([
                html.H1("Stock Price", 
                        style={'textAlign': 'center'}),
              
                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'Microsoft','value': 'MSFT'}], 
                             multi=True,
                             value=['MSFT'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='stockprice'),
                
                
                html.H1("Stock Market Volume", style={'textAlign': 'center'}),
         
                dcc.Dropdown(id='my-dropdown2',
                             options=[{'label': 'Microsoft','value': 'MSFT'}], 
                             multi=True,
                             value=['MSFT'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='volume')
            ], className="container"),
        ]),
        
        
        dcc.Tab(label='Stock Prediction',children=[
            html.Div([
                
                dcc.Dropdown(id='dropdown-company',
                     options=[{'label': 'Microsoft','value': 'MSFT'}], 
                     multi=False, placeholder="Choose company",value='MSFT',
                     style={"margin-left": "auto", "margin-top": "10px", "margin-bottom": "10px",
                            "margin-right": "auto", "width": "80%"}),
                
                dcc.Dropdown(id='dropdown-model',
                     options=[{'label': 'Extreme Gradient Boosting', 'value': 'XGBOOST'},
                              {'label': 'Recurrent Neural Network','value': 'RNN'}, 
                              {'label': 'Long Short Term Memory', 'value': 'LSTM'}], 
                     multi=False, placeholder="Choose model",value='LSTM',
                     style={"margin-left": "auto", "margin-top": "10px", "margin-bottom": "10px",
                            "margin-right": "auto", "width": "80%"}),
                
                dcc.Dropdown(id='dropdown-period',
                     options=[{'label': '15 minutes', 'value': 15}], 
                     multi=False, placeholder="Choose time period",value=15,
                     style={"margin-left": "auto", "margin-top": "10px", "margin-bottom": "10px",
                            "margin-right": "auto", "width": "80%"}),
  
                dcc.Dropdown(id='dropdown-indicator',
                     options=[{'label': 'Close Price','value': 'close'},
                              {'label': 'Price Rate of Change','value': 'ROC'}, 
                              {'label': 'Relative Strength Index', 'value': 'RSI'}, 
                              {'label': 'Moving Averages', 'value': 'MA'},
                              {'label': 'Bolling Bands', 'value': 'BB'}], 
                     multi=True, placeholder="Choose indicators",value=['close'],
                     style={"margin-left": "auto", "margin-top": "10px", "margin-bottom": "10px",
                            "margin-right": "auto", "width": "80%"}),
                
                html.Div([                
                    html.Button('Predict', 
                     id='button', 
                     style={"background-color": "#5DADE2", "border": "none", "color": "white", 
                            "padding": "15px 32px", "text-align": "center", "text-decoration": "none", 
                            "display": "inline-block", "font-size": "16px", 
                            "margin-left": "auto", "margin-top": "10px", 
                            "margin-bottom": "10px", "margin-right": "auto", "width": "20%"})
                ], style={"text-align": "center"}),

                dcc.Graph(id='predicted_graph')
                
            ])                

        ])


    ])
])


@app.callback(Output('stockprice', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"MSFT": "Microsoft"}
    trace1 = []
    trace2 = []
    trace3 = []
    trace4 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Open"],
                     mode='lines', opacity=0.8,
                     name=f'Open {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["High"],
                     mode='lines', opacity=0.7, 
                     name=f'High {dropdown[stock]}',textposition='bottom center'))
        trace3.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Low"],
                     mode='lines', opacity=0.6,
                     name=f'Low {dropdown[stock]}',textposition='bottom center'))
        trace4.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Close"],
                     mode='lines', opacity=0.5,
                     name=f'Close {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2, trace3, trace4]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Stock Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (USD)"})}
    return figure


@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"MSFT": "Microsoft",}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Volume"],
                     mode='lines', opacity=0.7,
                     name=f'Volume {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data, 
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Transactions Volume"})}
    return figure


@app.callback(    
    Output('predicted_graph', 'figure'),
               [Input('button', 'n_clicks')], 
               [
                   State('dropdown-company', 'value'), 
                   State('dropdown-model', 'value'),
                   State('dropdown-indicator', 'value'),
                   State('dropdown-period', 'value')
               ]
              )
def update_graph(n_clicks, companyName, modelName, indicatorArr, period):
    if companyName == None or modelName == None or indicatorArr == None or period == None:
        return null
    
    data = pd.read_csv("../DATA/" + companyName + '.csv')
    
    predictions = lstm_predict_future(data, modelName, indicatorArr, period)
    
    prediction_df = pd.Series(predictions)
    prediction_df = data['close'].append(pd.Series(predictions))
    prediction_df = prediction_df.reset_index()
    prediction_df = prediction_df.drop(columns=['index'], axis=1)
    prediction_df = prediction_df[0]
    prediction_df = prediction_df[-len(predictions):]
    
    figure={
        "data":[
            go.Scatter(
                x=data.index[-100:],
                y=data.close[-100:],
                mode='lines',
                name="Real Price"
            ),
            go.Scatter(
                x=prediction_df.index,
                y=prediction_df.values,
                mode='markers',
                name="Predicted Price"
            )
        ],
        "layout":go.Layout(
            title=f"Predicted stock price is {prediction_df.values[0]}.",
            xaxis={'title':'Data Point'},
            yaxis={'title':'Close Price'}
        )
    }
    return figure



if __name__=='__main__':
    app.run_server(debug=True, port=8060)