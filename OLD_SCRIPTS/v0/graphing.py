#### Change to adding the variables
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import os   
import pandas as pd
import datetime 
import math
import time
from sklearn.linear_model import LinearRegression
#### Change to adding the variables
def graph_annual(input_model):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x = input_model.df_annual.index,
        y = input_model.df_annual['tasmin'],
        marker_color = 'cornflowerblue',
        marker_size = 8,
        mode = 'lines',
        showlegend= False
    ))
    fig.add_trace(go.Scatter(
        x = input_model.df_annual.index,
        y = input_model.df_annual['tas'],
        marker_color = 'cornflowerblue',
        marker_size = 16,
        mode = 'lines',
        fill='tonexty',
        showlegend= False
    ))
    fig.add_trace(go.Scatter(
        x = input_model.df_annual.index,
        y = input_model.df_annual['tasmax'],
        marker_color = 'cornflowerblue',
        marker_size = 8,
        mode = 'lines',
        fill='tonexty',
        showlegend= False
    ))

    fig.update_layout(template = 'simple_white', title = input_model.model_name)
    fig.show()

def graph_variable(input_model, variable_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = input_model.df_data.index,
        y = input_model.df_data[variable_name],
        marker_color = 'cornflowerblue',
        marker_size = 16,
        mode = 'lines',
        showlegend= False
    ))
    fig.update_layout(template = 'simple_white', title = input_model.model_name+' '+variable_name)
    fig.show()

def graph_annual_variable(input_model, variable_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = input_model.df_annual.index,
        y = input_model.df_annual[variable_name],
        marker_color = 'cornflowerblue',
        marker_size = 16,
        mode = 'lines',
        showlegend= False
    ))
    fig.update_layout(template = 'simple_white', title = input_model.model_name+' '+variable_name)
    fig.show()


def graph_decade_variable(input_model, variable_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = input_model.df_decade.index,
        y = input_model.df_decade[variable_name],
        marker_color = 'cornflowerblue',
        marker_size = 16,
        mode = 'lines',
        showlegend= False
    ))
    fig.update_layout(template = 'simple_white', title = input_model.model_name+' '+variable_name)
    fig.show()


def regression(years, data):
    # Pass in panda series
    x = years.to_numpy()
    x = x.reshape((-1,1))
    y = data.to_numpy()
    #print('Performing Regression')
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x,y)
    intercept = model.intercept_
    coef = model.coef_
    return coef, intercept, r_sq

def graph_write_decade_variable(input_model, variable_name, unit):

    graph_start    = datetime.datetime(year = 2010, month = 1, day = 1, hour = 00, minute = 00, second = 00)
    graph_end      = datetime.datetime(year = 2100, month = 1, day = 1, hour = 00, minute = 00, second = 00)

    # Plotting the main figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = input_model.df_decade.index,
        y = input_model.df_decade[variable_name],
        name = 'Measured Values',
        marker_color = 'black',
        marker_size = 16,
        mode = 'lines',
    ))

    # Plotting the regression
    try:
        a, b, r_sq = regression(input_model.df_decade['Year'], input_model.df_decade[variable_name])
        n = 100
        df_regression = pd.DataFrame(0.0, index=range(n), columns = ['x', 'y'])
        x_datetime = pd.date_range(start = min(input_model.df_decade.index), end = max(input_model.df_decade.index), periods = n)
        for i in df_regression.index:
            df_regression.loc[i,'x'] = min(input_model.df_decade['Year']) + i*(max(input_model.df_decade['Year'])-min(input_model.df_decade['Year']))/n
            df_regression.loc[i,'y'] = a*df_regression.loc[i,'x']+ b

        fig.add_trace(go.Scatter(
            x=x_datetime,
            y=df_regression['y'],
            line=dict(color='firebrick', width=2, dash = 'dash'),
            mode='lines',
            name='Regression'
        ))

        #print(a) 
        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            x=0.1,
            y=1.1,
            text="Regression Slope: "+str(round(a[0],4)),
            showarrow=False,
            font=dict(size=14),
            )

    except: 
        a = 0.0
        b = 0.0
        r_sq = 0.0

    


    fig.update_xaxes(title = 'Decade', range = [graph_start, graph_end])
    fig.update_yaxes(title = unit)
    fig.update_layout(template = 'simple_white', title = input_model.model_name+' '+variable_name, legend=dict(orientation = "h"), height = 500, width = 1500)
    fig.write_image('output_plots/'+input_model.model_name+'_'+variable_name+'.png')