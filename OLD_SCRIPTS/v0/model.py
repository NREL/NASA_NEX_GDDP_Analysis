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

class model:
    def __init__(self, path, model_name, experiment_name):
        self.path = path
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.time_freq = 'day'
        #self.experiments = ['rcp45', 'rcp85']
        self.bool_data_compiled = False
        self.data_lat = 0.0
        self.data_lon = 0.0

        self.variables_to_compile = [
            'hurs',
            'pr', # precipitation
            'sfcWind', # wind
            'tas', 'tasmax', 'tasmin'] #temperature

        # Creating a placeholder while prototyping
        self.annual_variables = ['blank']
        self.decade_variables = ['blank']

        # Setting up dataframes to handle the variables
        if self.time_freq == 'day':
            date_time_span = pd.date_range(start = '2006-01-01 12:00:00', end = '2060-12-31 12:00:00', freq = 'd')
        self.df_data = pd.DataFrame(np.nan, columns = self.variables_to_compile, index = date_time_span)

        col = self.annual_variables
        date_time_span = pd.date_range(start = '2010-01-01 12:00:00', end = '2060-01-01 12:00:00', freq = 'y')
        self.df_annual = pd.DataFrame(np.nan, columns = self.variables_to_compile, index = date_time_span)

        col = self.decade_variables
        date_time_span = pd.date_range(start = '2010-01-01 12:00:00', end = '2060-01-01 12:00:00', freq = '10y')
        self.df_decade = pd.DataFrame(np.nan, columns = self.variables_to_compile, index = date_time_span)

        # Should run this on init and have access to it
        self.get_files_and_variables()

    # Pulls all the files and variables from a folder containing multiple
    # models, frequencies, and experiments
    def get_files_and_variables(self):
        self.file_list = []
        for variable in self.variables_to_compile:
            variable_path = self.path+'/'+self.experiment_name+'/r1i1p1f1/'+variable+'/'
            filenames = os.listdir(variable_path)            
            #print(filenames)

            for file in filenames:
                self.file_list.append(variable_path+file)

        # Saving files and variables to the data list
        #self.file_list = temp_file_list      
        #self.variable_list = np.unique(np.array(temp_variable_list))
        
    # For each file, reads a single file and retuns a series/dataframe with the index as datetime and values as the variable name
    def read_one_file(self, coord, filename, variable):
        handle = xr.open_dataset(filename)

        # Geting Lat Long
        lat = handle['lat'][...].values
        lon = handle['lon'][...].values-180
        ilat = np.argmin(np.abs(lat - coord[0]))
        ilon = np.argmin(np.abs(lon - coord[1]))
        #print(lat[ilat], lon[ilon])
        self.data_lat = lat[ilat]
        self.data_lon = lon[ilon]
        
        # Putting into a Data frame
        df = pd.DataFrame()
        time = handle['time'].values
        #time = xr.CFTimeIndex(time)
        #print(time)
        df.index = pd.Series(time)
        df[variable] = handle[variable][:,ilat, ilon].values

        # Performing temperature calculation if necessary
        if 'tas' in variable:
            df[variable] = df[variable]-273.0 # Converting to Celsius and adding model bias
        if 'pr' in variable:
            df[variable] = df[variable]*86400/25.4 # Converting to Inches
        if 'sfc'in variable:
            df[variable] = df[variable]*2.2369 #Converting to MPH

        return df[variable]


    # Stores all the data produced by reading one file into the data frame which can then carry the data with it through the model
    def compile_data(self, coord):
        print('Data Compiling for ', self.model_name)
        # Adding each data file to the internal data frame
        #for file in data_files:
        for file in self.file_list:
            #items = file.split('_')
            #variable_name = items[0]

            if 'hurs' in file:
                variable = 'hurs'
            elif 'pr' in file:
                variable = 'pr'
            elif 'sfcWind' in file:
                variable = 'sfcWind'
            elif 'tasmax' in file:
                variable = 'tasmax'
            elif 'tasmin' in file:
                variable = 'tasmin'
            elif 'tas' in file:
                variable = 'tas'
            else:
                variable = 'null'

            try:
                temp_series = self.read_one_file(coord, file, variable)
                self.df_data.loc[variable] = temp_series
            except:
                print(file)
                continue

        # Changing the data compiled boolean to true
        self.bool_data_compiled = True
        print('Data Compiled!')