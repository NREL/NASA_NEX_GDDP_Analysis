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

def analyze_model(input_model, thresholds_dict, cold_avg, hot_avg):
    # Year to 
    comparison_year_1 = 2010
    comparison_year_2 = 2050

    # Dropping NA values
    input_model.df_data.dropna(subset = ['tas', 'tasmin', 'tasmax'], inplace = True)
    #input_model.df_annual.index = pd.to_datetime(input_model.df_annual.index )

    # Dictionaries used in modeling
    months_dict = {
        1:'Jan',
        2:'Feb',
        3:'Mar',
        4:'Apr',
        5:'May',
        6:'Jun',
        7:'Jul',
        8:'Aug',
        9:'Sep',
        10:'Oct',
        11:'Nov',
        12:'Dec'
    }

    # Adding a Month and Year column to further analysis.
    input_model.df_data['Month'] = 0
    input_model.df_data['Year'] = 0
    for i in input_model.df_data.index:
        input_model.df_data.loc[i,'Month'] = i.month
        input_model.df_data.loc[i,'Year'] = i .year

    # Getting min and max year
    min_year = min(input_model.df_data['Year'])
    max_year = max(input_model.df_data['Year'])
    years = [*range(min_year,max_year)]

    # Getting variables to average for the annual average
    annual_average_variables = ['rhs', 
        'sfcWind', 'sfcWindmax', 
        'tas', 'tasmax', 'tasmin',
        ]
    annual_sum_variables = ['pr', 'prsn']
    #annual_extreme_variables = ['tasmax', 'tasmin', 'sfcWindmax']

    # Making a year variable for df_decade
    input_model.df_decade['Year'] = np.nan
    for time_stamp in input_model.df_decade.index:
        input_model.df_decade.loc[time_stamp, 'Year'] = time_stamp.year

    # Biasing the model
    baseline_start    = datetime.datetime(year = 2006, month = 1, day = 1, hour = 00, minute = 00, second = 00)
    baseline_end      = datetime.datetime(year = 2013, month = 1, day = 1, hour = 00, minute = 00, second = 00)
    mask = (input_model.df_data.index >= baseline_start) & (input_model.df_data.index < baseline_end)
    baseline_cold = input_model.df_data[mask]['tasmin'].mean()
    baseline_hot = input_model.df_data[mask]['tasmax'].mean()
    bias_cold = cold_avg - baseline_cold
    bias_hot = hot_avg - baseline_hot
    print('The cold bias is: ', bias_cold)
    print('The hot bias is: ', bias_hot)
    input_model.df_data['tasmin'] = input_model.df_data['tasmin'] + bias_cold
    input_model.df_data['tasmax'] = input_model.df_data['tasmax'] + bias_hot
    input_model.df_data['tas'] = input_model.df_data['tas'] + ((bias_hot+bias_cold)/2)
 
    # Calculating the heat index
    input_model.df_data['heatindex'] = np.nan
    c1 = -42.379
    c2 = 2.04901523
    c3 = 10.14333127
    c4 = -0.22475541
    c5 = -6.83783e-3
    c6 = -5.481717e-2
    c7 = 1.22874e-3 
    c8 = 8.5282e-4
    c9 = -1.99e-6

    for time_stamp in input_model.df_data.index:
        T = input_model.df_data.loc[time_stamp, 'tasmax']*9/5+32
        RH = input_model.df_data.loc[time_stamp, 'rhs']
        HI = 0.0
        if T < 80.0:
            HI = 0.5*(T+61.0+((T-68.0)*1.2)+RH*0.094)
        else:
            HI = c1 + c2*T + c3*RH + c4*T*RH + c5*T*T + c6*RH*RH + c7*T*T*RH + c8*T*RH*RH + c9*T*T*RH*RH

            if ((RH < 13) and (T>= 80) and (T<= 112)):
                adjustment = ((13-RH)/4)*(math.sqrt((17-abs(T-95))/17))
                HI = HI - adjustment
            elif ((RH > 85) and (T>= 80) and (T<= 87)):
                adjustment = ((RH-85)/10)*((87-T)/5)
                HI = HI - adjustment

        input_model.df_data.loc[time_stamp, 'heatindex'] = HI

    # Beginning Annual Compilation
    time_delta = datetime.timedelta(days = 365)
    # Calculating Annual Values
    for variable in annual_average_variables+annual_sum_variables: #+annual_extreme_variables:
        try:
            input_model.df_annual[variable] = np.nan
            for time_stamp in input_model.df_annual.index:
                mask = (input_model.df_data.index > time_stamp) & (input_model.df_data.index < time_stamp+time_delta)

                if variable in annual_average_variables:
                    input_model.df_annual.loc[time_stamp, variable] = input_model.df_data[mask][variable].mean()
                if variable in annual_sum_variables:
                    input_model.df_annual.loc[time_stamp, variable] = input_model.df_data[mask][variable].sum()
        except: 
            continue

    # Beginning Decade Compilation
    time_delta = datetime.timedelta(days = 3650)
    for variable in annual_average_variables+annual_sum_variables: #+annual_extreme_variables:
        try:
            input_model.df_decade[variable] = np.nan
            for time_stamp in input_model.df_decade.index:
                mask = (input_model.df_annual.index > time_stamp) & (input_model.df_annual.index < time_stamp+time_delta)
                input_model.df_decade.loc[time_stamp, variable] = input_model.df_annual[mask][variable].mean()
        except: 
            continue


    # Compiling Indicators 
    ##### A-Block #####
    ### Getting items a1: Decade average temperature change delta
    reference_time_stamp = input_model.df_decade.index[0]
    input_model.df_decade['Temperature_decade_average_delta'] = np.nan
    for time_stamp in input_model.df_decade.index:
        input_model.df_decade.loc[time_stamp, 'Temperature_decade_average_delta'] = input_model.df_decade.loc[time_stamp, 'tas'] - input_model.df_decade.loc[reference_time_stamp, 'tas']

    ### Getting items a2, a9, and a19: Decade average daily temperature range, windspeed, and maximum windspeed
    # Allocating data space 
    input_model.df_data['TemperatureRange_daily_average_raw'] = np.nan
    input_model.df_data['TemperatureRange_daily_average_raw'] = input_model.df_data['tasmax'] - input_model.df_data['tasmin']
    # Allocating decade data space
    input_model.df_decade['TemperatureRange_decade_average_raw'] = np.nan
    input_model.df_decade['TemperatureRange_decade_average_delta'] = np.nan
    input_model.df_decade['Windspeed_decade_average_raw'] = np.nan
    input_model.df_decade['Windspeed_decade_average_delta'] = np.nan
    input_model.df_decade['WindspeedMax_decade_average_raw'] = np.nan
    input_model.df_decade['WindspeedMax_decade_average_delta'] = np.nan
    time_delta = datetime.timedelta(days = 3650)
    for time_stamp in input_model.df_decade.index:
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta)
        input_model.df_decade.loc[time_stamp, 'TemperatureRange_decade_average_raw'] = input_model.df_data[mask]['TemperatureRange_daily_average_raw'].mean()
        input_model.df_decade.loc[time_stamp, 'Windspeed_decade_average_raw'] = input_model.df_data[mask]['sfcWind'].mean()
        input_model.df_decade.loc[time_stamp, 'WindspeedMax_decade_average_raw'] = input_model.df_data[mask]['sfcWindmax'].mean()
    for time_stamp in input_model.df_decade.index:
        input_model.df_decade.loc[time_stamp, 'TemperatureRange_decade_average_delta'] = input_model.df_decade.loc[time_stamp, 'TemperatureRange_decade_average_raw']-input_model.df_decade.loc[reference_time_stamp, 'TemperatureRange_decade_average_raw']
        input_model.df_decade.loc[time_stamp, 'Windspeed_decade_average_delta'] = input_model.df_decade.loc[time_stamp, 'Windspeed_decade_average_raw']-input_model.df_decade.loc[reference_time_stamp, 'Windspeed_decade_average_raw']
        input_model.df_decade.loc[time_stamp, 'WindspeedMax_decade_average_delta'] = input_model.df_decade.loc[time_stamp, 'WindspeedMax_decade_average_raw']-input_model.df_decade.loc[reference_time_stamp, 'WindspeedMax_decade_average_raw']
    

    ### Getting items a5 and a6: Decade average annual precipitation and snowfall
    input_model.df_decade['Precipitation_annual_average_raw'] = np.nan
    input_model.df_decade['Precipitation_annual_average_delta'] = np.nan
    input_model.df_decade['Snowfall_annual_average_raw'] = np.nan
    input_model.df_decade['Snowfall_annual_average_delta'] = np.nan
    time_delta = datetime.timedelta(days = 3650)
    for time_stamp in input_model.df_decade.index:
        mask = (input_model.df_annual.index >= time_stamp) & (input_model.df_annual.index < time_stamp+time_delta)
        input_model.df_decade.loc[time_stamp, 'Precipitation_annual_average_raw'] = input_model.df_annual[mask]['pr'].mean()
        input_model.df_decade.loc[time_stamp, 'Snowfall_annual_average_raw'] = input_model.df_annual[mask]['prsn'].mean()
    for time_stamp in input_model.df_decade.index:
        input_model.df_decade.loc[time_stamp, 'Precipitation_annual_average_delta'] = input_model.df_decade.loc[time_stamp, 'Precipitation_annual_average_raw']-input_model.df_decade.loc[reference_time_stamp, 'Precipitation_annual_average_raw']
        input_model.df_decade.loc[time_stamp, 'Snowfall_annual_average_delta'] = input_model.df_decade.loc[time_stamp, 'Snowfall_annual_average_raw']-input_model.df_decade.loc[reference_time_stamp, 'Snowfall_annual_average_raw']
        

    ### Getting items a3, a4, a7, and a8: Decade average annual seasonal (Summer and Winter) temperature and snowfall
    # First creating annual values then will average across the decade
    input_model.df_annual['TemperatureSummer_annual_average_raw'] = np.nan
    input_model.df_annual['TemperatureWinter_annual_average_raw'] = np.nan
    input_model.df_annual['PrecipitationSummer_annual_average_raw'] = np.nan
    input_model.df_annual['PrecipitationWinter_annual_average_raw'] = np.nan
    time_delta = datetime.timedelta(days = 365)
    for time_stamp in input_model.df_annual.index:
        # Defining summer and winter
        summer_start    = datetime.datetime(year = time_stamp.year, month = 6, day = 21, hour = 00, minute = 00, second = 00)
        summer_end      = datetime.datetime(year = time_stamp.year, month = 9, day = 22, hour = 00, minute = 00, second = 00)
        winter_start    = datetime.datetime(year = time_stamp.year-1, month = 12, day = 21, hour = 00, minute = 00, second = 00)
        winter_end      = datetime.datetime(year = time_stamp.year, month = 3, day = 20, hour = 00, minute = 00, second = 00)
        # Summer Mask
        mask = (input_model.df_data.index >= summer_start) & (input_model.df_data.index < summer_end)
        #print(time_stamp, ' ', time_stamp.year, ' ', summer_start, ' ', summer_end)
        input_model.df_annual.loc[time_stamp,'TemperatureSummer_annual_average_raw'] = input_model.df_data[mask]['tas'].mean()
        input_model.df_annual.loc[time_stamp,'PrecipitationSummer_annual_average_raw'] = input_model.df_data[mask]['pr'].sum()
        # Winter Mask
        mask = (input_model.df_data.index >= winter_start) & (input_model.df_data.index < winter_end)
        input_model.df_annual.loc[time_stamp,'TemperatureWinter_annual_average_raw'] = input_model.df_data[mask]['tas'].mean()
        input_model.df_annual.loc[time_stamp,'PrecipitationWinter_annual_average_raw'] = input_model.df_data[mask]['pr'].sum()
    # Annual raw values now exist. Converting to the decadal delta values 
    # raw values
    input_model.df_decade['TemperatureSummer_decade_average_raw'] = np.nan
    input_model.df_decade['TemperatureWinter_decade_average_raw'] = np.nan
    input_model.df_decade['PrecipitationSummer_decade_average_raw'] = np.nan
    input_model.df_decade['PrecipitationWinter_decade_average_raw'] = np.nan
    # delta values
    input_model.df_decade['TemperatureSummer_decade_average_delta'] = np.nan
    input_model.df_decade['TemperatureWinter_decade_average_delta'] = np.nan
    input_model.df_decade['PrecipitationSummer_decade_average_delta'] = np.nan
    input_model.df_decade['PrecipitationWinter_decade_average_delta'] = np.nan
    time_delta = datetime.timedelta(days = 3650)
    for time_stamp in input_model.df_decade.index:
        mask = (input_model.df_annual.index >= time_stamp) & (input_model.df_annual.index < time_stamp+time_delta)
        input_model.df_decade.loc[time_stamp, 'TemperatureSummer_decade_average_raw'] = input_model.df_annual[mask]['TemperatureSummer_annual_average_raw'].mean()
        input_model.df_decade.loc[time_stamp, 'TemperatureWinter_decade_average_raw'] = input_model.df_annual[mask]['TemperatureWinter_annual_average_raw'].mean()
        input_model.df_decade.loc[time_stamp, 'PrecipitationSummer_decade_average_raw'] = input_model.df_annual[mask]['PrecipitationSummer_annual_average_raw'].mean()
        input_model.df_decade.loc[time_stamp, 'PrecipitationWinter_decade_average_raw'] = input_model.df_annual[mask]['PrecipitationWinter_annual_average_raw'].mean()
    reference_time_stamp = input_model.df_decade.index[0]
    for time_stamp in input_model.df_decade.index:
        input_model.df_decade.loc[time_stamp, 'TemperatureSummer_decade_average_delta'] = input_model.df_decade.loc[time_stamp, 'TemperatureSummer_decade_average_raw'] - input_model.df_decade.loc[reference_time_stamp, 'TemperatureSummer_decade_average_raw']
        input_model.df_decade.loc[time_stamp, 'TemperatureWinter_decade_average_delta'] = input_model.df_decade.loc[time_stamp, 'TemperatureWinter_decade_average_raw'] - input_model.df_decade.loc[reference_time_stamp, 'TemperatureWinter_decade_average_raw']
        input_model.df_decade.loc[time_stamp, 'PrecipitationSummer_decade_average_delta'] = input_model.df_decade.loc[time_stamp, 'PrecipitationSummer_decade_average_raw'] - input_model.df_decade.loc[reference_time_stamp, 'PrecipitationSummer_decade_average_raw']
        input_model.df_decade.loc[time_stamp, 'PrecipitationWinter_decade_average_delta'] = input_model.df_decade.loc[time_stamp, 'PrecipitationWinter_decade_average_raw'] - input_model.df_decade.loc[reference_time_stamp, 'PrecipitationWinter_decade_average_raw']

    ##### B-Block #####
    # Getting items b1, b4, b7, b8, b9, b10, b11, b12, and b15
    # Annual Values
    input_model.df_annual['HotDays_annual_average_count'] = np.nan
    input_model.df_annual['ColdDays_annual_average_count'] = np.nan
    input_model.df_annual['DryDays_annual_average_count'] = np.nan
    input_model.df_annual['WetDays_annual_average_count'] = np.nan
    input_model.df_annual['SnowDays_annual_average_count'] = np.nan
    input_model.df_annual['WindyDryDays_annual_average_count'] = np.nan
    input_model.df_annual['WindyWetDays_annual_average_count'] = np.nan
    input_model.df_annual['WindySnowDays_annual_average_count'] = np.nan
    input_model.df_annual['WindyWetNoSnowDays_annual_average_count'] = np.nan
    input_model.df_annual['HeatIndexFrequency_annual_average_count'] = np.nan
    # Decade Values
    input_model.df_decade['HotDays_decade_average_count'] = np.nan
    input_model.df_decade['ColdDays_decade_average_count'] = np.nan
    input_model.df_decade['DryDays_decade_average_count'] = np.nan
    input_model.df_decade['WetDays_decade_average_count'] = np.nan
    input_model.df_annual['SnowDays_decade_average_count'] = np.nan
    input_model.df_decade['WindyDryDays_decade_average_count'] = np.nan
    input_model.df_decade['WindyWetDays_decade_average_count'] = np.nan
    input_model.df_decade['WindySnowDays_decade_average_count'] = np.nan
    input_model.df_decade['WindyWetNoSnowDays_annual_average_count'] = np.nan
    input_model.df_decade['HeatIndexFrequency_decade_average_count'] = np.nan
    # Getting annual values
    time_delta = datetime.timedelta(days = 365)
    for time_stamp in input_model.df_annual.index:
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['tasmax'] > 35)
        input_model.df_annual.loc[time_stamp,'HotDays_annual_average_count'] = len(input_model.df_data[mask]['tasmax'])
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['tasmin'] < 0)
        input_model.df_annual.loc[time_stamp,'ColdDays_annual_average_count'] = len(input_model.df_data[mask]['tasmin'])
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['pr'] < 1e-3)
        input_model.df_annual.loc[time_stamp,'DryDays_annual_average_count'] = len(input_model.df_data[mask]['pr'])
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['pr'] > 1.0)
        input_model.df_annual.loc[time_stamp,'WetDays_annual_average_count'] = len(input_model.df_data[mask]['pr'])
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['prsn'] > 1.0)
        input_model.df_annual.loc[time_stamp,'SnowDays_annual_average_count'] = len(input_model.df_data[mask]['prsn'])
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['pr'] < 1e-3) & (input_model.df_data['sfcWindmax'] > 20.0)
        input_model.df_annual.loc[time_stamp,'WindyDryDays_annual_average_count'] = len(input_model.df_data[mask]['sfcWindmax'])
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['pr'] > 1.0) & (input_model.df_data['sfcWindmax'] > 20.0)
        input_model.df_annual.loc[time_stamp,'WindyWetDays_annual_average_count'] = len(input_model.df_data[mask]['sfcWindmax'])
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['prsn'] > 1.0) & (input_model.df_data['sfcWindmax'] > 20.0)
        input_model.df_annual.loc[time_stamp,'WindySnowDays_annual_average_count'] = len(input_model.df_data[mask]['sfcWindmax'])
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['pr'] > 1.0) & (input_model.df_data['sfcWindmax'] > 20.0) & (input_model.df_data['prsn'] < 1e-3) & (input_model.df_data['tasmin'] < 0.0)
        input_model.df_annual.loc[time_stamp,'WindyWetNoSnowDays_annual_average_count'] = len(input_model.df_data[mask]['sfcWindmax'])
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['heatindex'] > 100.0) 
        input_model.df_annual.loc[time_stamp,'HeatIndexFrequency_annual_average_count'] = len(input_model.df_data[mask]['heatindex'])
    
    # Getting decade Values
    time_delta = datetime.timedelta(days = 3650)
    for time_stamp in input_model.df_decade.index:
        mask = (input_model.df_annual.index >= time_stamp) & (input_model.df_annual.index < time_stamp+time_delta)
        input_model.df_decade.loc[time_stamp, 'HotDays_decade_average_count']   = input_model.df_annual[mask]['HotDays_annual_average_count'].mean()
        input_model.df_decade.loc[time_stamp, 'ColdDays_decade_average_count']  = input_model.df_annual[mask]['ColdDays_annual_average_count'].mean()
        input_model.df_decade.loc[time_stamp, 'DryDays_decade_average_count']   = input_model.df_annual[mask]['DryDays_annual_average_count'].mean()
        input_model.df_decade.loc[time_stamp, 'WetDays_decade_average_count']   = input_model.df_annual[mask]['WetDays_annual_average_count'].mean()
        input_model.df_decade.loc[time_stamp, 'SnowDays_decade_average_count']   = input_model.df_annual[mask]['SnowDays_annual_average_count'].mean()
        input_model.df_decade.loc[time_stamp, 'WindyDryDays_decade_average_count']   = input_model.df_annual[mask]['WindyDryDays_annual_average_count'].mean()
        input_model.df_decade.loc[time_stamp, 'WindyWetDays_decade_average_count']   = input_model.df_annual[mask]['WindyWetDays_annual_average_count'].mean()
        input_model.df_decade.loc[time_stamp, 'WindySnowDays_decade_average_count']   = input_model.df_annual[mask]['WindySnowDays_annual_average_count'].mean()
        input_model.df_decade.loc[time_stamp, 'WindyWetNoSnowDays_decade_average_count']   = input_model.df_annual[mask]['WindyWetNoSnowDays_annual_average_count'].mean()
        input_model.df_decade.loc[time_stamp, 'HeatIndexFrequency_decade_average_count']   = input_model.df_annual[mask]['HeatIndexFrequency_annual_average_count'].mean()

    # Getting b2, b3, b5, b6, b13
    input_model.df_annual['HeatWaveFrequency_annual_average_count'] = np.nan
    input_model.df_annual['HeatWaveDuration_annual_average_raw'] = np.nan
    input_model.df_annual['ColdSnapFrequency_annual_average_count'] = np.nan
    input_model.df_annual['ColdSnapDuration_annual_average_raw'] = np.nan
    input_model.df_annual['DryDaysConsecutive_annual_maximum_raw'] = np.nan
    # Getting 2010-2019 baseline
    baseline_start    = datetime.datetime(year = 2010, month = 1, day = 1, hour = 00, minute = 00, second = 00)
    baseline_end      = datetime.datetime(year = 2020, month = 1, day = 1, hour = 00, minute = 00, second = 00)
    mask = (input_model.df_data.index >= baseline_start) & (input_model.df_data.index < baseline_end)
    baseline_temperatures = input_model.df_data[mask]['tas'].to_list() + input_model.df_data[mask]['tasmax'].to_list() + input_model.df_data[mask]['tasmin'].to_list()
    baseline_temperatures.sort()
    # Update threshold values to take thresholds as inputs
    hot_percentile_005 = baseline_temperatures[int(0.995*len(baseline_temperatures))]
    cold_percentile_005 = baseline_temperatures[int(0.5*len(baseline_temperatures))]
    time_delta = datetime.timedelta(days = 365)
    time_delta_day = datetime.timedelta(days = 1)
    for time_stamp in input_model.df_annual.index:
        # HeatWaves
        heatwave_count = 0
        heatwave_durations = []
        day = time_stamp
        while day < time_stamp+time_delta:
            # Error related to dropping the np.nan values so excepting cases where the day does not exist.
            try: 
                if input_model.df_data.loc[day, 'tasmax'] > hot_percentile_005:
                    heatwave_count += 1
                    heatwave_start = day
                    while input_model.df_data.loc[day, 'tasmax'] > hot_percentile_005:
                        day = day + time_delta_day
                    heatwave_end = day
                    # Getting Durations
                    duration = heatwave_end-heatwave_start
                    heatwave_durations.append(duration.days)
                day = day + time_delta_day
            except:
                day = day + time_delta_day
        input_model.df_annual.loc[time_stamp, 'HeatWaveFrequency_annual_average_count'] = heatwave_count
        if len(heatwave_durations) >= 1 :
            input_model.df_annual.loc[time_stamp, 'HeatWaveDuration_annual_average_raw'] = np.mean(np.array(heatwave_durations))
        else:
            input_model.df_annual.loc[time_stamp, 'HeatWaveDuration_annual_average_raw'] = 0.0
        # Cold Snaps
        coldsnap_count = 0
        coldsnap_durations = []
        day = time_stamp
        while day < time_stamp+time_delta:
            # Error related to dropping the np.nan values so excepting cases where the day does not exist.
            try: 
                if input_model.df_data.loc[day, 'tasmin'] < cold_percentile_005:
                    coldsnap_count += 1
                    coldsnap_start = day
                    while input_model.df_data.loc[day, 'tasmin'] < cold_percentile_005:
                        day = day + time_delta_day
                    coldsnap_end = day
                    # Getting Durations
                    duration = coldsnap_end-coldsnap_start
                    coldsnap_durations.append(duration.days)
                    
                day = day + time_delta_day
            except:
                day = day + time_delta_day
        input_model.df_annual.loc[time_stamp, 'ColdSnapFrequency_annual_average_count'] = coldsnap_count
        if len(heatwave_durations) >= 1 :
            input_model.df_annual.loc[time_stamp, 'ColdSnapDuration_annual_average_raw'] = np.mean(np.array(coldsnap_durations))
        else:
            input_model.df_annual.loc[time_stamp, 'ColdSnapDuration_annual_average_raw'] = 0.0
        # Dry Days
        max_dry_days = 0
        dry_day_count = 0
        day = time_stamp
        while day < time_stamp+time_delta:
            # Error related to dropping the np.nan values so excepting cases where the day does not exist.
            try: 
                if input_model.df_data.loc[day, 'pr'] < 1e-3:
                    dry_day_count = 1
                    while input_model.df_data.loc[day, 'pr'] < 1e-3:
                        day = day + time_delta_day
                        dry_day_count += 1

                    if dry_day_count > max_dry_days:
                        max_dry_days = dry_day_count

                day = day + time_delta_day
            except:
                day = day + time_delta_day

        input_model.df_annual.loc[time_stamp, 'DryDaysConsecutive_annual_maximum_raw'] = max_dry_days
    # Coallating into decade values
    input_model.df_decade['HeatWaveFrequency_decade_average_count'] = np.nan
    input_model.df_decade['HeatWaveDuration_decade_average_raw'] = np.nan
    input_model.df_decade['ColdSnapFrequency_decade_average_count'] = np.nan
    input_model.df_decade['ColdSnapDuration_decade_average_raw'] = np.nan
    input_model.df_decade['DryDaysConsecutive_decade_maximum_raw'] = np.nan
    time_delta = datetime.timedelta(days = 3650)
    for time_stamp in input_model.df_decade.index:
        mask = (input_model.df_annual.index >= time_stamp) & (input_model.df_annual.index < time_stamp+time_delta)
        input_model.df_decade.loc[time_stamp, 'HeatWaveFrequency_decade_average_count']     = input_model.df_annual[mask]['HeatWaveFrequency_annual_average_count'].mean()
        input_model.df_decade.loc[time_stamp, 'HeatWaveDuration_decade_average_raw']        = input_model.df_annual[mask]['HeatWaveDuration_annual_average_raw'].mean()
        input_model.df_decade.loc[time_stamp, 'ColdSnapFrequency_decade_average_count']     = input_model.df_annual[mask]['ColdSnapFrequency_annual_average_count'].mean()
        input_model.df_decade.loc[time_stamp, 'ColdSnapDuration_decade_average_raw']        = input_model.df_annual[mask]['ColdSnapDuration_annual_average_raw'].mean()
        input_model.df_decade.loc[time_stamp, 'DryDaysConsecutive_decade_maximum_raw']        = input_model.df_annual[mask]['DryDaysConsecutive_annual_maximum_raw'].mean()


    ##### C-Block #####
    # Getting items c1, c4 and c7: Maximum and minimum decadal temperatures
    time_delta = datetime.timedelta(days = 3650)
    input_model.df_decade['Temperature_decade_max_raw'] = np.nan
    input_model.df_decade['Temperature_decade_min_raw'] = np.nan
    for time_stamp in input_model.df_decade.index:
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta)
        input_model.df_decade.loc[time_stamp, 'Temperature_decade_max_raw'] = input_model.df_data[mask]['tasmax'].max()
        input_model.df_decade.loc[time_stamp, 'Temperature_decade_min_raw'] = input_model.df_data[mask]['tasmin'].min()
        input_model.df_decade.loc[time_stamp, 'TemperatureRange_decade_max_raw'] = input_model.df_data[mask]['TemperatureRange_daily_average_raw'].max()

    # Getting items c2, c3, c5, c6 and c13
    # These are intensity indicators that suggest the maximum heatwave/coldsnap temperature and durations in a given year and decade.
    # There is a good chunk of this code that is repeated from above but could be condensed. Kept separate to maximize readability
    input_model.df_annual['HeatWaveTemperature_annual_max_raw'] = np.nan
    input_model.df_annual['HeatWaveDuration_annual_max_raw'] = np.nan
    input_model.df_annual['ColdSnapTemperature_annual_min_raw'] = np.nan
    input_model.df_annual['ColdSnapDuration_annual_max_raw'] = np.nan
    input_model.df_annual['DryDaysConsecutiveTempurateMax_annual_average_raw'] = np.nan
    time_delta = datetime.timedelta(days = 365)
    time_delta_day = datetime.timedelta(days = 1)
    # Calculating heat wave maximum temperatures and duration in a given year
    for time_stamp in input_model.df_annual.index:
        heatwave_maximum_temp = 0.0
        heatwave_maximum_duration = 0
        day = time_stamp
        while day < time_stamp+time_delta:
            # Error related to dropping the np.nan values so excepting cases where the day does not exist.
            try:
                if input_model.df_data.loc[day, 'tasmax'] > hot_percentile_005:
                    heatwave_start = day
                    while input_model.df_data.loc[day, 'tasmax'] > hot_percentile_005:
                        # Updating maximum temperature if greater than maximum recorded
                        if input_model.df_data.loc[day, 'tasmax'] > heatwave_maximum_temp:
                            heatwave_maximum_temp = input_model.df_data.loc[day, 'tasmax']
                            #print('new high ', input_model.df_data.loc[day, 'tasmax'])
                        day = day + time_delta_day
                    heatwave_end = day
                    heatwave_duration = heatwave_end - heatwave_start
                    if heatwave_duration.days > heatwave_maximum_duration:
                        heatwave_maximum_duration = heatwave_duration.days
                day = day + time_delta_day
            except:
                day = day + time_delta_day
        input_model.df_annual.loc[time_stamp, 'HeatWaveTemperature_annual_max_raw'] = heatwave_maximum_temp
        input_model.df_annual.loc[time_stamp, 'HeatWaveDuration_annual_max_raw'] = heatwave_maximum_duration
    # Calculating cold snap minimum temperatures and duration in a given year
    for time_stamp in input_model.df_annual.index:
        coldsnap_minimum_temp = 0.0
        coldsnap_maximum_duration = 0
        day = time_stamp
        while day < time_stamp+time_delta:
            # Error related to dropping the np.nan values so excepting cases where the day does not exist.
            try:
                if input_model.df_data.loc[day, 'tasmin'] < cold_percentile_005:
                    coldsnap_start = day
                    while input_model.df_data.loc[day, 'tasmin'] < cold_percentile_005:
                        # Updating maximum temperature if greater than maximum recorded
                        if input_model.df_data.loc[day, 'tasmin'] < coldsnap_minimum_temp:
                            coldsnap_minimum_temp = input_model.df_data.loc[day, 'tasmin']
                        day = day + time_delta_day
                    coldsnap_end = day
                    coldsnap_duration = coldsnap_end - coldsnap_start
                    if coldsnap_duration.days > coldsnap_maximum_duration:
                        coldsnap_maximum_duration = coldsnap_duration.days
                day = day + time_delta_day
            except:
                day = day + time_delta_day
        input_model.df_annual.loc[time_stamp, 'ColdSnapTemperature_annual_min_raw'] = coldsnap_minimum_temp
        input_model.df_annual.loc[time_stamp, 'ColdSnapDuration_annual_max_raw'] = coldsnap_maximum_duration
    # Calculating consecutive dry day maximum temperatures
    for time_stamp in input_model.df_annual.index:
        drydays_maximum_temp = 0.0
        drydays_maximum_duration = 0
        day = time_stamp
        while day < time_stamp+time_delta:
            # Error related to dropping the np.nan values so excepting cases where the day does not exist.
            try:
                if input_model.df_data.loc[day, 'pr'] < 1e-3:
                    dryday_start = day
                    while input_model.df_data.loc[day, 'pr'] < 1e-3:
                        # Updating maximum temperature if greater than maximum recorded
                        if input_model.df_data.loc[day, 'tasmax'] > drydays_maximum_temp:
                            drydays_maximum_temp = input_model.df_data.loc[day, 'tasmax']
                        day = day + time_delta_day
                    dryday_end = day
                    drydays_duration = dryday_end - dryday_start
                    if drydays_duration.days > drydays_maximum_duration:
                        drydays_maximum_duration = drydays_duration.days
                day = day + time_delta_day
            except:
                day = day + time_delta_day
        input_model.df_annual.loc[time_stamp, 'DryDaysConsecutiveTempurateMax_annual_average_raw'] = drydays_maximum_temp
    # Converting to decade values
    time_delta = datetime.timedelta(days = 3650)
    input_model.df_decade['HeatWaveTemperature_decade_max_raw'] = np.nan
    input_model.df_decade['HeatWaveDuration_decade_max_raw'] = np.nan
    input_model.df_decade['ColdSnapTemperature_decade_max_raw'] = np.nan
    input_model.df_decade['ColdSnapDuration_decade_max_raw'] = np.nan
    input_model.df_decade['DryDaysConsecutiveTempurateMax_decade_average_raw'] = np.nan
    for time_stamp in input_model.df_decade.index:
        mask = (input_model.df_annual.index >= time_stamp) & (input_model.df_annual.index < time_stamp+time_delta)
        input_model.df_decade.loc[time_stamp, 'HeatWaveTemperature_decade_max_raw']   = input_model.df_annual[mask]['HeatWaveTemperature_annual_max_raw'].max()
        input_model.df_decade.loc[time_stamp, 'HeatWaveDuration_decade_max_raw']   = input_model.df_annual[mask]['HeatWaveDuration_annual_max_raw'].max()
        input_model.df_decade.loc[time_stamp, 'ColdSnapTemperature_decade_min_raw']   = input_model.df_annual[mask]['ColdSnapTemperature_annual_min_raw'].min()
        input_model.df_decade.loc[time_stamp, 'ColdSnapDuration_decade_max_raw']   = input_model.df_annual[mask]['ColdSnapDuration_annual_max_raw'].max()
        input_model.df_decade.loc[time_stamp, 'DryDaysConsecutiveTempurateMax_decade_average_raw']   = input_model.df_annual[mask]['DryDaysConsecutiveTempurateMax_annual_average_raw'].mean()


    # Getting items c8, c9, c10, c11, c12, and c14: The averages during extreme events such as rainfall
    # Annual Values
    #input_model.df_annual['WetDaysPrecipitation_annual_average_raw'] = np.nan
    #input_model.df_annual['SnowDaysPrecipitation_annual_average_raw'] = np.nan
    #input_model.df_annual['WindyDaysWindspeed_annual_average_raw'] = np.nan
    #input_model.df_annual['WindyWetDaysWindspeed_annual_average_raw'] = np.nan
    #input_model.df_annual['WindySnowDaysWindspeed_annual_average_raw'] = np.nan
    #input_model.df_annual['WindyWetNoSnowDaysWindspeed_annual_average_raw'] = np.nan
    # Decade Values
    input_model.df_decade['WetDaysPrecipitation_decade_average_raw'] = np.nan
    input_model.df_decade['SnowDaysPrecipitation_decade_average_raw'] = np.nan
    input_model.df_decade['WindyDryDaysWindspeed_decade_average_raw'] = np.nan
    input_model.df_decade['WindyWetDaysWindspeed_decade_average_raw'] = np.nan
    input_model.df_decade['WindySnowDaysWindspeed_decade_average_raw'] = np.nan
    input_model.df_decade['WindyWetNoSnowDaysWindspeed_decade_average_raw'] = np.nan
    # Getting annual values
    time_delta = datetime.timedelta(days = 3650)
    for time_stamp in input_model.df_decade.index:
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['pr'] > 1.0)
        input_model.df_decade.loc[time_stamp, 'WetDaysPrecipitation_decade_average_raw'] = input_model.df_data[mask]['pr'].mean()
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['prsn'] > 1.0)
        input_model.df_decade.loc[time_stamp, 'SnowDaysPrecipitation_decade_average_raw'] = input_model.df_data[mask]['prsn'].mean()
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['pr'] < 1e-3) & (input_model.df_data['sfcWindmax'] > 20.0)
        input_model.df_decade.loc[time_stamp, 'WindyDryDaysWindspeed_decade_average_raw'] = input_model.df_data[mask]['sfcWind'].mean()
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['pr'] > 1.0) & (input_model.df_data['sfcWindmax'] > 20.0)
        input_model.df_decade.loc[time_stamp, 'WindyWetDaysWindspeed_decade_average_raw'] = input_model.df_data[mask]['sfcWind'].mean()*input_model.df_data[mask]['pr'].mean()
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['prsn'] > 1.0) & (input_model.df_data['sfcWindmax'] > 20.0)
        input_model.df_decade.loc[time_stamp, 'WindySnowDaysWindspeed_decade_average_raw'] = input_model.df_data[mask]['sfcWind'].mean()*input_model.df_data[mask]['prsn'].mean()
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['pr'] > 1.0) & (input_model.df_data['sfcWindmax'] > 20.0) & (input_model.df_data['prsn'] < 1e-3) & (input_model.df_data['tasmin'] < 0.0)
        input_model.df_decade.loc[time_stamp, 'WindyWetNoSnowDaysWindspeed_decade_average_raw'] = input_model.df_data[mask]['sfcWind'].mean()*input_model.df_data[mask]['pr'].mean()


    print('Analysis Complete')