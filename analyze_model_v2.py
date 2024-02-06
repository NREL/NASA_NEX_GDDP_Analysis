# Importing all dependencies
from dependencies import *

def analyze_model(input_model):
    # Year to Compar
    baseline_year_start = 2010
    baseline_year_end = 2020
    comparison_year = 2050
    # Dropping NA values
    input_model.df_data.dropna(subset = ['tas', 'tasmin', 'tasmax'], inplace = True)

    # Getting min and max year
    min_year = min(input_model.df_data.index).year
    max_year = max(input_model.df_data.index).year
    years = [*range(min_year,max_year+1)]

    # Getting variables to average for the annual average
    annual_average_variables = ['hurs', 
        'sfcWind',
        'tas', 'tasmax', 'tasmin',
        ]
    annual_sum_variables = ['pr']
    monthly_extreme_variables = ['tasmas', 'tasmin']
    annual_extreme_variables = ['tasmax', 'tasmin']


    ##### Beginning calculations #####
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
        RH = input_model.df_data.loc[time_stamp, 'hurs']
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

    # Monthly Compilation
    time_delta = datetime.timedelta(days = 30)
    for variable in annual_average_variables+annual_sum_variables: #+annual_extreme_variables:
        for time_stamp in input_model.df_monthly.index:
            mask = (input_model.df_data.index > time_stamp) & (input_model.df_data.index < time_stamp+time_delta)
            if variable in annual_average_variables:
                input_model.df_monthly.loc[time_stamp, variable] = input_model.df_data[mask][variable].mean()
            if variable in annual_sum_variables:
                input_model.df_monthly.loc[time_stamp, variable] = input_model.df_data[mask][variable].sum()

    # Annual Compilation
    time_delta = datetime.timedelta(days = 365)
    for variable in annual_average_variables+annual_sum_variables: #+annual_extreme_variables:
        input_model.df_annual[variable] = np.nan
        for time_stamp in input_model.df_annual.index:
            mask = (input_model.df_data.index > time_stamp) & (input_model.df_data.index < time_stamp+time_delta)
            if variable in annual_average_variables:
                input_model.df_annual.loc[time_stamp, variable] = input_model.df_data[mask][variable].mean()
            if variable in annual_sum_variables:
                input_model.df_annual.loc[time_stamp, variable] = input_model.df_data[mask][variable].sum()

    # Decade Compilation
    time_delta = datetime.timedelta(days = 3650)
    for variable in annual_average_variables+annual_sum_variables: #+annual_extreme_variables:
        try:
            input_model.df_decade[variable] = np.nan
            for time_stamp in input_model.df_decade.index:
                mask = (input_model.df_annual.index > time_stamp) & (input_model.df_annual.index < time_stamp+time_delta)
                input_model.df_decade.loc[time_stamp, variable] = input_model.df_annual[mask][variable].mean()
        except: 
            continue

    ##### Compiling Indicators #####
    # TXx, TNx, TXn, TNn
    input_model.df_monthly['TXx'] = np.nan
    time_delta = datetime.timedelta(days = 30)
    for time_stamp in input_model.df_monthly.index:
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta)
        input_model.df_monthly.loc[time_stamp, 'TXx'] = input_model.df_data[mask]['tasmax'].max()
        input_model.df_monthly.loc[time_stamp, 'TNx'] = input_model.df_data[mask]['tasmin'].max()
        input_model.df_monthly.loc[time_stamp, 'TXn'] = input_model.df_data[mask]['tasmax'].min()
        input_model.df_monthly.loc[time_stamp, 'TNn'] = input_model.df_data[mask]['tasmin'].min()

    # DTR - Daily Temperature Range
    input_model.df_data['daily_temperature_range'] = input_model.df_data['tasmax'] - input_model.df_data['tasmin']
    time_delta = datetime.timedelta(days = 30)
    for time_stamp in input_model.df_monthly.index:
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta)
        input_model.df_monthly.loc[time_stamp, 'DTR'] = input_model.df_data[mask]['daily_temperature_range'].mean()

    # RX1 and RX5day
    input_model.df_annual['RX1day']  = np.nan
    input_model.df_annual['RX5day']  = np.nan
    input_model.df_decade['RX1day']  = np.nan
    input_model.df_decade['RX5day']  = np.nan
    time_delta = datetime.timedelta(days = 365)
    time_delta_day = datetime.timedelta(days = 1)
    time_delta_5_days = datetime.timedelta(days = 5)
    for time_stamp in input_model.df_annual.index:
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta)
        input_model.df_annual.loc[time_stamp, 'RX1day'] = input_model.df_data[mask]['pr'].max()
        # Getting 5 day maximum
        day = time_stamp
        max_pr = 0.0
        while day < time_stamp+time_delta:
            subset_mask = (input_model.df_data.index >= day) & (input_model.df_data.index < day+time_delta_5_days)
            tmp_pr = input_model.df_data[mask]['pr'].sum()
            
            if tmp_pr > max_pr:
                max_pr = tmp_pr
            day = day + time_delta_day
        input_model.df_annual.loc[time_stamp, 'RX5day']=max_pr
    # Getting Decadal Values
    time_delta = datetime.timedelta(days = 3650)
    for time_stamp in input_model.df_decade.index:
        mask = (input_model.df_annual.index >= time_stamp) & (input_model.df_annual.index < time_stamp+time_delta)
        input_model.df_decade.loc[time_stamp, 'RX1day'] = input_model.df_annual[mask]['RX1day'].max()
        input_model.df_decade.loc[time_stamp, 'RX5day'] = input_model.df_annual[mask]['RX5day'].max()

    # TN10p, TX10p, TN90p, TX90p - Percentile days and WSDI and CSDI
    input_model.df_annual['TN10p']  = np.nan
    input_model.df_annual['TX10p']  = np.nan
    input_model.df_annual['TN90p']  = np.nan
    input_model.df_annual['TX90p']  = np.nan
    baseline_start    = datetime.datetime(year = baseline_year_start, month = 1, day = 1, hour = 00, minute = 00, second = 00)
    baseline_end      = datetime.datetime(year = baseline_year_end, month = 1, day = 1, hour = 00, minute = 00, second = 00)
    mask = (input_model.df_data.index >= baseline_start) & (input_model.df_data.index < baseline_end)
    baseline_hot_temperatures =  input_model.df_data[mask]['tasmax'].to_list() 
    baseline_hot_temperatures.sort()
    baseline_cold_temperatures =  input_model.df_data[mask]['tasmin'].to_list() 
    baseline_cold_temperatures.sort()
        # Getting hot and cold percentile values
    hot_percentile_10 = baseline_hot_temperatures[int(0.10*len(baseline_hot_temperatures))]
    hot_percentile_90 = baseline_hot_temperatures[int(0.90*len(baseline_hot_temperatures))]
    cold_percentile_10 = baseline_cold_temperatures[int(0.10*len(baseline_cold_temperatures))]
    cold_percentile_90 = baseline_cold_temperatures[int(0.90*len(baseline_cold_temperatures))]
    time_delta = datetime.timedelta(days = 365)
        # Starting calculation loop
    for time_stamp in input_model.df_annual.index:
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['tasmin']<cold_percentile_10)
        input_model.df_annual.loc[time_stamp, 'TN10p'] = len(input_model.df_data[mask]['tasmin'])
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['tasmax']<hot_percentile_10)
        input_model.df_annual.loc[time_stamp, 'TX10p'] = len(input_model.df_data[mask]['tasmax'])
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['tasmin']>cold_percentile_90)
        input_model.df_annual.loc[time_stamp, 'TN90p'] = len(input_model.df_data[mask]['tasmin'])
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['tasmax']>hot_percentile_90)
        input_model.df_annual.loc[time_stamp, 'TX90p'] = len(input_model.df_data[mask]['tasmax'])     
        # WSDI and CSDI
    time_delta_day = datetime.timedelta(days = 1)
    input_model.df_annual['WSDI']  = np.nan
    input_model.df_annual['CSDI']  = np.nan
    input_model.df_annual['WSDI_duration']  = np.nan
    input_model.df_annual['CSDI_duration']  = np.nan
    for time_stamp in input_model.df_annual.index:
        warm_spell_count = 0
        warm_spell_durations = []
        day = time_stamp
        while day < time_stamp+time_delta:
            # Error related to dropping the np.nan values so excepting cases where the day does not exist.
            try: 
                if input_model.df_data.loc[day, 'tasmax'] > hot_percentile_90:
                    warm_spell_start = day
                    while input_model.df_data.loc[day, 'tasmax'] > hot_percentile_90:
                        day = day + time_delta_day
                    warm_spell_end = day
                    duration = warm_spell_end-warm_spell_start
                    if duration.days > 6:
                        warm_spell_count += 1
                        warm_spell_durations.append(duration.days)
                day = day + time_delta_day
            except:
                day = day + time_delta_day   
            input_model.df_annual.loc[time_stamp, 'WSDI'] = warm_spell_count 
            if len(warm_spell_durations) >= 1:
                input_model.df_annual.loc[time_stamp, 'WSDI_duration'] = np.mean(warm_spell_durations)
            else:
                input_model.df_annual.loc[time_stamp, 'WSDI_duration'] = np.nan
        cold_spell_count = 0
        cold_spell_durations = []
        day = time_stamp
        while day < time_stamp+time_delta:
            # Error related to dropping the np.nan values so excepting cases where the day does not exist.
            try: 
                if input_model.df_data.loc[day, 'tasmin'] < cold_percentile_10:
                    cold_spell_start = day
                    while input_model.df_data.loc[day, 'tasmin'] < cold_percentile_10:
                        day = day + time_delta_day
                    cold_spell_end = day
                    duration = cold_spell_end - cold_spell_start
                    if duration.days > 6:
                        cold_spell_count += 1
                        cold_spell_durations.append(duration.days)
                day = day + time_delta_day
            except:
                day = day + time_delta_day
            input_model.df_annual.loc[time_stamp, 'CSDI'] = cold_spell_count
            if len(cold_spell_durations) >= 1:
                input_model.df_annual.loc[time_stamp, 'CSDI_duration'] = np.mean(cold_spell_durations)
            else:
                input_model.df_annual.loc[time_stamp, 'CSDI_duration'] = np.nan
    # Getting Decadal Values
    input_model.df_decade['TN10p']  = np.nan
    input_model.df_decade['TX10p']  = np.nan
    input_model.df_decade['TN90p']  = np.nan
    input_model.df_decade['TX90p']  = np.nan
    input_model.df_decade['WSDI']  = np.nan
    input_model.df_decade['CSDI']  = np.nan
    input_model.df_decade['WSDI_duration']  = np.nan
    input_model.df_decade['CSDI_duration']  = np.nan
    time_delta = datetime.timedelta(days = 3650)
    for time_stamp in input_model.df_decade.index:
        mask = (input_model.df_annual.index >= time_stamp) & (input_model.df_annual.index < time_stamp+time_delta)
        input_model.df_decade.loc[time_stamp, 'TN10p'] = input_model.df_annual[mask]['TN10p'].mean()
        input_model.df_decade.loc[time_stamp, 'TX10p'] = input_model.df_annual[mask]['TX10p'].mean()
        input_model.df_decade.loc[time_stamp, 'TN90p'] = input_model.df_annual[mask]['TN90p'].mean()
        input_model.df_decade.loc[time_stamp, 'TX90p'] = input_model.df_annual[mask]['TX90p'].mean()
        input_model.df_decade.loc[time_stamp, 'WSDI'] = input_model.df_annual[mask]['WSDI'].mean()
        input_model.df_decade.loc[time_stamp, 'CSDI'] = input_model.df_annual[mask]['CSDI'].mean()
        input_model.df_decade.loc[time_stamp, 'WSDI_duration'] = input_model.df_annual[mask]['WSDI_duration'].mean()
        input_model.df_decade.loc[time_stamp, 'CSDI_duration'] = input_model.df_annual[mask]['CSDI_duration'].mean()


    # FD0, SU25, TR20
    input_model.df_annual['FD0']  = np.nan
    input_model.df_annual['SU25']  = np.nan
    input_model.df_annual['TR20']  = np.nan
    time_delta = datetime.timedelta(days = 365)
    for time_stamp in input_model.df_annual.index:
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['tasmin']<0)
        input_model.df_annual.loc[time_stamp, 'FD0'] = len(input_model.df_data[mask]['tasmin'])
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['tasmax']>25)
        input_model.df_annual.loc[time_stamp, 'SU25'] = len(input_model.df_data[mask]['tasmax'])
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['tasmin']>20)
        input_model.df_annual.loc[time_stamp, 'TR20'] = len(input_model.df_data[mask]['tasmin'])
    # Getting Decadal Values
    input_model.df_decade['FD0']  = np.nan
    input_model.df_decade['SU25']  = np.nan
    input_model.df_decade['TR20']  = np.nan
    time_delta = datetime.timedelta(days = 3650)
    for time_stamp in input_model.df_decade.index:
        mask = (input_model.df_annual.index >= time_stamp) & (input_model.df_annual.index < time_stamp+time_delta)
        input_model.df_decade.loc[time_stamp, 'FD0'] = input_model.df_annual[mask]['FD0'].mean()
        input_model.df_decade.loc[time_stamp, 'SU25'] = input_model.df_annual[mask]['SU25'].mean()
        input_model.df_decade.loc[time_stamp, 'TR20'] = input_model.df_annual[mask]['TR20'].mean()
    
    # RX1, RX5 
    input_model.df_monthly['RX1day'] = np.nan
    input_model.df_monthly['RX5day'] = np.nan
    time_delta = datetime.timedelta(days = 30)
    for time_stamp in input_model.df_monthly.index:
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta)
        input_model.df_monthly.loc[time_stamp, 'RX1day'] = input_model.df_data[mask]['pr'].max()
        tmp_max = 0
        tmp_time_delta = datetime.timedelta(days = 5)
        for tmp_time_stamp in input_model.df_data[mask].index:
            subset_mask = (input_model.df_data.index >= tmp_time_stamp) & (input_model.df_data.index < tmp_time_stamp+tmp_time_delta)
            tmp_sum = input_model.df_data[subset_mask]['pr'].sum()
            if tmp_sum > tmp_max:
                tmp_max = tmp_sum
        input_model.df_monthly.loc[time_stamp, 'RX5day'] = tmp_max

    # PRCPTOT, R10, R20, R1, RLT1, SDII
    input_model.df_annual['PRCPTOT']  = np.nan
    input_model.df_annual['R10']  = np.nan
    input_model.df_annual['R20']  = np.nan
    input_model.df_annual['R1']  = np.nan
    input_model.df_annual['RLT1']  = np.nan
    input_model.df_annual['SDII']  = np.nan
    time_delta = datetime.timedelta(days = 365)
    for time_stamp in input_model.df_annual.index:
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta)
        input_model.df_annual.loc[time_stamp, 'PRCPTOT'] = input_model.df_data[mask]['pr'].sum()
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['pr']>10)
        input_model.df_annual.loc[time_stamp, 'R10'] = len(input_model.df_data[mask]['pr'])
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['pr']>20)
        input_model.df_annual.loc[time_stamp, 'R20'] = len(input_model.df_data[mask]['pr'])
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['pr']>1)
        input_model.df_annual.loc[time_stamp, 'R1'] = len(input_model.df_data[mask]['pr'])
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['pr']<1)
        input_model.df_annual.loc[time_stamp, 'RLT1'] = len(input_model.df_data[mask]['pr'])
    input_model.df_annual['SDII'] = input_model.df_annual['PRCPTOT']/input_model.df_annual['R1']
    # Getting Decadal Values
    input_model.df_decade['PRCPTOT']  = np.nan
    input_model.df_decade['R10']  = np.nan
    input_model.df_decade['R20']  = np.nan
    input_model.df_decade['R1']  = np.nan
    input_model.df_decade['RLT1']  = np.nan
    input_model.df_decade['SDII']  = np.nan
    time_delta = datetime.timedelta(days = 3650)
    for time_stamp in input_model.df_decade.index:
        mask = (input_model.df_annual.index >= time_stamp) & (input_model.df_annual.index < time_stamp+time_delta)
        input_model.df_decade.loc[time_stamp, 'PRCPTOT'] = input_model.df_annual[mask]['PRCPTOT'].mean()
        input_model.df_decade.loc[time_stamp, 'R10'] = input_model.df_annual[mask]['R10'].mean()
        input_model.df_decade.loc[time_stamp, 'R20'] = input_model.df_annual[mask]['R20'].mean()
        input_model.df_decade.loc[time_stamp, 'R1'] = input_model.df_annual[mask]['R1'].mean()
        input_model.df_decade.loc[time_stamp, 'RLT1'] = input_model.df_annual[mask]['RLT1'].mean()
        input_model.df_decade.loc[time_stamp, 'SDII'] = input_model.df_annual[mask]['SDII'].mean()

    # R95p, R99p
    baseline_start    = datetime.datetime(year = baseline_year_start, month = 1, day = 1, hour = 00, minute = 00, second = 00)
    baseline_end      = datetime.datetime(year = baseline_year_end, month = 1, day = 1, hour = 00, minute = 00, second = 00)
    mask = (input_model.df_data.index >= baseline_start) & (input_model.df_data.index < baseline_end)
    baseline_preicipiation =  input_model.df_data[mask]['pr'].to_list() 
    baseline_preicipiation.sort()
    precipitation_percentile_95 = baseline_preicipiation[int(0.95*len(baseline_preicipiation))]
    precipitation_percentile_99 = baseline_preicipiation[int(0.99*len(baseline_preicipiation))]
    time_delta = datetime.timedelta(days = 365)
    input_model.df_annual['R95p']  = np.nan
    input_model.df_annual['R99p']  = np.nan
    for time_stamp in input_model.df_annual.index:
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['pr']>precipitation_percentile_95)
        input_model.df_annual.loc[time_stamp, 'R95p'] = len(input_model.df_data[mask]['pr'])
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['pr']>precipitation_percentile_99)
        input_model.df_annual.loc[time_stamp, 'R99p'] = len(input_model.df_data[mask]['pr'])
    # Getting Decadal Values
    input_model.df_decade['R95p']  = np.nan
    input_model.df_decade['R99p']  = np.nan
    time_delta = datetime.timedelta(days = 3650)
    for time_stamp in input_model.df_decade.index:
        mask = (input_model.df_annual.index >= time_stamp) & (input_model.df_annual.index < time_stamp+time_delta)
        input_model.df_decade.loc[time_stamp, 'R95p'] = input_model.df_annual[mask]['R95p'].mean()
        input_model.df_decade.loc[time_stamp, 'R99p'] = input_model.df_annual[mask]['R95p'].mean()

    # CDD, CWD
    time_delta = datetime.timedelta(days = 365)
    time_delta_day = datetime.timedelta(days = 1)
    input_model.df_annual['CDD']  = np.nan
    input_model.df_annual['CWD']  = np.nan
    for time_stamp in input_model.df_annual.index:
        cdd = 0
        day = time_stamp
        while day < time_stamp+time_delta:
            try:
                current_dry_day_count = 0
                while input_model.df_data.loc[day,'pr'] < 1:
                    current_dry_day_count += 1
                    day = day + time_delta_day

                if current_dry_day_count > cdd:
                    cdd = current_dry_day_count

                day = day + time_delta_day
            except: 
                day = day + time_delta_day
        input_model.df_annual.loc[time_stamp, 'CDD'] = cdd
        cwd = 0
        day = time_stamp
        while day < time_stamp+time_delta:
            try:
                current_wet_day_count = 0
                while input_model.df_data.loc[day,'pr'] > 1:
                    current_wet_day_count += 1
                    day = day + time_delta_day

                if current_wet_day_count > cwd:
                    cwd = current_wet_day_count

                day = day + time_delta_day
            except: 
                day = day + time_delta_day
        input_model.df_annual.loc[time_stamp, 'CWD'] = cwd
    # Getting Decadal Values
    input_model.df_decade['CDD']  = np.nan
    input_model.df_decade['CWD']  = np.nan
    time_delta = datetime.timedelta(days = 3650)
    for time_stamp in input_model.df_decade.index:
        mask = (input_model.df_annual.index >= time_stamp) & (input_model.df_annual.index < time_stamp+time_delta)
        input_model.df_decade.loc[time_stamp, 'CDD'] = input_model.df_annual[mask]['CDD'].mean()
        input_model.df_decade.loc[time_stamp, 'CWD'] = input_model.df_annual[mask]['CWD'].mean()

    ##### Compiling Custom Indicators #####
    ##### A-Block #####
    # a1
    reference_time_stamp = input_model.df_decade.index[0]
    input_model.df_decade['Temperature_decade_average_delta'] = np.nan
    for time_stamp in input_model.df_decade.index:
        input_model.df_decade.loc[time_stamp, 'Temperature_decade_average_delta'] = input_model.df_decade.loc[time_stamp, 'tas'] - input_model.df_decade.loc[reference_time_stamp, 'tas']

    # a2, a9
    input_model.df_decade['TemperatureRange_decade_average_raw'] = np.nan
    input_model.df_decade['TemperatureRange_decade_average_delta'] = np.nan
    input_model.df_decade['Windspeed_decade_average_raw'] = np.nan
    input_model.df_decade['Windspeed_decade_average_delta'] = np.nan
    time_delta = datetime.timedelta(days = 3650)
    for time_stamp in input_model.df_decade.index:
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta)
        input_model.df_decade.loc[time_stamp, 'TemperatureRange_decade_average_raw'] = input_model.df_data[mask]['daily_temperature_range'].mean()
        input_model.df_decade.loc[time_stamp, 'TemperatureRange_decade_average_delta'] = input_model.df_decade.loc[time_stamp, 'TemperatureRange_decade_average_raw']-input_model.df_decade.loc[reference_time_stamp, 'TemperatureRange_decade_average_raw']
        input_model.df_decade.loc[time_stamp, 'Windspeed_decade_average_raw'] = input_model.df_data[mask]['sfcWind'].mean()
        input_model.df_decade.loc[time_stamp, 'Windspeed_decade_average_delta'] = input_model.df_decade.loc[time_stamp, 'Windspeed_decade_average_raw']-input_model.df_decade.loc[reference_time_stamp, 'Windspeed_decade_average_raw']
        
    # a5
    input_model.df_decade['Precipitation_annual_average_raw'] = np.nan
    input_model.df_decade['Precipitation_annual_average_delta'] = np.nan
    time_delta = datetime.timedelta(days = 3650)
    for time_stamp in input_model.df_decade.index:
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta)
        input_model.df_decade.loc[time_stamp, 'Precipitation_annual_average_raw'] = input_model.df_data[mask]['pr'].mean()
        #input_model.df_decade.loc[time_stamp, 'Precipitation_annual_average_delta'] = input_model.df_decade.loc[time_stamp, 'Precipitation_annual_average_raw']-input_model.df_decade.loc[reference_time_stamp, 'Precipitation_annual_average_raw']
    
    # a3, a4, a7, a8
    input_model.df_annual['TemperatureSummer_annual_average_raw'] = np.nan
    input_model.df_annual['TemperatureWinter_annual_average_raw'] = np.nan
    input_model.df_annual['PrecipitationSummer_annual_average_raw'] = np.nan
    input_model.df_annual['PrecipitationWinter_annual_average_raw'] = np.nan
    time_delta = datetime.timedelta(days = 365)
    for time_stamp in input_model.df_annual.index:
        # Defining Summer and Winter Start
        summer_start    = datetime.datetime(year = time_stamp.year, month = 6, day = 1, hour = 00, minute = 00, second = 00)
        summer_end      = datetime.datetime(year = time_stamp.year, month = 9, day = 1, hour = 00, minute = 00, second = 00)
        winter_start    = datetime.datetime(year = time_stamp.year-1, month = 12, day = 1, hour = 00, minute = 00, second = 00)
        winter_end      = datetime.datetime(year = time_stamp.year, month = 3, day = 1, hour = 00, minute = 00, second = 00)
        mask = (input_model.df_data.index >= summer_start) & (input_model.df_data.index < summer_end)
        input_model.df_annual.loc[time_stamp,'TemperatureSummer_annual_average_raw'] = input_model.df_data[mask]['tas'].mean()
        input_model.df_annual.loc[time_stamp,'PrecipitationSummer_annual_average_raw'] = input_model.df_data[mask]['pr'].sum()
        mask = (input_model.df_data.index >= winter_start) & (input_model.df_data.index < winter_end)
        input_model.df_annual.loc[time_stamp,'TemperatureWinter_annual_average_raw'] = input_model.df_data[mask]['tas'].mean()
        input_model.df_annual.loc[time_stamp,'PrecipitationWinter_annual_average_raw'] = input_model.df_data[mask]['pr'].sum()
    # Getting Decade Valuse
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
    # b1, v4, b15
        # Annual
    input_model.df_annual['HotDays_annual_average_count'] = np.nan
    input_model.df_annual['ColdDays_annual_average_count'] = np.nan
    input_model.df_annual['HeatIndexFrequency_annual_average_count'] = np.nan
        # Decade
    input_model.df_decade['HotDays_decade_average_count'] = np.nan
    input_model.df_decade['ColdDays_decade_average_count'] = np.nan
    input_model.df_decade['HeatIndexFrequency_decade_average_count'] = np.nan
    time_delta = datetime.timedelta(days = 365)
    for time_stamp in input_model.df_annual.index:
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['tasmax'] > 35)
        input_model.df_annual.loc[time_stamp,'HotDays_annual_average_count'] = len(input_model.df_data[mask]['tasmax'])
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['tasmin'] < 0)
        input_model.df_annual.loc[time_stamp,'ColdDays_annual_average_count'] = len(input_model.df_data[mask]['tasmin'])
    time_delta = datetime.timedelta(days = 3650)
    for time_stamp in input_model.df_decade.index:
        mask = (input_model.df_annual.index >= time_stamp) & (input_model.df_annual.index < time_stamp+time_delta)
        input_model.df_decade.loc[time_stamp, 'HotDays_decade_average_count']   = input_model.df_annual[mask]['HotDays_annual_average_count'].mean()
        input_model.df_decade.loc[time_stamp, 'ColdDays_decade_average_count']  = input_model.df_annual[mask]['ColdDays_annual_average_count'].mean()
        input_model.df_decade.loc[time_stamp, 'HeatIndexFrequency_decade_average_count']   = input_model.df_annual[mask]['HeatIndexFrequency_annual_average_count'].mean()

    # b2, b3, b5, b6
    input_model.df_annual['HeatWaveFrequency_annual_average_count'] = np.nan
    input_model.df_annual['HeatWaveDuration_annual_average_raw'] = np.nan
    input_model.df_annual['ColdSnapFrequency_annual_average_count'] = np.nan
    input_model.df_annual['ColdSnapDuration_annual_average_raw'] = np.nan
    baseline_start    = datetime.datetime(year = baseline_year_start, month = 1, day = 1, hour = 00, minute = 00, second = 00)
    baseline_end      = datetime.datetime(year = baseline_year_end, month = 1, day = 1, hour = 00, minute = 00, second = 00)
    mask = (input_model.df_data.index >= baseline_start) & (input_model.df_data.index < baseline_end)
    baseline_temperatures = input_model.df_data[mask]['tas'].to_list() + input_model.df_data[mask]['tasmax'].to_list() + input_model.df_data[mask]['tasmin'].to_list()
    baseline_temperatures.sort()
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

    ##### C-Block #####
    # c1, c4, c7
    input_model.df_decade['Temperature_decade_max_raw'] = np.nan
    input_model.df_decade['Temperature_decade_min_raw'] = np.nan
    input_model.df_decade['TemperatureRange_decade_max_raw'] = np.nan
    time_delta = datetime.timedelta(days = 3650)
    for time_stamp in input_model.df_decade.index:
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta)
        input_model.df_decade.loc[time_stamp, 'Temperature_decade_max_raw'] = input_model.df_data[mask]['tasmax'].max()
        input_model.df_decade.loc[time_stamp, 'Temperature_decade_min_raw'] = input_model.df_data[mask]['tasmin'].min()
        input_model.df_decade.loc[time_stamp, 'TemperatureRange_decade_max_raw'] = input_model.df_data[mask]['daily_temperature_range'].max()

    # c2, c3, c5, c6
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

    ##### D-Block #####
    ## D1 - Heating Degree Days
    ## D2 - Cooling Degree Days
    input_model.df_data['degree_days'] = np.nan
    T_min = input_model.df_data['tasmin']*9/5+32
    T_max = input_model.df_data['tasmax']*9/5+32
    input_model.df_data['degree_days'] = (T_max+T_min)/2 - 65

    input_model.df_annual['Heating Degree Days'] = np.nan
    input_model.df_annual['Cooling Degree Days'] = np.nan
    time_delta = datetime.timedelta(days = 365)
    for time_stamp in input_model.df_annual.index:
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['degree_days'] < 0)
        input_model.df_annual.loc[time_stamp, 'Heating Degree Days'] = -1*sum(input_model.df_data[mask]['degree_days'])
        mask = (input_model.df_data.index >= time_stamp) & (input_model.df_data.index < time_stamp+time_delta) & (input_model.df_data['degree_days'] > 0)
        input_model.df_annual.loc[time_stamp,'Cooling Degree Days'] = sum(input_model.df_data[mask]['degree_days'])

    time_delta = datetime.timedelta(days = 3650)
    for time_stamp in input_model.df_decade.index:
        mask = (input_model.df_annual.index >= time_stamp) & (input_model.df_annual.index < time_stamp+time_delta)
        input_model.df_decade.loc[time_stamp, 'Heating Degree Days'] = input_model.df_annual[mask]['Heating Degree Days'].mean()
        input_model.df_decade.loc[time_stamp, 'Cooling Degree Days'] = input_model.df_annual[mask]['Cooling Degree Days'].mean()

    print('Analysis Complete')