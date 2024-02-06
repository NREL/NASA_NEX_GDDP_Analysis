# Importing all dependencies
from dependencies import *

class model_obj:
    def __init__(self, path, model_name, experiment_name):
        self.path = path
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.bool_data_compiled = False
        self.max_year = '2099'
        #self.data_lat = 0.0
        #self.data_lon = 0.0

        self.variables_to_compile = [
            'hurs', 'huss', # humidity
            'rlds', 'rsds', # long and short wave radiation
            'pr', # precipitation
            'sfcWind', # wind
            'tas', 'tasmax', 'tasmin'] #temperature

        # Setting up dataframes to handle the variables
        date_time_span = pd.date_range(start = '2015-01-01 12:00:00', end = self.max_year+'-12-31 12:00:00', freq = 'd')
        self.df_data = pd.DataFrame(np.nan, columns = self.variables_to_compile, index = date_time_span)

        date_time_span = pd.date_range(start = '2015-01-01 12:00:00', end = self.max_year+'-12-31 12:00:00', freq = 'M')
        self.df_monthly = pd.DataFrame(np.nan, columns = self.variables_to_compile, index = date_time_span)

        date_time_span = pd.date_range(start = '2015-01-01 12:00:00', end = self.max_year+'-12-31 12:00:00', freq = 'y')
        self.df_annual = pd.DataFrame(np.nan, columns = self.variables_to_compile, index = date_time_span)

        date_time_span = pd.date_range(start = '2020-01-01 12:00:00', end = self.max_year+'-12-31 12:00:00', freq = '10y')
        self.df_decade = pd.DataFrame(np.nan, columns = self.variables_to_compile, index = date_time_span)

    # Gets all the files to read in a directory for a model
    def get_files(self):
        self.file_list = []
        for variable in self.variables_to_compile:
            # Except statement for different ensembles
            paths = [
                self.path+'/'+self.model_name+'/'+self.experiment_name+'/r1i1p1f1/'+variable+'/',
                self.path+'/'+self.model_name+'/'+self.experiment_name+'/r1i1p1f2/'+variable+'/',
                self.path+'/'+self.model_name+'/'+self.experiment_name+'/r1i1p1f3/'+variable+'/',
                self.path+'/'+self.model_name+'/'+self.experiment_name+'/r3i1p1f1/'+variable+'/',
                self.path+'/'+self.model_name+'/'+self.experiment_name+'/r4i1p1f1/'+variable+'/',
            ]
            for variable_path in paths:
                try:
                    filenames = os.listdir(variable_path)
                    break
                except:
                    print('Wrong Directory, trying next directory.')
            
            for file in filenames:
                self.file_list.append(variable_path+file)


    # For each file, reads a single file and retuns a series/dataframe with the index as datetime and values as the variable name
    def read_one_file(self, coord, filename, variable):
        handle = xr.open_dataset(filename)

        # Geting Lat Long
        lat = handle['lat'][...].values
        lon = handle['lon'][...].values-180
        ilat = np.argmin(np.abs(lat - coord[0]))
        ilon = np.argmin(np.abs(lon - coord[1]))
        self.data_lat = lat[ilat]
        self.data_lon = lon[ilon]
        
        # Putting into a Data frame
        df = pd.DataFrame()
        # This try handles two formats for weather file inputs, pd.datetime and cftime, if it can't convert to 
        # datetime natively, recodes the time to handle cftime
        try: 
            time = handle.indexes['time']
            df.index = pd.Series(pd.to_datetime(time))
        except: 
            time = handle.indexes['time'].to_datetimeindex(unsafe = True)
            df.index = pd.Series(pd.to_datetime(time))
        df[variable] = handle[variable][:,ilat, ilon].values

        # Performing temperature calculation if necessary
        if 'tas' in variable:
            df[variable] = df[variable]-273.0 # Converting to Celsius and adding model bias
        if 'pr' in variable:
            df[variable] = df[variable]*86400 # Converting to mm

        return df[variable]

    # Stores all the data produced by reading one file into the data frame which can then carry the data with it through the model
    def compile_data(self, coord):
        print('Data Compiling for ', self.model_name)
        # Adding each data file to the internal data frame
        for file in self.file_list:

            if 'hurs' in file:
                variable = 'hurs'
            elif 'huss' in file:
                variable = 'huss'
            elif 'rlds' in file:
                variable = 'rlds'
            elif 'rsds' in file:
                variable = 'rsds'
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

            try:
                temp_series = self.read_one_file(coord, file, variable)
                self.df_data.loc[temp_series.index, variable] = temp_series
            except:
                print(file)
                continue

        # Changing the data compiled boolean to true
        self.bool_data_compiled = True
        print('Data Compiled!')