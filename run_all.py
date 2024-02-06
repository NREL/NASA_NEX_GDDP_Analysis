# Importing all dependencies
from joblib import register_parallel_backend
from dependencies import *

# Importing Custom Scripts
from load_model_v2 import model_obj
#from analyze_model_v2 import analyze_model
#from graphing_v2 import graph

def loading_raw_files(coord, rank=0, mpi_size=1):
    # Defining climate analysis inputs
    path = 'NEX-GDDP-CMIP6'
    experiments = ['ssp245', 'ssp585']

    # Getting list of all model names dynamically
    model_file_list = os.listdir(path)
    model_name_list = []
    for item in model_file_list:
        if 'download' in item:
            continue
        elif '.' in item:
            continue
        else:
            model_name_list.append(item)
    model_name_list.sort()

    # Getting list of models along with file names
    model_list = []
    for model_name in model_name_list:
        for experiment in experiments:
            model_list.append(model_obj(path, model_name, experiment))
    
    # Getting files
    bucket = int(len(model_list))
    if mpi_size > 1: 
        bucket = int(len(model_list)/mpi_size)

    start_idx = int(rank*bucket)

    if (rank+1)*bucket > len(model_list): 
        end_idx = int(len(model_list))
    else:
         end_idx = int((rank+1)*bucket)

    for model in model_list[start_idx : end_idx]:
        print(f"Rank {rank} out of {mpi_size} :  start index {start_idx} , end index {end_idx}", flush=True)
        try:
            model.get_files()
            model.compile_data(coord)
            model.df_data = model.df_data.round(decimals=2)
            model.df_data.to_csv('tmp/'+model.model_name+'_'+model.experiment_name+'.csv')
        except: 
            print(model.model_name, ' cannot be read')
            continue

    # # Getting files
    # for model in model_list[45:]:
    #     try:
    #         model.get_files()
    #         model.compile_data(coord)
    #         model.df_data.to_csv('tmp/'+model.model_name+'_'+model.experiment_name+'.csv')
    #     except: 
    #         print(model.model_name, ' cannot be read')
    #         continue

    # Compiling the data into dataframes
    '''   
    for model in model_list:
        model.compile_data(coord)

    for model in model_list:
        model.df_data.to_csv('tmp/'+model.model_name+'_'+model.experiment_name+'_'+str(model.data_lat)+'_'+str(model.data_lon)+'.csv')
    '''
    print(coord)

    return model_list

def load_dataframes():
    path = 'tmp'
    files = os.listdir(path)

    model_list = []
    for file in files:
        if '.DS' in file:
            continue
        items = file.split('_')
        model = model_obj(path+'/'+file, items[0], items[1])
        model.df_data = pd.read_csv(path+'/'+file, header = 0, index_col='Unnamed: 0')
        model.df_data.index = pd.to_datetime(model.df_data.index)
        model_list.append(model)

    return model_list
