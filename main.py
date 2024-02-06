# Importing all dependencies
from dependencies import *
from run_all import *
from analyze_model_v2 import analyze_model
from graphing_v2 import *

# module to parallelize runs 
from mpi4py import MPI

# comunicator object
comm = MPI.COMM_WORLD
#rank of the MPI process
rank = comm.Get_rank()
mpi_size = comm.Get_size()

bool_load_all = True
coord = (39.74256572165156, -105.16857014167543)
#coord_golden = (39.74256572165156, -105.16857014167543)
coord_langley = (37.08920266916765, -76.37973529040504)

coord = coord_langley

# Loading the model from raw files or loading them from dataframes
if bool_load_all:    
    model_list = loading_raw_files(coord, rank, mpi_size)
else:
    model_list = load_dataframes(rank, mpi_size)