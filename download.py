import os
import subprocess

model_names = [
    'ACCESS-CM2',
    'ACCESS-ESM1-5',
    'BCC-CSM2-MR',
    'CanESM5',
    'CESM2-WACCM',
    'CESM2',
    'CMCC-CM2-SR5',
    'CMCC-ESM2',
    'CNRM-CM6-1',
    'CNRM-ESM2-1',
    'EC-Earth3-Veg-LR',
    'EC-Earth3',
    'FGOALS-g3',
    'GFDL-CM4',
    'GFDL-CM4_gr2',
    'GFDL-ESM4',
    'GISS-E2-1-G',
    'HadGEM3-GC31-LL',
    'HadGEM3-GC31-MM',
    'IITM-ESM',
    'INM-CM4-8',
    'INM-CM5-0',
    'IPSL-CM6A-LR',
    'KACE-1-0-G',
    'KIOST-ESM',
    'MIROC-ES2L',
    'MIROC6',
    'MPI-ESM1-2-HR',
    'MPI-ESM1-2-LR',
    'MRI-ESM2-0',
    'NESM3',
    'NorESM2-LM',
    'NorESM2-MM',
    'TaiESM1',
    'UKESM1-0-LL'
]

for model_name in model_names[10:11]:
    cmd_245 = 'aws s3 sync --no-sign-request s3://nex-gddp-cmip6/NEX-GDDP-CMIP6/'+model_name+'/ssp245/ NEX-GDDP-CMIP6/'+model_name+'/ssp245/'
    cmd_585 = 'aws s3 sync --no-sign-request s3://nex-gddp-cmip6/NEX-GDDP-CMIP6/'+model_name+'/ssp585/ NEX-GDDP-CMIP6/'+model_name+'/ssp585/'
    try:
        print(cmd_245)
        os.system(cmd_245)
        print(cmd_585)
        os.system(cmd_585)
    except: 
        continue
