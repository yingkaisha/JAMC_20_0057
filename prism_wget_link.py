'''
PRISM NRT data
Generating download links
source: http://prism.oregonstate.edu/documents/PRISM_downloads_web_service.pdf

The script produces 'wget_{}.txt'.format(var)
PRISM data can be obtained by wget this txt file. 
'''

# general tools
import sys
import subprocess
from glob import glob
from datetime import datetime, timedelta

# custom tools
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/utils/')
sys.path.insert(0, '/glade/u/home/ksha/WORKSPACE/DL_downscaling/')
from namelist import * 

# date range
base = datetime(2015, 1, 1, 0)
N_days = 365 + 366 + 365 + 365 + 365 # 2015-2020 (period ending)
date_list = [base + timedelta(days=x) for x in range(N_days)]
print('Generating wget script from {} to {}'.
      format(datetime.strftime(date_list[0], '%Y%m%d'), datetime.strftime(date_list[-1], '%Y%m%d')))

# macros
vars = ['PCT']
del_old = False # delete old files
del_txt = True  # delete old wget scripts

# wget script target path
dirs = {}
dirs['PCT']   = PRISM_PCT_dir
dirs['TMAX']  = PRISM_TMAX_dir
dirs['TMIN']  = PRISM_TMIN_dir
dirs['TMEAN'] = PRISM_TMEAN_dir

# PRISM server keywords
keywords = {} # 
keywords['PCT']   = 'ppt'
keywords['TMAX']  = 'tmax'
keywords['TMIN']  = 'tmin'
keywords['TMEAN'] = 'tmean'

for var in vars:
    print('===== Extracting {} ====='.format(var))
    
    if del_old:
        # delete old files
        cmd = 'rm -rf {}*20*'.format(dirs[var])
        print(cmd)
        subprocess.call(cmd, shell=True)
    if del_txt:
        # delete old wget script
        cmd = 'rm -f {}*txt*'.format(dirs[var])
        print(cmd)
        subprocess.call(cmd, shell=True)
    
    # base_link
    base_link = 'http://services.nacse.org/prism/data/public/4km/{}/'.format(keywords[var])

    # wget link gen
    filename = 'wget_{}.txt'.format(var)
    print('Creating {}'.format(filename))
    f_io = open(dirs[var]+filename, 'w')
    for date in date_list:
        # print(...)
        f_io.write(base_link+datetime.strftime(date, '%Y%m%d')+'\n') # multi-lines
    f_io.close()
