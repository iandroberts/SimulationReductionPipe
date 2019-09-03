from __future__ import print_function
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from astropy.table import Table
import scipy.stats as st

####################

def ADpx(vx, x):
    H0 = 68.
    N = len(vx)

    vxH0 = (vx - np.average(vx)) + H0 * (x - np.average(x))
    Ax = st.anderson(vxH0).statistic
    AAx = (1 + 0.75/N + 2.25/N**2) * Ax
    
    if AAx < 0.2:
        pval_x = 1. - np.exp(-13.436 + 101.14 * AAx - 223.73 * AAx**2)
    elif AAx < 0.34:
        pval_x = 1. - np.exp(-8.318 + 42.796 * AAx - 59.938 * AAx**2)
    elif AAx < 0.6:
        pval_x = np.exp(0.9177 - 4.279 * AAx - 1.38 * AAx**2)
    elif AAx < 10:
        pval_x = np.exp(1.2937 - 5.709 * AAx + 0.0186 * AAx**2)
    else:
        pval_x = 3.7e-24

    return pval_x

def ADpy(vy, y):
    H0 = 68.
    N = len(vy)

    vyH0 = (vy - np.average(vy)) + H0 * (y - np.average(y))
    Ay = st.anderson(vyH0).statistic
    AAy = (1 + 0.75/N + 2.25/N**2) * Ay

    if AAy < 0.2:
        pval_y = 1. - np.exp(-13.436 + 101.14 * AAy - 223.73 * AAy**2)
    elif AAy < 0.34:
        pval_y = 1. - np.exp(-8.318 + 42.796 * AAy - 59.938 * AAy**2)
    elif AAy < 0.6:
        pval_y = np.exp(0.9177 - 4.279 * AAy - 1.38 * AAy**2)
    elif AAy < 10:
        pval_y = np.exp(1.2937 - 5.709 * AAy + 0.0186 * AAy**2)
    else:
        pval_y = 3.7e-24

    return pval_y

def ADpz(vz, z):
    
    H0 = 68.
    N = len(vz)
    
    vzH0 = (vz - np.average(vz)) + H0 * (z - np.average(z))
    Az = st.anderson(vzH0).statistic
    AAz = (1 + 0.75/N + 2.25/N**2) * Az

    if AAz < 0.2:
        pval_z = 1. - np.exp(-13.436 + 101.14 * AAz - 223.73 * AAz**2)
    elif AAz < 0.34:
        pval_z = 1. - np.exp(-8.318 + 42.796 * AAz - 59.938 * AAz**2)
    elif AAz < 0.6:
        pval_z = np.exp(0.9177 - 4.279 * AAz - 1.38 * AAz**2)
    elif AAz < 10:
        pval_z = np.exp(1.2937 - 5.709 * AAz + 0.0186 * AAz**2)
    else:
        pval_z = 3.7e-24

    return pval_z

def f(df):
    d = {}
    d['pval_x'] = ADpx(df['col20'], df['col17'])
    d['pval_y'] = ADpx(df['col21'], df['col18'])
    d['pval_z'] = ADpx(df['col22'], df['col19'])

    return pd.Series(d, index=['pval_x', 'pval_y', 'pval_z'])

####################

def main(s):
    parent = pd.read_table('/home/idroberts/mdpl2/processed_halo_cats/md{}_parentMass.dat'.format(s), delimiter='\s+', engine='c', header=None)

    print('...')
    print('Parents loaded...')
    print('...')

    gaID = pd.read_table('/home/idroberts/mdpl2/processed_halo_cats/md{}_gaID.dat'.format(s), delimiter='\s+', engine='c', header=None)

    print('...')
    print('IDs loaded...')
    print('...')

    datalist = []

    for ga_chunk in tqdm(pd.read_table('/home/idroberts/mdpl2/processed_halo_cats/md{}_gaMass.dat'.format(s), delimiter='\s+', engine='c', header=None, chunksize=1000000)):
        merged = pd.merge(ga_chunk, gaID, left_on=1, right_on=0)

        datalist.append(merged)

    df = pd.concat(datalist)

    print('...')
    print('Chunks merged...')
    print('...')

    new_names = []
    for i in np.arange(df.shape[1]):
        new_names.append('col{}'.format(i))

    new_names_par = []
    for i in np.arange(parent.shape[1]):
        new_names_par.append('col{}'.format(i))

    df.columns = new_names
    parent.columns = new_names_par
    

    grouped = df[['col1', 'col6']].groupby(by='col6', as_index=False).count()

    print('...')
    print('Finding parents with 10 or more members...')
    print('...')

    groupedN10 = grouped[grouped['col1']>=10]

    mergedN10 = pd.merge(df, groupedN10, on='col6')

    mergedN10_parent = pd.merge(parent, groupedN10, left_on='col1', right_on='col6')

    print('...')
    print('Applying AD test...')
    print('...')

    ADframe = mergedN10[['col6','col17','col18','col19','col20','col21','col22']].groupby(by='col6').apply(f).reset_index()

    ADframe.rename(columns={'col6':'upid'}, inplace=True)

    time0 = time.time()

    print('...')
    print('Converting GA frame to Astropy table...')
    print('...')

    t = Table.from_pandas(mergedN10)

    print('...')
    print('Converting parent frame to Astropy table...')
    print('...')

    tpar = Table.from_pandas(mergedN10_parent)

    print('...')
    print('Converting AD frame to Astropy table...')
    print('...')

    tAD = Table.from_pandas(ADframe)

    print('...')
    print('Saving GA frame to fits table...')
    print('...')

    t.write('md{}_ga_N10.fits'.format(s), format='fits', overwrite=True)

    time1 = time.time()

    print('That took {:.1f} seconds'.format(time1 - time0))

    time0 = time.time()

    print('...')
    print('Saving parent frame to fits table...')
    print('...')

    tpar.write('md{}_parent.fits'.format(s), format='fits', overwrite=True)

    time1 = time.time()

    print('That took {:.1f} seconds'.format(time1 - time0))

    time0 = time.time()

    print('...')
    print('Saving AD frame to fits table...')
    print('...')

    tAD.write('md{}_upid_ADp_H0.fits'.format(s), format='fits', overwrite=True)

    time1 = time.time()

    print('That took {:.1f} seconds'.format(time1 - time0))

    return None


#s_in = np.genfromtxt('/home/idroberts/mdpl2/snaplist.txt', usecols=(0,), unpack=True, dtype='str')
#print(str(s_in).zfill(6))
#for s in tqdm(s_in):
s = '50320'
    
main(s.zfill(6))
