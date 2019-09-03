from __future__ import print_function
import numpy as np
from tqdm import tqdm, trange
import pandas as pd
from pathos.pools import ProcessPool

s_in = input('Snapshot scale? ')
s = s_in.zfill(6)

#haloID, Pid, Upid, Mpeak = np.loadtxt('mdz0_gaInput.dat', unpack=True)
df = pd.read_csv('/home/idroberts/mdpl2/processed_halo_cats/md{}_gaInput.dat'.format(s), delimiter='\s+', names=['haloID', 'pid', 'upid', 'Mp'])

print('Loaded')

################################

maskSH = np.isin(df.pid, -1, invert=True) # identify sub-halos

SH_id = np.array(df.haloID)[maskSH] # subhalo id
SH_pid = np.array(df.pid)[maskSH] # subhalo immmediate parent id
SH_upid = np.array(df.upid)[maskSH] # subhalo ultimate parent id
SH_M = np.array(df.Mp)[maskSH] # subhalo peak mass

iGalMass = np.where((SH_M >= 10 ** 11) & (SH_M < 10 ** 12.5))

SH_id_GM = SH_id[iGalMass] 
SH_pid_GM = SH_pid[iGalMass]
SH_upid_GM = SH_upid[iGalMass] 
SH_M_GM = SH_M[iGalMass] 


##########################
### Construct tree #########################################################
##########################

SHtree = dict()

mask = np.isin(SH_pid, SH_upid) # level one subhalo mask
 
for x in trange(1, 15): # loop through suhalo tree levels

    SHtree[x] = SH_id[mask]

    mask = np.isin(SH_pid, SHtree[x]) # level 'x' subhalo mask

    if np.sum(mask) == 0:
        break

for i in range(1,len(SHtree)): # exclude "ambiguous parent" subhalos
    mask1 = np.isin(SHtree[i], SHtree[i+1], invert=True)
    mask2 = np.isin(SHtree[i+1], SHtree[i], invert=True)

    current1 = SHtree[i]
    current2 = SHtree[i+1]

    SHtree[i] = current1[mask1]
    SHtree[i+1] = current2[mask2]

    print(np.sum(np.isin(SHtree[i], SHtree[i+1])), len(SHtree[i]))

##########################
### Level one #############################################################
##########################

mask1 = np.isin(SH_id, SHtree[1])

SH_id1 = SH_id[mask1]
SH_pid1 = SH_pid[mask1]
SH_upid1 = SH_upid[mask1]
SH_M1 = SH_M[mask1]

iGM1 = np.where((SH_M1 >= 10 ** 11) & (SH_M1 < 10 ** 12.5))

SH_id1_GM = SH_id1[iGM1]
SH_M1_GM = SH_M1[iGM1]

mask1_noGMChildren = np.isin(SH_id1_GM, SH_pid_GM, invert=True)
mask1_GMChildren = np.isin(SH_id1_GM, SH_pid_GM)

GA1a = SH_id1_GM[mask1_noGMChildren]

print('No. of GA at lev1 wo children:', len(GA1a))

SH_id1_GM_wChild = SH_id1_GM[mask1_GMChildren]
SH_M1_GM_wChild = SH_M1_GM[mask1_GMChildren]

def level1(SH_id1_GM_wChild, SH_M1_GM_wChild, SH_pid, SH_M, indices):
    GAlist = np.zeros(len(SH_id1_GM_wChild))

    for i in tqdm(indices):
        iChild = np.where(SH_pid == SH_id1_GM_wChild[i])

        if (len(iChild[0]) > 0):
            Msub = np.sum(SH_M[iChild])
            Mrem = SH_M1_GM_wChild[i] - Msub
            if (Mrem >= 10 ** 11) and (Mrem < 10 ** 12.5):
                GAlist[i] = SH_id1_GM_wChild[i]
    return GAlist

Nproc = 20
IDs_split = np.array_split(np.arange(SH_id1_GM_wChild.size), Nproc)
pool = ProcessPool(Nproc)

result = {}

for i in range(Nproc):
    result['res{}'.format(i)] = pool.apipe(level1, SH_id1_GM_wChild, SH_M1_GM_wChild, SH_pid, SH_M, IDs_split[i])

ans = {}

GAlist = np.concatenate([result['res{}'.format(i)].get() for i in np.arange(Nproc)])

#for i in range(Nproc):
    #ans['ans{}'.format(i)] = result['res{}'.format(i)].get()


GA1b = GAlist[GAlist > 0]

print('No. of GA at lev1 w children:', len(GA1b))

GA1 = np.concatenate((GA1a, GA1b))

##########################
### Level two #############################################################
##########################

mask2 = np.isin(SH_id, SHtree[2])

SH_id2 = SH_id[mask2]
SH_pid2 = SH_pid[mask2]
SH_upid2 = SH_upid[mask2]
SH_M2 = SH_M[mask2]

iGM2 = np.where((SH_M2 >= 10 ** 11) & (SH_M2 < 10 ** 12.5))

SH_id2_GM = SH_id2[iGM2]
SH_M2_GM = SH_M2[iGM2]

mask2_noGMChildren = np.isin(SH_id2_GM, SH_pid_GM, invert=True)
mask2_GMChildren = np.isin(SH_id2_GM, SH_pid_GM)

GA2a = SH_id2_GM[mask2_noGMChildren]

print('No. of GA at lev2 wo children:', len(GA2a))

SH_id2_GM_wChild = SH_id2_GM[mask2_GMChildren]
SH_M2_GM_wChild = SH_M2_GM[mask2_GMChildren]

IDs_split = np.array_split(np.arange(SH_id2_GM_wChild.size), Nproc)
pool = ProcessPool(Nproc)

result = {}

for i in range(Nproc):
    result['res{}'.format(i)] = pool.apipe(level1, SH_id2_GM_wChild, SH_M2_GM_wChild, SH_pid, SH_M, IDs_split[i])

ans = {}

#for i in range(Nproc):
    #ans['ans{}'.format(i)] = result['res{}'.format(i)].get()

GAlist = np.concatenate([result['res{}'.format(i)].get() for i in np.arange(Nproc)])

GA2b = GAlist[GAlist > 0]

print('No. of GA at lev2 w children:', len(GA2b))

GA2 = np.concatenate((GA2a, GA2b))

##########################
### Level three ##########################################################
##########################

mask3 = np.isin(SH_id, SHtree[3])

SH_id3 = SH_id[mask3]
SH_pid3 = SH_pid[mask3]
SH_upid3 = SH_upid[mask3]
SH_M3 = SH_M[mask3]

iGM3 = np.where((SH_M3 >= 10 ** 11) & (SH_M3 < 10 ** 12.5))

SH_id3_GM = SH_id3[iGM3]
SH_M3_GM = SH_M3[iGM3]

mask3_noGMChildren = np.isin(SH_id3_GM, SH_pid_GM, invert=True)
mask3_GMChildren = np.isin(SH_id3_GM, SH_pid_GM)

GA3a = SH_id3_GM[mask3_noGMChildren]

print('No. of GA at lev3 wo children:', len(GA3a))

SH_id3_GM_wChild = SH_id3_GM[mask3_GMChildren]
SH_M3_GM_wChild = SH_M3_GM[mask3_GMChildren]

IDs_split = np.array_split(np.arange(SH_id3_GM_wChild.size), Nproc)
pool = ProcessPool(Nproc)

result = {}

for i in range(Nproc):
    result['res{}'.format(i)] = pool.apipe(level1, SH_id3_GM_wChild, SH_M3_GM_wChild, SH_pid, SH_M, IDs_split[i])

ans = {}

GAlist = np.concatenate([result['res{}'.format(i)].get() for i in np.arange(Nproc)])

GA3b = GAlist[GAlist > 0]

print('No. of GA at lev3 w children:', len(GA3b))

GA3 = np.concatenate((GA3a, GA3b))

##########################
### Level four ###########################################################
##########################

mask4 = np.isin(SH_id, SHtree[4])

SH_id4 = SH_id[mask4]
SH_pid4 = SH_pid[mask4]
SH_upid4 = SH_upid[mask4]
SH_M4 = SH_M[mask4]

iGM4 = np.where((SH_M4 >= 10 ** 11) & (SH_M4 < 10 ** 12.5))

SH_id4_GM = SH_id4[iGM4]
SH_M4_GM = SH_M4[iGM4]

mask4_noGMChildren = np.isin(SH_id4_GM, SH_pid_GM, invert=True)
mask4_GMChildren = np.isin(SH_id4_GM, SH_pid_GM)

GA4a = SH_id4_GM[mask4_noGMChildren]

print('No. of GA at lev4 wo children:', len(GA4a))

SH_id4_GM_wChild = SH_id4_GM[mask4_GMChildren]
SH_M4_GM_wChild = SH_M4_GM[mask4_GMChildren]

IDs_split = np.array_split(np.arange(SH_id4_GM_wChild.size), Nproc)
pool = ProcessPool(Nproc)

result = {}

for i in range(Nproc):
    result['res{}'.format(i)] = pool.apipe(level1, SH_id4_GM_wChild, SH_M4_GM_wChild, SH_pid, SH_M, IDs_split[i])

ans = {}

GAlist = np.concatenate([result['res{}'.format(i)].get() for i in np.arange(Nproc)])

GA4b = GAlist[GAlist > 0]

print('No. of GA at lev4 w children:', len(GA4b))

GA4 = np.concatenate((GA4a, GA4b))

##########################
### Level five ###########################################################
##########################

mask5 = np.isin(SH_id, SHtree[5])

SH_id5 = SH_id[mask5]
SH_pid5 = SH_pid[mask5]
SH_upid5 = SH_upid[mask5]
SH_M5 = SH_M[mask5]

iGM5 = np.where((SH_M5 >= 10 ** 11) & (SH_M5 < 10 ** 12.5))

SH_id5_GM = SH_id5[iGM5]
SH_M5_GM = SH_M5[iGM5]

mask5_noGMChildren = np.isin(SH_id5_GM, SH_pid_GM, invert=True)
mask5_GMChildren = np.isin(SH_id5_GM, SH_pid_GM)

GA5a = SH_id5_GM[mask5_noGMChildren]

print('No. of GA at lev5 wo children:', len(GA5a))

SH_id5_GM_wChild = SH_id5_GM[mask5_GMChildren]
SH_M5_GM_wChild = SH_M5_GM[mask5_GMChildren]

IDs_split = np.array_split(np.arange(SH_id5_GM_wChild.size), Nproc)
pool = ProcessPool(Nproc)

result = {}

for i in range(Nproc):
    result['res{}'.format(i)] = pool.apipe(level1, SH_id5_GM_wChild, SH_M5_GM_wChild, SH_pid, SH_M, IDs_split[i])

GAlist = np.concatenate([result['res{}'.format(i)].get() for i in np.arange(Nproc)])

GA5b = GAlist[GAlist > 0]

print('No. of GA at lev5 w children:', len(GA5b))

GA5 = np.concatenate((GA5a, GA5b))

##########################
### Level six ###########################################################
##########################

mask6 = np.isin(SH_id, SHtree[6])

SH_id6 = SH_id[mask6]
SH_pid6 = SH_pid[mask6]
SH_upid6 = SH_upid[mask6]
SH_M6 = SH_M[mask6]

iGM6 = np.where((SH_M6 >= 10 ** 11) & (SH_M6 < 10 ** 12.5))

SH_id6_GM = SH_id6[iGM6]
SH_M6_GM = SH_M6[iGM6]

mask6_noGMChildren = np.isin(SH_id6_GM, SH_pid_GM, invert=True)
mask6_GMChildren = np.isin(SH_id6_GM, SH_pid_GM)

GA6a = SH_id6_GM[mask6_noGMChildren]

print('No. of GA at lev6 wo children:', len(GA6a))

SH_id6_GM_wChild = SH_id6_GM[mask6_GMChildren]
SH_M6_GM_wChild = SH_M6_GM[mask6_GMChildren]

IDs_split = np.array_split(np.arange(SH_id6_GM_wChild.size), Nproc)
pool = ProcessPool(Nproc)

result = {}

for i in range(Nproc):
    result['res{}'.format(i)] = pool.apipe(level1, SH_id6_GM_wChild, SH_M6_GM_wChild, SH_pid, SH_M, IDs_split[i])

GAlist = np.concatenate([result['res{}'.format(i)].get() for i in np.arange(Nproc)])

GA6b = GAlist[GAlist > 0]

print('No. of GA at lev6 w children:', len(GA6b))

GA6 = np.concatenate((GA6a, GA6b))

##########################
### Level seven ###########################################################
##########################

mask7 = np.isin(SH_id, SHtree[7])

SH_id7 = SH_id[mask7]
SH_pid7 = SH_pid[mask7]
SH_upid7 = SH_upid[mask7]
SH_M7 = SH_M[mask7]

iGM7 = np.where((SH_M7 >= 10 ** 11) & (SH_M7 < 10 ** 12.5))

SH_id7_GM = SH_id7[iGM7]
SH_M7_GM = SH_M7[iGM7]

mask7_noGMChildren = np.isin(SH_id7_GM, SH_pid_GM, invert=True)
mask7_GMChildren = np.isin(SH_id7_GM, SH_pid_GM)

GA7a = SH_id7_GM[mask7_noGMChildren]

print('No. of GA at lev7 wo children:', len(GA7a))

SH_id7_GM_wChild = SH_id7_GM[mask7_GMChildren]
SH_M7_GM_wChild = SH_M7_GM[mask7_GMChildren]

IDs_split = np.array_split(np.arange(SH_id7_GM_wChild.size), Nproc)
pool = ProcessPool(Nproc)

result = {}

for i in range(Nproc):
    result['res{}'.format(i)] = pool.apipe(level1, SH_id7_GM_wChild, SH_M7_GM_wChild, SH_pid, SH_M, IDs_split[i])

GAlist = np.concatenate([result['res{}'.format(i)].get() for i in np.arange(Nproc)])

GA7b = GAlist[GAlist > 0]

print('No. of GA at lev7 w children:', len(GA7b))

GA7 = np.concatenate((GA7a, GA7b))

##########################
### Total ##############################################################
##########################

GA = np.concatenate((GA1, GA2, GA3, GA4, GA5, GA6, GA7))

print('Total no. of GA:', len(GA))

np.savetxt('/home/idroberts/mdpl2/processed_halo_cats/md{}_gaID.dat'.format(s), GA, fmt='%d')
