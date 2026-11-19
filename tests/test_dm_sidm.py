"""DM/SIDM integration tests. Run .venv/bin/python tests/test_dm_sidm.py.
Uses a temporary directory; requires the CMake build and local MPI sockets.
"""
from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile
import json
import h5py
import numpy as np
import pandas as pd
from cosmic.sample import InitialCMCTable

REPO = Path(__file__).resolve().parents[1]
OUT = Path(tempfile.mkdtemp(prefix='cmc-sidm-tests-'))
EXE = Path(os.environ.get('CMC_TEST_EXECUTABLE', REPO/'build/src/cmc/cmc')).resolve()
RESULTS = []
BASE = (REPO/'examples/PlummerSphere.ini').read_text()
s,b=InitialCMCTable.sampler('cmc_point_mass',cluster_profile='plummer',size=1000,
    r_max=100,seed=12345,virial_radius=1.,tidal_radius=1e6)
s['m']=1.;s['k']=1;s['Reff']=0.
s.mass_of_cluster=None;s.scaled_to_nbody_units=False
InitialCMCTable.write(s,b,filename=str(OUT/'stars.hdf5'))
for name in ['dm','mixed','unequal','one','odd','cross_rank','bad_type']:
    shutil.copy2(OUT/'stars.hdf5',OUT/f'{name}.hdf5')
    with h5py.File(OUT/f'{name}.hdf5','r+') as f:
        ds=f['CLUS_OBJ_DATA/block0_values'];cols=list(f['CLUS_OBJ_DATA/block0_items'].asstr()[:])
        rows=slice(1,-1) if name in ['dm','unequal'] else (slice(1,2) if name=='one' else slice(1,998) if name=='odd' else slice(1,-1,2))
        if name=='cross_rank': rows=slice(500,502)  # np=2, MIN_CHUNK_SIZE=20
        ds[rows,cols.index('k')]=-200
        if name=='unequal': ds[1,cols.index('m')]*=2
        if name=='bad_type': ds[1,cols.index('k')]=-201

# Valid binary table shape, with deliberately forbidden DM labels in either
# the COM or a component. Exercise both rank-local input and particle loading.
for name in ['dm_binary_com','dm_binary_member']:
    objects=pd.read_hdf(OUT/'stars.hdf5','CLUS_OBJ_DATA')
    binaries=pd.read_hdf(OUT/'stars.hdf5','CLUS_BINARY_DATA')
    with h5py.File(OUT/'stars.hdf5') as f:
        attrs=dict(f['CLUS_OBJ_DATA/block0_values'].attrs)
    objects.iloc[1,objects.columns.get_loc('binind')]=1
    if name=='dm_binary_com': objects.iloc[1,objects.columns.get_loc('k')]=-200
    row=dict(index=1,id1=1,k1=-200 if name=='dm_binary_member' else 1,
             m1=.0005,Reff1=0,id2=1001,k2=1,m2=.0005,Reff2=0,a=1e-5,e=.1)
    binaries=pd.concat([binaries.iloc[:1],pd.DataFrame([row],columns=binaries.columns)],ignore_index=True).astype(float)
    objects.to_hdf(OUT/f'{name}.hdf5',key='CLUS_OBJ_DATA',mode='w')
    binaries.to_hdf(OUT/f'{name}.hdf5',key='CLUS_BINARY_DATA')
    with h5py.File(OUT/f'{name}.hdf5','a') as f:
        d=f['CLUS_OBJ_DATA/block0_values']
        for k,v in attrs.items(): d.attrs[k]=v
        d.attrs['NBINARY']=1

DEFAULT=dict(INPUT_FILE='initial.hdf5',BINBIN=0,BINSINGLE=0,THREEBODYBINARIES=0,
    ONLY_FORM_BH_THREEBODYBINARIES=0,SS_COLLISION=0,BH_CAPTURE=0,
    TIDAL_CAPTURE=0,TC_POLYTROPE=0,WD_TC=0,RELAXATION=1,PERTURB=1,
    STELLAR_EVOLUTION=0,STOPATCORECOLLAPSE=0,TIDALLY_STRIP_STARS=0,
    T_MAX_COUNT=4,T_MAX=100,SAMPLESIZE=16,SNAPSHOTTING=1,SNAPSHOT_DELTACOUNT=1,
    SNAPSHOT_CORE_COLLAPSE=0,SNAPSHOT_WINDOWS='NULL')

def run(name,source='dm',np_=2,fail=None,restart=None,env_changes=None,**changes):
    folder=OUT/name;folder.mkdir();(folder/'initial.hdf5').symlink_to(OUT/f'{source}.hdf5')
    cfg=BASE
    for k,v in (DEFAULT|changes).items():
        pattern=r'^'+k+r'\s*=.*$'
        if re.search(pattern,cfg,re.M):cfg=re.sub(pattern,f'{k} = {v}',cfg,flags=re.M)
        else: cfg=cfg.replace('[cmc]',f'[cmc]\n{k} = {v}',1)
    (folder/'run.ini').write_text(cfg)
    args=['mpirun','--oversubscribe','--bind-to','none','-np',str(np_),str(EXE)]
    if restart:
        shutil.copytree(restart/'test-RESTART',folder/'test-RESTART')
        args+=['-R','1']
    args+=['run.ini','test']
    if restart:args+=['test']
    with (folder/'run.log').open('w') as log:
        result=subprocess.run(args,cwd=folder,stdout=log,stderr=subprocess.STDOUT,timeout=100,
            env=os.environ|{'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}|(env_changes or {}))
    text=(folder/'run.log').read_text()
    if fail: assert result.returncode and fail in text,(name,text[-3000:])
    else: assert result.returncode==0,(name,text[-3000:])
    RESULTS.append({'name':name,'ranks':np_,'returncode':result.returncode,'expected_failure':bool(fail)})
    print('PASS',name,flush=True)
    return folder

def snapshots(folder):
    with h5py.File(folder/'test.snapshots.h5') as f:
        return [f[k][:] for k in sorted(f.keys(),key=lambda k:int(k.split('(')[0]))]

def check_dm(folder,count):
    initial_masses = None
    for a in snapshots(folder):
        dm=a[a['startype']==-200]
        assert len(dm)==count,(len(dm),count)
        assert np.all(dm['binflag']==-100)
        assert np.all(dm['radius_RSUN']==0)
        mass_by_id = dict(zip(dm['id'],dm['m_MSUN']))
        if initial_masses is None: initial_masses=mass_by_id
        assert mass_by_id==initial_masses, 'DM mass changed during evolution'

zero=run('gravity_only')
zero_off=run('gravity_only_relax_off',RELAXATION=0)
assert (zero/'test.dyn.dat').read_bytes()==(zero_off/'test.dyn.dat').read_bytes()
check_dm(zero,1000)
# Deliberately large cross section for an inexpensive event-rich numerical test.
active=run('sidm',SIDM_SIGMA_OVER_M=1e6,T_MAX_COUNT=40,PERTURB=0)
x=np.atleast_2d(np.loadtxt(active/'test.sidm.dat'))
assert x[:,5].sum()>20
assert abs(x[:,5].sum()-x[:,6].sum())<6*np.sqrt(x[:,6].sum())
assert np.max(x[:,7])<=.05*(1+1e-10)
assert np.max(np.abs(x[:,8]))<1e-12
check_dm(active,1000)
for name,source,count in [('mixed','mixed',500),('odd','odd',997),('single_dm','one',1)]:
    f=run(name,source=source,SIDM_SIGMA_OVER_M=1e6,SIDM_SHELL_SIZE=31)
    check_dm(f,count)
serial=run('serial',np_=1,SIDM_SIGMA_OVER_M=1e6)
check_dm(serial,1000)
se=run('stellar_evolution',source='mixed',STELLAR_EVOLUTION=1,SIDM_SIGMA_OVER_M=1e6)
check_dm(se,500)
run('reject_bad_type',source='bad_type',fail='invalid integer')
run('reject_dm_binary_com',source='dm_binary_com',fail='DM particles must be single')
run('reject_dm_binary_member',source='dm_binary_member',fail='DM particles cannot be binary')
run('bad_mass',source='unequal',SIDM_SIGMA_OVER_M=1,fail='equal-mass')
run('bad_sigma',SIDM_SIGMA_OVER_M=-1,fail='nonnegative')
run('nan_sigma',SIDM_SIGMA_OVER_M='nan',fail='nonnegative')
run('bad_probability',SIDM_PMAX=.9,fail='SIDM_PMAX must')
run('bad_shell',SIDM_SHELL_SIZE=1,fail='SIDM_SHELL_SIZE must')
# Gravity is sourced by all particles, including DM, at the initial snapshot.
a=snapshots(zero)[0]; m=a['m_MSUN']/a['m_MSUN'].sum(); r=a['r']
phi=-np.cumsum(m)/r-(np.cumsum((m/r)[::-1])[::-1]-m/r)+1e-10  # SF_INFINITY=1e10
np.testing.assert_allclose(a['star_phi'],phi,rtol=1e-11,atol=1e-11)
# Existing stellar runs have identical evolution with omitted vs explicit zero SI.
stars=run('stars_default',source='stars')
stars0=run('stars_zero',source='stars',SIDM_SIGMA_OVER_M=0)
assert (stars/'test.dyn.dat').read_bytes()==(stars0/'test.dyn.dat').read_bytes()
strong=run('dm_excluded_strong',THREEBODYBINARIES=1,SS_COLLISION=1,
           BH_CAPTURE=1,BINBIN=1,BINSINGLE=1,MIN_BINARY_HARDNESS=.1)
check_dm(strong,1000)
assert not [l for l in (strong/'test.3bb.log').read_text().splitlines() if l and not l.startswith('#')]
short=run('checkpoint',SIDM_SIGMA_OVER_M=1e6,PERTURB=0)
full=run('continuous',SIDM_SIGMA_OVER_M=1e6,PERTURB=0,T_MAX_COUNT=8)
resumed=run('restart',restart=short,SIDM_SIGMA_OVER_M=1e6,PERTURB=0,T_MAX_COUNT=8)
check_dm(resumed,1000)
xf=np.atleast_2d(np.loadtxt(full/'test.sidm.dat'))
xr=np.atleast_2d(np.loadtxt(resumed/'test.sidm.dat'))
np.testing.assert_allclose(xr,xf[-len(xr):],rtol=1e-12,atol=1e-12)
linear1=run('linear_sigma1',SIDM_SIGMA_OVER_M=1,PERTURB=0,T_MAX_COUNT=2)
linear2=run('linear_sigma2',SIDM_SIGMA_OVER_M=2,PERTURB=0,T_MAX_COUNT=2)
l1=np.atleast_2d(np.loadtxt(linear1/'test.sidm.dat'))[0]
l2=np.atleast_2d(np.loadtxt(linear2/'test.sidm.dat'))[0]
# The limiter can shorten dt when sigma doubles; compare rates, not P.
np.testing.assert_allclose(l2[6:8]/l2[2],2*l1[6:8]/l1[2],rtol=1e-14)
# The same DM trajectories must result with shared or replicated gravity arrays.
replicated=run('replicated',SIDM_SIGMA_OVER_M=1e6,env_changes={'CMC_SHARED_GLOBALS':'0'})
shared_control=run('shared_control',SIDM_SIGMA_OVER_M=1e6,env_changes={'CMC_SHARED_GLOBALS':'1'})
np.testing.assert_array_equal(np.loadtxt(replicated/'test.sidm.dat'),np.loadtxt(shared_control/'test.sidm.dat'))
np.testing.assert_array_equal(snapshots(replicated)[-1]['vr'],snapshots(shared_control)[-1]['vr'])
four=run('four_ranks',np_=4,SIDM_SIGMA_OVER_M=1e6,NUM_CENTRAL_STARS=100)
check_dm(four,1000)
cross=run('cross_rank_pair',source='cross_rank',SIDM_SIGMA_OVER_M=1e10,PERTURB=0,T_MAX_COUNT=20)
check_dm(cross,2)
xc=np.atleast_2d(np.loadtxt(cross/'test.sidm.dat'))
assert xc[0,4]==1 and xc[0,6]>0, 'DM neighbors on different MPI ranks must be paired'
summary={'status':'passed','scenarios':RESULTS,'events':float(x[:,5].sum()),
         'expected_events':float(x[:,6].sum()),'max_relative_sidm_energy_error':float(np.max(np.abs(x[:,8])))}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('SIDM events / expected:',x[:,5].sum(),x[:,6].sum())
print('Maximum relative SI energy error:',np.max(np.abs(x[:,8])))
print('All DM/SIDM tests passed; outputs:',OUT)
