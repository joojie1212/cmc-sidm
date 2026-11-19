#!/usr/bin/env python3
"""End-to-end tests against an unmodified executable built at the same revision.
Run with a Python containing COSMIC, h5py, pandas and numpy, plus working MPI.
"""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import json
import numpy as np
import pandas as pd
import h5py
from cosmic.sample import InitialCMCTable

p = argparse.ArgumentParser()
p.add_argument('--baseline', type=Path, required=True)
p.add_argument('--cmc', type=Path, required=True)
p.add_argument('--work', type=Path, required=True)
p.add_argument('--section', choices=['all','core','extra'], default='all')
a = p.parse_args()
a.baseline = a.baseline.resolve(); a.cmc = a.cmc.resolve(); a.work = a.work.resolve()
a.work.mkdir(parents=True, exist_ok=True)
assert not any(a.work.iterdir()), 'Use an empty test directory'
root = Path(__file__).resolve().parents[2]
template = (root/'examples/PlummerSphere.ini').read_text()
results = []

def fixture(name, n=10240, nb=0, concentrated=False):
    out = a.work/(name+'.hdf5')
    s,b = InitialCMCTable.sampler('cmc_point_mass', cluster_profile='plummer',
        size=n, r_max=100, seed=12345, virial_radius=1.6976527263135504)
    InitialCMCTable.write(s,b.iloc[:0].copy(), filename=str(out),
        virial_radius=1.6976527263135504, tidal_radius=1e6, metallicity=.02,
        central_bh=0, scale_with_central_bh=False)
    if nb:
        s = pd.read_hdf(out,'CLUS_OBJ_DATA')
        b = pd.read_hdf(out,'CLUS_BINARY_DATA')
        rows = np.arange(1,nb+1) if concentrated else np.linspace(1,n,nb,dtype=int)
        refs = np.random.default_rng(12).permutation(nb)+1
        data = np.zeros((nb+1,11))
        for obj, ref in zip(rows, refs):
            s.iloc[obj,s.columns.get_loc('binind')] = ref
            x = s.iloc[obj]
            data[ref] = [ref,x['id'],x['k'],x['m']/2,0,n+ref,x['k'],x['m']/2,0,1e-5,.3]
        with h5py.File(out,'r') as f:
            attrs = dict(f['CLUS_OBJ_DATA/block0_values'].attrs)
        s.to_hdf(out,key='CLUS_OBJ_DATA',mode='w')
        pd.DataFrame(data,columns=b.columns).to_hdf(out,key='CLUS_BINARY_DATA',mode='a')
        with h5py.File(out,'a') as f:
            ds = f['CLUS_OBJ_DATA/block0_values']
            for k in ('EXTNAME','NOBJ','MCLUS','RVIR','RTID','Z'): ds.attrs[k]=attrs[k]
            ds.attrs['NBINARY']=nb
    return out

def config(path, inputfile, steps=4, **extra):
    opts=dict(INPUT_FILE=str(inputfile), CMC_GAMMA=.01, T_MAX=100,
        T_MAX_PHYS=1e9, T_MAX_COUNT=steps, MAX_WCLOCK_TIME=120,
        CHECKPOINT_INTERVAL=10000, CHECKPOINTS_TO_KEEP=2, STOPATCORECOLLAPSE=0,
        SNAPSHOTTING=0, SNAPSHOT_CORE_COLLAPSE=0, SNAPSHOT_WINDOWS='NULL',
        BH_SNAPSHOTTING=0, TIDALLY_STRIP_STARS=0, STELLAR_EVOLUTION=0,
        THREEBODYBINARIES=0, ONLY_FORM_BH_THREEBODYBINARIES=0,
        BINBIN=0,BINSINGLE=0,SS_COLLISION=0,TIDAL_CAPTURE=0,TC_POLYTROPE=0,
        BH_CAPTURE=0,SAMPLESIZE=32)
    opts.update(extra); text=re.sub(r'^SIDM_\w+\s*=.*\n', '', template, flags=re.M)
    for k,v in opts.items():
        text,count=re.subn(r'^'+k+r'\s*=.*$',f'{k} = {v}',text,flags=re.M)
        if not count: text=text.replace('[cmc]',f'[cmc]\n{k} = {v}',1)
    path.write_text(text)

def run(name, exe, ranks, inputfile, steps=4, restart=None, soft=False, fail=False, **opts):
    out=a.work/name; out.mkdir(exist_ok=True)
    config(out/'run.ini',inputfile,steps,**opts)
    prefix='model'
    args=[]
    if restart:
        parent=a.work/restart
        shutil.copytree(parent/'model-RESTART',out/'model-RESTART')
        if soft:
            for f in parent.glob('model.*'):
                if f.is_file(): shutil.copy2(f,out/f.name)
            args=['-r','1']; prefix='model'
        else:
            args=['-R','1']; prefix='resumed'
    cmd=['mpirun','--oversubscribe','--bind-to','none','-np',str(ranks),str(exe),*args,'run.ini',prefix]
    if restart and not soft: cmd.append('model')
    with (out/'console.log').open('w') as log:
        done=subprocess.run(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,timeout=120,
            env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    if fail:
        assert done.returncode != 0, name
    elif done.returncode:
        raise AssertionError(f'{name}: exit {done.returncode}; inspect {out}/console.log')
    results.append(dict(name=name,returncode=done.returncode))
    print(name, 'passed',flush=True)
    return out,prefix

def compare(x,y):
    for suffix in ['dyn.dat','lagrad.dat','bin.dat']:
        left=np.loadtxt(x[0]/(x[1]+'.'+suffix));right=np.loadtxt(y[0]/(y[1]+'.'+suffix))
        np.testing.assert_allclose(left,right,rtol=1e-12,atol=1e-13,err_msg=suffix)

if a.section in ('all','core'):
    single=fixture('single')
    binary=fixture('primordial',nb=43)
    for fixture_name,ic in [('single',single),('binary',binary)]:
        for ranks in [1,2,4]:
            old=run(f'{fixture_name}_old_{ranks}',a.baseline,ranks,ic)
            new=run(f'{fixture_name}_new_{ranks}',a.cmc,ranks,ic)
            compare(old,new)
    # Real encounters and stellar evolution also consume primordial binary fields.
    old=run('encounters_old',a.baseline,2,binary,BINBIN=1,BINSINGLE=1,THREEBODYBINARIES=1)
    new=run('encounters_new',a.cmc,2,binary,BINBIN=1,BINSINGLE=1,THREEBODYBINARIES=1)
    compare(old,new)
    # Compare restarts using the same original checkpoint; then new checkpoints.
    for kind,ic in [('single',single),('binary',binary)]:
        old=run(kind+'_restart_old',a.baseline,2,ic,steps=6,restart=kind+'_old_2')
        new=run(kind+'_restart_legacy',a.cmc,2,ic,steps=6,restart=kind+'_old_2')
        compare(old,new)
        local=run(kind+'_restart_local',a.cmc,2,ic,steps=6,restart=kind+'_new_2')
        compare(old,local)
        oldsoft=run(kind+'_soft_old',a.baseline,2,ic,steps=6,restart=kind+'_old_2',soft=True)
        newsoft=run(kind+'_soft_new',a.cmc,2,ic,steps=6,restart=kind+'_new_2',soft=True)
        compare(oldsoft,newsoft)
    remainder=fixture('remainder', n=10016, nb=17)
    run('remainder_new',a.cmc,4,remainder)
    # Force nonuniform local binary capacity growth (old loader would overflow).
    concentrated=fixture('concentrated',n=12000,nb=1490,concentrated=True)
    run('concentrated_new',a.cmc,8,concentrated,steps=2)
    run('concentrated_restart',a.cmc,8,concentrated,steps=4,restart='concentrated_new')
    # Reject a changed process count before interpreting checkpoint particle arrays.
    out,_=run('wrong_ranks',a.cmc,4,binary,steps=6,restart='binary_new_2',fail=True)
    assert 'checkpoint:' in (out/'console.log').read_text()
    # Restart must need only input metadata: remove particle datasets but keep attributes.
    header_only=a.work/'header_only.hdf5'
    with h5py.File(binary,'r') as src,h5py.File(header_only,'w') as dst:
        ds=dst.create_dataset('CLUS_OBJ_DATA/block0_values',shape=(0,8),dtype='f8')
        for key in ('NOBJ','NBINARY','MCLUS','RVIR','RTID','Z'):
            ds.attrs[key]=src['CLUS_OBJ_DATA/block0_values'].attrs[key]
    run('header_only_restart',a.cmc,2,header_only,steps=6,restart='binary_new_2')
if a.section in ('all','extra'):
    ic=fixture('extra_primordial',nb=31)
    old=run('snapshot_old',a.baseline,2,ic,SNAPSHOTTING=1)
    new=run('snapshot_new',a.cmc,2,ic,SNAPSHOTTING=1)
    compare(old,new)
    # Compare actual object/binary snapshot fields, not just bulk diagnostics.
    with h5py.File(old[0]/'model.snapshots.h5','r') as x, h5py.File(new[0]/'model.snapshots.h5','r') as y:
        names=[]
        x.visititems(lambda name,obj: names.append(name) if isinstance(obj,h5py.Dataset) else None)
        assert names
        for name in names:
            left=x[name][:];right=y[name][:]
            assert left.dtype==right.dtype and left.shape==right.shape
            if left.dtype.names:
                for field in left.dtype.names:
                    # Stock snapshot eta calls calc_average_mass_sqr with local
                    # indices across rank boundaries, reading unused global slots.
                    # It is an existing diagnostic bug, not an input-state field.
                    if field == 'eta':
                        continue
                    np.testing.assert_allclose(left[field],right[field],rtol=1e-12,atol=1e-13,err_msg=name+field)
            else:
                np.testing.assert_allclose(left,right,rtol=1e-12,atol=1e-13,err_msg=name)
    old=run('evolution_old',a.baseline,2,ic,STELLAR_EVOLUTION=1)
    new=run('evolution_new',a.cmc,2,ic,STELLAR_EVOLUTION=1)
    compare(old,new)
    # Corruption must fail explicitly rather than silently reading partial arrays.
    broken=a.work/'broken'
    broken.mkdir()
    shutil.copytree(new[0]/'model-RESTART',broken/'model-RESTART')
    part=broken/'model-RESTART/model.restart.1-0.bin'
    with part.open('r+b') as f: f.truncate(part.stat().st_size-64)
    out,_=run('truncated_checkpoint',a.cmc,2,ic,steps=6,restart='broken',fail=True)
    assert 'checkpoint: size mismatch' in (out/'console.log').read_text()
    # Central mass is carried by the input sentinel, outside all local row ranges.
    central=fixture('central')
    with h5py.File(central,'a') as f:
        f['CLUS_OBJ_DATA/block0_values'][0,2]=0.01
    old=run('central_old',a.baseline,2,central)
    new=run('central_new',a.cmc,2,central)
    compare(old,new)
(a.work/'results.json').write_text(json.dumps(results,indent=2)+'\n')
print('ALL REGRESSIONS PASSED',flush=True)
