#!/usr/bin/env python3
"""Create a co-spatial star + DM Plummer initial condition for CMC.

Run with the repository venv. Masses are solar masses; radii are parsecs.
This is a common-profile demonstration, not a separately fitted dark halo.
"""
import argparse
from pathlib import Path
import numpy as np
from cosmic.sample import InitialCMCTable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--nstar', type=int, default=10000)
    parser.add_argument('--ndm', type=int, default=10000)
    parser.add_argument('--star-mass', type=float, default=1.)
    parser.add_argument('--dm-mass-total', type=float, default=10000.)
    parser.add_argument('--rvir', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; choose a new filename')
    if args.nstar < 0 or args.ndm < 1 or args.nstar+args.ndm < 100:
        parser.error('nstar >= 0, ndm >= 1, and at least 100 total particles required')
    if not all(np.isfinite(v) and v > 0 for v in [args.star_mass,args.dm_mass_total,args.rvir]):
        parser.error('masses and rvir must be positive and finite')
    s,b = InitialCMCTable.sampler('cmc_point_mass', cluster_profile='plummer',
        size=args.nstar+args.ndm, r_max=100, seed=args.seed,
        virial_radius=args.rvir, tidal_radius=1e6)
    s['m']=args.star_mass; s['k']=1; s['Reff']=0.
    # Random labels independently of radius; all DM have the same macro mass.
    dm=np.random.default_rng(args.seed).choice(len(s),size=args.ndm,replace=False)
    s.iloc[dm,s.columns.get_loc('m')]=args.dm_mass_total/args.ndm
    s.iloc[dm,s.columns.get_loc('k')]=-200
    s.mass_of_cluster=None; s.scaled_to_nbody_units=False
    InitialCMCTable.write(s,b,filename=str(args.output))
    print(f'Wrote {args.output}: {args.nstar} stars + {args.ndm} DM; '
          f'DM macro mass={args.dm_mass_total/args.ndm:g} Msun')


if __name__ == '__main__':
    main()
