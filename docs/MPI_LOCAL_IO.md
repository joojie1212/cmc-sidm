# MPI rank-local HDF5 initialization

This describes the input/restart layer. The current executable additionally uses
node-shared global snapshots by default on Linux; see `MPI_NODE_SHARED.md` for
storage semantics, memory accounting and deployment of the combined version.

This change is based on original CMC commit
`e919b447cd1d5d2b855e1d57c716219c207560fb`, with the previously added
DM/SIDM and NO_BH_BINARIES extensions removed from the active source tree.
The original physics algorithms are retained. The current HDF5 input format
and INI files do not need conversion or a new runtime flag.

## Backup and rollback

`local-backups/LATEST_MPI_BACKUP.txt` identifies the pre-change backup.
It contains `source-and-build.tar.gz`, its SHA-256, `changes.patch`, the
original HEAD and status, and the retired extension files. The archive includes
the entire working source, nested COSMIC source, tests, build artifacts and
installed `CMC/`. Existing `runs/` (about 35 GB) and `analysis/` remain untouched
in place; environments, credentials, Git metadata and older backups are not
included. The archive was read back and checked before restoring tracked
source files to HEAD. Restoring the archive recovers the previous code/binaries;
remove newly added local-I/O files separately for an exact tree rollback.

The untouched original baseline executable used for comparison is also saved
in the backup as `original-baseline-cmc`. It is distinct from the pre-change
installed executable, which contained the old extensions.

## Implementation

- `parser()` reads only `NOBJ`, `NBINARY`, `MCLUS`, `RVIR`, `RTID` and `Z`.
- The existing `findLimits()` determines ownership, including `MIN_CHUNK_SIZE`
  and remainder handling. Each rank then independently opens the same HDF5
  file read-only and reads its own rows through a hyperslab.
- Temporary reads are bounded to 4096 rows. Local input arrays have size
  `N_local + 2`, rather than `N_global + 2`. Sentinel rows are read separately
  to preserve central-mass and outer-boundary semantics.
- Primordial binary references are taken from local object `binind` values.
  References are sorted for I/O and contiguous runs are read in batches; binary
  records are placed in local slots in object traversal order. Thus binary table
  order does not have to match object ownership or object order. Global metadata
  `NBINARY` is retained; local `binind` values refer to compact local binary rows.
- Initial binary/star capacities grow if a nonuniform partition requires it.
  Initial binary rows not referenced on this rank are never loaded here.
- `star_m` and `star_r` are populated from local objects and synchronized with
  `MPI_Allgatherv`. `star_phi` is still computed by the original code.
- Local input arrays are freed immediately after transfer to runtime structures.
- HDF5 failures, invalid shapes and invalid local references abort the MPI job
  explicitly. The existing code uses `int` MPI counts/indices; overly large
  headers beyond that supported range are rejected.

Serial HDF5 is sufficient; MPI-enabled HDF5 is not required. Independent reads
can still stress a shared filesystem. This change does not convert the CMC
potential calculation or global arrays to distributed storage.

For N=1e8, the old HDF5 reader temporarily holds about 13.2 decimal GB per rank
(input arrays, complete table buffer, row pointers). The new input object arrays
are approximately `60 * (N_local+2)` bytes plus local binary arrays, references
and a roughly 360 KiB row buffer. At 128 balanced ranks, object input arrays are
about 47 MB/rank. These are allocation estimates, not measured whole-job peaks.
With replicated mode selected, the global arrays, runtime stars/binaries, MPI buffers and checkpoints
still require memory. A full 1e8 production run has NOT been tested locally.

## Checkpoints

Both `-r` (soft restart) and `-R` (hard restart) continue to work. Each rank reads
only its own checkpoint. The initial HDF5 is opened for physical metadata only;
its particle/binary datasets are not read on restart. Keep the original input
file available at the configured `INPUT_FILE` path.

New checkpoints append a versioned trailer with rank count, rank number, local
star/binary capacities, snapshot-window count and structure sizes. The payload
physics structures retain their original layout. Readers validate file length,
short reads and basic layout before allocating/loading the arrays. New checkpoint
writes also check short writes and close failures.

- Use the SAME MPI process count to restart; changing node placement is fine.
  Redistributing checkpoints across a different process count is NOT implemented.
- Legacy original-version checkpoints without the trailer are accepted with the
  original inferred capacities and a strict file-size check. Keep the same MPI
  count, original input metadata, snapshot windows and compatible compiler/ABI.
- This is NOT a portable checkpoint format. Equal structure sizes alone cannot
  prove ABI compatibility. Checkpoints made by the retired locally modified
  DM/NO_BH code are not promised compatible.
- Do not resume a capacity-expanded new checkpoint with an old executable.
- Legacy FITS input remains on the existing full-read path. The memory optimization
  applies to HDF5, which is used by the server run packages.

Example (run in the directory holding `plummer-RESTART/`):

```bash
# Continue checkpoint 3 and append to the same output prefix:
mpirun -np 128 /path/to/new/cmc -r 3 run.ini plummer
# Or use a new output prefix, retaining the original checkpoint prefix:
mpirun -np 128 /path/to/new/cmc -R 3 run.ini resumed plummer
```

The existing fresh-run batch scripts reject existing outputs and do not select
checkpoint numbers automatically. Replace their launch command intentionally
when preparing a restart job; do not delete checkpoint files to bypass guards.

## Build and server deployment

Use the C compiler, Fortran compiler, MPI, HDF5 and GSL modules used for the
original executable. The patch package contains no machine-specific binary.
Apply its patch to a clean, matching original checkout after backing up server
source and executable. `git apply --check mpi-local-io.patch` must pass before
applying; do not force a patch onto a different revision.

Build/install into a separate prefix so the server original remains available:

```bash
cmake -S . -B build-local-io \
  -DCMAKE_C_COMPILER=mpicc -DCMAKE_Fortran_COMPILER=mpifort \
  -DCMAKE_INSTALL_PREFIX="$PWD/CMC-local-io" -DFITS=OFF
cmake --build build-local-io -j 8
cmake --install build-local-io
```

Point the server batch script at `$HOME/CMC-COSMIC/CMC-local-io/bin/cmc` (or the
actual install path). Check environment scripts as well: some explicitly reset
`CMC_EXEC` to the old `CMC/bin/cmc`. Loading the environment does not recompile or
replace the executable. Test a small model before the 1e8 production run.

The local workspace `CMC/bin/cmc` has been rebuilt/installed with this change;
its old binary is preserved in the backup archive. No server was accessed,
modified or submitted to from this session.

## Validation

`tests/mpi_local_io/regression.py` generates reproducible inputs and compares
against an executable built from the unchanged original HEAD. Run with a Python
containing COSMIC, NumPy, pandas and h5py, and with MPI available:

```bash
python tests/mpi_local_io/regression.py \
  --baseline /path/to/original-cmc --cmc /path/to/new-cmc \
  --work /empty/test/directory
```

Validated cases include:

- Equal-mass single objects and scrambled primordial binary table references,
  with 1, 2 and 4 ranks; `.dyn.dat`, `.lagrad.dat`, `.bin.dat` agree within
  rtol=1e-12/atol=1e-13.
- Binary encounters/three-body formation enabled, and stellar evolution enabled,
  compared against the original executable.
- Soft/hard restarts from original and new checkpoints, compared numerically.
- 8 ranks with almost all primordial binaries on rank 0, forcing local binary
  capacity growth, followed by restart.
- Non-even object partitions and central-mass sentinel preservation.
- HDF5 containing only header metadata used successfully on checkpoint restart.
- Wrong MPI count and truncated checkpoint rejected.
- Snapshot particle/binary fields compared, excluding the stock `eta` diagnostic.

Two original-code issues surfaced during tests and are recorded rather than
silently broadening the physics changes: the original initialization can fail
in `get_global_idx()` for certain remainder partitions (new local initialization
avoids that conversion), and snapshot `eta` calls `calc_average_mass_sqr()` with
indices that can access unused global-array slots across rank boundaries. The
latter diagnostic is excluded from snapshot comparisons; all other fields are
checked. Production snapshots in the existing server package remain disabled.
Compiler warnings from original Fewbody string-buffer code remain unchanged.

Machine-readable local run reports are in `tests/mpi_local_io/validation/`.
