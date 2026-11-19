# Node-shared global arrays (Linux)

This extends the rank-local HDF5 input implementation. On Linux, the rebuilt
executable enables shared storage by default; existing INI/input files remain
valid. `CMC_SHARED_GLOBALS=0` selects the previous replicated storage for an A/B
comparison or rollback without rebuilding. Use the same value on every rank.

## Why direct shared writes would be wrong

CMC changes `star_m`, `star_r` and `star_phi` during each rank's local encounters,
stellar evolution, object creation/destruction and orbit calculations. Other
ranks must continue to see their previous snapshot until an existing publication
point. New objects can also temporarily use the same global tail indices on
different ranks. A direct replacement of `malloc` by writable MPI shared windows
would introduce races and alter the physics.

The implementation therefore uses:

- `MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)` to identify node membership;
- one POSIX shared-memory backing object per node, containing all three arrays;
- a private `MAP_PRIVATE` view per rank. Unchanged pages share physical memory;
  writes create private copy-on-write (COW) pages and remain invisible to peers;
- a writable shared view used only by the node leader at synchronization points.

This is Linux MPI-coordinated POSIX shared memory, not an MPI RMA window. Normal
particle computation remains parallel. The existing array indexing and local
mutation sites do not change.

## Publication and potential calculation

At initialization, after sorting, and after the initial stellar-evolution mass
update, each rank stages only its owned slice. A node barrier protects the source
snapshot. `MPI_Gatherv` collects the slices into rank 0's published array, and
`MPI_Bcast` distributes it among node leaders. Node barriers isolate readers
from updates. Counts/displacements need not group a node's ranks contiguously.

Each rank then remaps ONLY the published page range, discarding superseded COW
pages. Values outside the publication range are preserved, including sentinel
values, private new-object slots, and bytes sharing the boundary pages. Publishing
mass does not discard private radius or potential updates.

The original sequential potential loop is executed once per node, on the shared
snapshot. Its scalar outputs are broadcast within the node, and each rank refreshes
the potential range. This uses the original arithmetic order. The existing
potential invocation points occur after mass/radius publication; it is not used
in the middle of unsynchronized local encounters.

This does not distribute the potential across nodes. Every node retains a whole
snapshot. It also does not make stellar/binary structures or output buffers shared.

## Memory and operating requirements

For N=1e8 without primordial binaries, the three global arrays have approximately
2.46 GiB total capacity per node (including the existing 10% capacity margin),
instead of approximately 315 GiB for 128 replicated copies. Private COW pages,
page tables, local input/star/binary arrays, sort/communication buffers, staging,
and checkpoint/page-cache memory are additional. A full production peak is NOT
claimed from this capacity estimate.

The executable checks available POSIX shared memory and reserves backing space
before mapping it. Check on compute nodes:

```bash
df -h /dev/shm
```

At least the logged backing size must be available for each job/node. The backing
also counts against the job's memory limit; shared storage does not bypass Slurm
limits. Insufficient capacity causes an explicit error, not a silent replicated
fallback. Administrators may need to enlarge `/dev/shm`; a deliberate fallback is:

```bash
export CMC_SHARED_GLOBALS=0
```

Backing names are unlinked after all node ranks open them and BEFORE physical
reservation. The kernel releases storage once the last mapping/descriptor closes,
including process termination. A successful run should not leave named multi-GB
objects on the node. MPI groups must share a POSIX `/dev/shm` namespace, as on the
server's normal Linux batch nodes (separately isolated containers need checking).

The implementation requires MPI-3 node communicators and Linux COW semantics.
Non-Linux builds default to replicated mode. It uses no extra Python dependency.

## Build and deployment

The release contains two patches. Apply ONLY the appropriate one:

- `from-local-io.patch`: against the previously delivered rank-local HDF5 version;
- `from-original.patch`: against clean original commit
  `e919b447cd1d5d2b855e1d57c716219c207560fb`, including both improvements.

Back up server code and executable first. From the server repository:

```bash
git apply --check /path/to/node-shared/from-local-io.patch
git apply /path/to/node-shared/from-local-io.patch
# Load the same MPI/HDF5/GSL/compiler environment as before.
cmake -S . -B build-node-shared \
  -DCMAKE_C_COMPILER=mpicc -DCMAKE_Fortran_COMPILER=mpifort \
  -DCMAKE_INSTALL_PREFIX="$PWD/CMC-node-shared" -DFITS=OFF
cmake --build build-node-shared -j 8
cmake --install build-node-shared
```

Use `$HOME/CMC-COSMIC/CMC-node-shared/bin/cmc` in the ACTUAL `mpirun` command in
both batch scripts. Earlier scripts may hardcode `CMC-local-io/bin/cmc`, so merely
exporting `CMC_EXEC` may not select the new executable. Check the launch line.

Shared mode prints a line per node leader, e.g.:

```text
rank 0: node-shared globals: 64 ranks, ... MiB backing total; private COW updates
```

Start with 2 nodes x 64 ranks and the existing 480 GiB/node request. Do not assume
one-node 128-rank production is safe until measuring complete timestep/checkpoint
peaks. No server job was submitted as part of this implementation.

Checkpoint layout is unchanged from the prior local-I/O version. Soft/hard
restart still requires the same number of MPI ranks and a compatible ABI; ranks
may be placed on different nodes. Changing `CMC_SHARED_GLOBALS` on restart is
supported. Global arrays are reconstructed from rank-local checkpoint states.
Legacy original checkpoint support and restrictions in `MPI_LOCAL_IO.md` remain.

## Verification and limits

Before editing, the previous source, build and installed program were saved at
the path recorded in `local-backups/LATEST_SHARED_BACKUP.txt`; the archive hash
and previous executable are included. Existing scientific run results were not
changed.

The integration suite in `tests/mpi_local_io/regression.py` was run against the
previous local-I/O executable in three configurations: shared single-node,
shared emulated two-node groups, and replicated fallback. It compares the same
single/binary initial conditions, encounters, stellar evolution, global outputs,
snapshots (excluding the previously documented stock eta issue), concentrated
primordial binaries, central-mass sentinel, remainder partitions, soft/hard
restart, and checkpoint error handling.

Emulated groups use `CMC_SHARED_TEST_GROUPS=2`: ranks 0,2,... and 1,3,... become
two separate sharing groups on one host, exercising noncontiguous ownership and
the node-leader communication path. Do NOT set this testing variable in normal
production. This is NOT a real multi-host performance or filesystem test.

`tests/node_shared/shared_arrays.c` independently checks private writes to the
same offsets, owner publication, boundary-page preservation, potential/scalar
publication, shrinking partitions and tail values. The reproducible runner is:

```bash
bash tests/node_shared/run_storage_tests.sh /tmp/cmc-storage-test
```

It uses `/proc/self/smaps_rollup` PSS summed over ranks, so shared pages are not
mistakenly counted once per RSS mapping. Results for 4 ranks and N=2,000,003 are
saved in `tests/node_shared/validation/`. These are storage-harness measurements,
not whole-CMC peak-memory claims. Real multi-host and N=1e8 tests remain to be
performed on the server.

Potential computation is no longer repeated per rank, but publication adds
leader coordination and page remapping. NUMA placement and root-gather throughput
can affect speed. No production speedup is promised; compare wall time per step
and node/cgroup memory usage with otherwise identical settings.
