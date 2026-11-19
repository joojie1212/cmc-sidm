#!/usr/bin/env bash
# Requires a configured CMC build, working MPI, Python 3 and Linux /proc.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
work=${1:-$(mktemp -d /tmp/cmc-shared-storage.XXXXXXXX)}
mkdir -p "$work"
work=$(cd "$work" && pwd)
build=${CMC_BUILD_DIR:-$root/build}
hdf5_flags=()
if [[ -n "${HDF5_INCLUDE_DIR:-}" ]]; then
    hdf5_flags=("-I$HDF5_INCLUDE_DIR")
elif command -v h5cc >/dev/null; then
    read -ra wrapper_flags <<< "$(h5cc -show)"
    for flag in "${wrapper_flags[@]}"; do
        if [[ "$flag" == -I* ]]; then hdf5_flags+=("$flag"); fi
    done
elif command -v pkg-config >/dev/null && pkg-config --exists hdf5; then
    read -ra hdf5_flags <<< "$(pkg-config --cflags hdf5)"
else
    echo 'Set HDF5_INCLUDE_DIR or load the HDF5 module providing h5cc.' >&2
    exit 1
fi
"${MPICC:-mpicc}" -O2 -fcommon "${hdf5_flags[@]}" -I"$build" -I"$root/include/common" \
    -I"$root/include/cmc" -I"$root/include/bse_wrap" \
    "$root/tests/node_shared/shared_arrays.c" "$root/src/cmc/cmc_shared_globals.c" \
    -lrt -lm -o "$work/test-arrays"
# Open MPI options match the current local/server test environment.
CMC_SHARED_GLOBALS=0 mpirun --oversubscribe --bind-to none -np 4 \
    "$work/test-arrays" 2000003 >"$work/replicated.log" 2>&1
CMC_SHARED_GLOBALS=1 CMC_SHARED_TEST_GROUPS=1 mpirun --oversubscribe --bind-to none -np 4 \
    "$work/test-arrays" 2000003 >"$work/shared-one-node.log" 2>&1
CMC_SHARED_GLOBALS=1 CMC_SHARED_TEST_GROUPS=2 mpirun --oversubscribe --bind-to none -np 4 \
    "$work/test-arrays" 2000003 >"$work/shared-two-groups.log" 2>&1
python3 - "$work" <<'PY'
from pathlib import Path
import json,re,sys
work=Path(sys.argv[1]); report={}
for name in ['replicated','shared-one-node','shared-two-groups']:
    text=(work/(name+'.log')).read_text()
    assert 'SHARED ARRAY TESTS PASSED' in text
    report[name]=int(re.search(r'sum_pss_kib=(\d+)',text)[1])
assert report['shared-one-node'] < report['replicated']*.7, report
assert report['shared-one-node'] < report['shared-two-groups'] < report['replicated'], report
(work/'memory-results.json').write_text(json.dumps(report,indent=2)+'\n')
print('All storage tests passed. Sum PSS (KiB):',report)
print('Logs:',work)
PY
