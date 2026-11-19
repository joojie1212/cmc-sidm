# CMC SIDM

基于 [CMC-COSMIC](https://github.com/ClusterMonteCarlo/CMC-COSMIC) 的独立研究版本：
保留当前 MPI 本地 HDF5 输入、节点共享引力数组，并加入常数截面的 SIDM。
本仓库不包含 `NO_BH_BINARIES` 扩展；恒星与 BH 使用原有 CMC 物理逻辑。

- `main`：MPI 改动 + SIDM，用于后续开发。
- `mpi-base`：加入 SIDM 前的 MPI 基线，用于对照。
- `origin`：`git@github.com:joojie1212/cmc-sidm.git`。
- `upstream`：原始 `ClusterMonteCarlo/CMC-COSMIC`。

这个目录有独立的 Git、源码和编译目录。原来的 `CMC-COSMIC` 目录无需修改。
上游提交历史、已有 cmc-sidm 初始化提交和 COSMIC/METISSE 子模块版本均保留。

## 编译

需要 C/C++/Fortran 编译器、MPI、CMake、HDF5、GSL 和 zlib。
在 Ubuntu 上可安装 `build-essential gfortran cmake libopenmpi-dev openmpi-bin
libhdf5-dev libgsl-dev zlib1g-dev`。

```bash
git clone --recurse-submodules git@github.com:joojie1212/cmc-sidm.git
cd cmc-sidm
cmake -S . -B build \
  -DCMAKE_C_COMPILER=mpicc -DCMAKE_Fortran_COMPILER=mpifort \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD/install" -DFITS=OFF
cmake --build build -j 4
cmake --install build
```

可执行文件为 `build/src/cmc/cmc` 或 `install/bin/cmc`。
Python 仅用于初始条件生成和集成测试，不是运行 CMC 二进制的依赖。
需要这些工具时，可创建独立环境：

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ./src/bse_wrap/bse/COSMIC
```

## SIDM 用法

DM 输入类型为 `k=-200`，要求单粒子 `binind=0`，不参加引力弛豫、恒星演化、
恒星碰撞或双星形成。它们参与总引力势、轨道演化和 DM–DM 自散射。
第一版支持**等质量 DM 数值粒子、弹性各向同性、速度无关的截面**。

在运行 ini 的 `[cmc]` 段设置：

```ini
SIDM_SIGMA_OVER_M = 1.0
SIDM_PMAX = 0.05
SIDM_SHELL_SIZE = 32
```

截面单位为 cm²/g；默认 `SIDM_SIGMA_OVER_M=0`，即 DM 仅有平均引力作用。
`RELAXATION` 控制恒星弛豫，SIDM 独立于 `RELAXATION` 和 `PERTURB` 开关。
初始条件中必须有 DM，设置截面本身不会自动添加粒子。

```bash
mkdir -p runs/demo
python tools/make_dm_plummer.py runs/demo/stars_dm.hdf5 \
  --nstar 10000 --ndm 10000 --star-mass 1 --dm-mass-total 10000 --rvir 1
cp examples/PlummerSphere.ini runs/demo/run.ini
```

编辑 `runs/demo/run.ini`，将 `INPUT_FILE` 设为 `stars_dm.hdf5`，并设置截面及运行时长。
然后从运行目录启动：

```bash
cd runs/demo
mpirun -np 2 ../../build/src/cmc/cmc run.ini result
```

这只是恒星和 DM 共用 Plummer 分布的示例。独立暗晕尺度半径需要另行构造平衡初始条件。
散射记录位于 `result.sidm.dat`，普通快照以 `startype=-200` 标识 DM。

## MPI 与物理限制

- Linux 默认使用节点共享引力快照；`CMC_SHARED_GLOBALS=0` 切换为原来的复制模式。
- SIDM 只在根进程汇集全体 DM，其他进程仅保留自己的 DM 切片。
  散射仍由根进程执行，大模型需要评估该进程的内存和通信开销。
- 保留 CMC 球对称、轨道平均假设。高碰撞率下需检查其适用性。
- 改变 DM 数值粒子数量时，恒星弛豫的库仑对数及物理时间单位也需保持一致；
  详见 [SIDM 的算法、单位换算与验证说明](docs/DM_SIDM.rst)。

MPI 实现见 [本地输入](docs/MPI_LOCAL_IO.md) 与 [节点共享内存](docs/MPI_NODE_SHARED.md)。

## 测试

```bash
cc -std=c99 -Wall -Wextra -Iinclude/cmc tests/sidm_kernel.c -lm -o /tmp/test_sidm_kernel
/tmp/test_sidm_kernel
python tests/test_dm_sidm.py
bash tests/node_shared/run_storage_tests.sh
```

集成测试需要安装本仓库的 COSMIC 子模块及其 Python 依赖，并允许 MPI 使用本地 socket。
测试在临时目录生成数据。`CMC_TEST_EXECUTABLE` 可指定另一个待测 CMC 可执行文件。

与 MPI 基线比较恒星/双星行为：

```bash
python tests/mpi_local_io/regression.py \
  --baseline /path/to/mpi-baseline/cmc \
  --cmc build/src/cmc/cmc --work /tmp/cmc-mpi-comparison
```

GitHub Actions 仅手动触发：在仓库的 Actions 页面选择 `CMC SIDM build and tests`，
点击 `Run workflow`。普通提交、推送和 PR 不会自动启动云端测试。
本地编译和上述测试命令照常可用。

## 日常更新

只在新的 `cmc-sidm` 目录修改、编译和提交。保存文件不会自动上传；通过提交和推送同步：

```bash
git status
git add <修改的源码或文档>
git commit -m "说明本次修改"
git push origin main
```

较大改动建议先在功能分支测试后合并到 `main`。同步原 CMC 时先添加 upstream（新克隆需要）：

```bash
git remote add upstream https://github.com/ClusterMonteCarlo/CMC-COSMIC.git
git fetch upstream
git switch -c sync-upstream
git merge upstream/master
# 解决冲突、编译和测试后，再合并到 main。
```

`runs/`、`analysis/`、备份、安装目录和构建产物不纳入版本控制。
COSMIC/METISSE 通过固定提交的子模块管理；迁移到新机器请使用 `--recurse-submodules`。

## 来源与引用

原始 CMC 基线为 `e919b447cd1d5d2b855e1d57c716219c207560fb`。
MPI 基线来自 2026-11-19 的本地 CMC 工作版本，SIDM 从保留的实现中迁移并适配该 MPI 层。
本项目是研究扩展，不替代上游 CMC/COSMIC 的引用要求。
请参阅 [上游介绍及论文引用](docs/UPSTREAM_README.md#citing-the-code)。
