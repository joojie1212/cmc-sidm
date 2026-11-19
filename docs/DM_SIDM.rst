CMC 中的 DM 粒子与 SIDM 散射
==========================

本实现把 DM 作为独立的粒子种类，保留其质量对球对称引力势的贡献，
允许 DM–DM 弹性散射，关闭 DM–DM 和 DM–star 的引力两体弛豫。
star–star 弛豫仍由原有 ``RELAXATION`` 开关控制。
第一版的 SIDM 为速度无关、各向同性的常数截面模型，要求 DM 数值粒子等质量。

输入与运行
----------

在 FITS/HDF5 的 ``CLUS_OBJ_DATA`` 中，将 DM 的 ``k`` 设为 ``-200``
（源码常量 ``CMC_DM_TYPE``），``binind=0``，``Reff=0``。
质量、半径、径向速度和切向速度使用原来的 CMC 初始条件单位。
质量是数值粒子所代表的总质量，不是微观暗物质粒子的质量。
恒星仍使用原来的类型编号。双星中不允许有 DM 成员。

可用示例生成器建立恒星与 DM 具有相同 Plummer 分布的测试模型::

    .venv/bin/python tools/make_dm_plummer.py /tmp/stars_dm.hdf5 \
        --nstar 10000 --ndm 10000 --star-mass 1 \
        --dm-mass-total 10000 --rvir 1

其中质量单位为太阳质量，``rvir`` 单位为 pc。该生成器不是具有独立暗晕
尺度半径的双组分平衡模型；真实暗晕应单独构造与总势一致的分布函数。
仅给旧粒子改类型不会自动增加系统质量。

复制 ``examples/PlummerSphere.ini`` 到新的运行目录，在 ``[cmc]`` 中设置::

    INPUT_FILE = /tmp/stars_dm.hdf5
    SIDM_SIGMA_OVER_M = 1.0
    SIDM_PMAX = 0.05
    SIDM_SHELL_SIZE = 32
    RELAXATION = 1

然后使用新编译的程序::

    cmake --build build -j 4
    mpirun -np 2 /path/to/CMC-COSMIC/build/src/cmc/cmc run.ini result

``SIDM_SIGMA_OVER_M`` 的单位为 cm²/g，默认 0，即有引力但没有 SI 的 DM。
``SIDM_PMAX`` 为每个抽样配对的概率上限，默认 0.05，允许范围为 (0, 0.1]。
``SIDM_SHELL_SIZE`` 为每个散射球壳的目标 DM 数，默认 32，至少为 2。
没有 DM 的旧输入可直接运行，不会消耗 SIDM 随机数。
程序不会自动添加 DM，必须在初始条件中提供它们。

``PERTURB`` 和 ``RELAXATION`` 仅控制原来的恒星动力学步骤，SIDM 独立运行。
因此 ``PERTURB=0`` 仍可用于隔离测试 SIDM；关闭 SI 请使用截面为零。
DM 跳过 SSE/BSE、toy aging、恒星碰撞、潮汐俘获、双星相互作用和三体双星形成。
原有轨道更新、逃逸及外部潮汐处理仍适用于 DM。

SI 如何进入 CMC
---------------

新增实现为 ``src/cmc/cmc_sidm.c``，速度散射核位于
``include/cmc/cmc_sidm_kernel.h``。
每步先确定包含 SIDM 限制的时间步，然后进行 SIDM、恒星动力学、恒星演化，
最后沿用 CMC 的轨道重采样、径向排序和引力势更新。

1. 引力势直接使用 ``star_m`` 和 ``star_r``，其中同时包括恒星和 DM。
   ``potential_calculate`` 不过滤 DM，因此 DM 有自引力，恒星也会感受 DM 的平均势。
2. 引力弛豫配对只要含有 DM 就跳过，弛豫时间步估计也不计入 DM 的质量矩。
   保留原有全粒子配对和总数密度估计，不能在跳过混合配对后再把密度乘一次恒星比例。
   在局部分布平滑且物种混合的近似下，配对抽样本身已提供该比例。
3. SIDM 独立收集所有 MPI rank 的 DM，按半径分球壳，壳内随机打乱并成对抽样。
   球壳内部边界取相邻 DM 半径的中点，最内边界为 0，最外边界外推最后半个间距。
   末尾单粒子并入前一壳；全系统不足两个 DM 时不散射。
4. 每个配对随机选取相对切向方位，得到共同局部坐标系中的三维相对速度。
   使用下面的概率决定是否散射；一个粒子每步最多参加一次事件。
5. 对接受的事件，在质心系抽取均匀方向：cos(theta)=2U-1，phi=2πV。
   保留相对速度大小，转换回两个粒子的速度。
   此等质量弹性核守恒配对的动量与动能，然后更新 ``vr``、``vt``，
   调用 ``set_star_EJ`` 更新 E 和 J。球对称表示只存角动量大小，
   这里的动量守恒指散射核的共同局部坐标系。

概率与单位
----------

设球壳中有 n 个等质量 DM 数值粒子，每个物理质量为 M，球壳体积为 V，
截面质量比为 s=σ/mχ，配对相对速度为 w。单个无序对的事件率为::

    lambda_ij = s M w / V

随机匹配中，指定一对被抽到的概率：偶数 n 时为 1/(n-1)，奇数 n 时为 1/n。
因此对已抽到的配对使用::

    A(n) = n-1  (n 为偶数)
    A(n) = n    (n 为奇数)
    P_pair = A(n) s M w Δt / V

奇数球壳中的轮空粒子由打乱顺序随机决定。这一补偿保证期望事件数等于对
所有无序对求和；粒子的散射率趋近于 rho_DM s <w>。
概率采用一阶 Bernoulli 离散，时间步强制保持小概率，既不把大概率裁剪为 1，
也不把 ``1-exp(-P_pair)`` 当作抽样权重补偿。

实际代码使用的量为::

    M = m_code * units.mstar                       [g]
    V = V_code * units.l^3                         [cm³]
    w = w_code * units.l / units.t                 [cm/s]
    Δt = Dt * units.t * N / ln(GAMMA*N)             [s]

其中 N 是 CMC 初始化的 ``clus.N_STAR``，包括 DM 数值粒子。
CMC 的时间单位不是直接的 ``units.t``；遗漏 N/ln(GAMMA*N) 会改变 SI 物理散射率。
代入后 ``P_pair`` 无量纲。宏观粒子数变化时保持 DM 总质量和 σ/mχ 不变，
而不是把微观粒子的质量代入势或另乘一个微观粒子数。

现有恒星弛豫的库仑对数仍采用 CMC 的 ``ln(GAMMA*N)`` 约定。
改变 DM 数值采样数而保持同一物理恒星系统时，应保持 ``GAMMA*N`` 不变，
避免仅因总数值对象数改变而改变恒星弛豫率。例如以恒星模型参数 GAMMA_star
及初始恒星对象数 N_star 为基准，混合模型可设
``CMC_GAMMA = GAMMA_star * N_star / N``（ini 参数名为 CMC_GAMMA），且需 ``GAMMA*N > 1``。
这也改变 CMC 代码时间的物理换算；SIDM 使用运行时 GAMMA 做一致换算。

每壳用 ``w <= 2*vmax`` 给出配对概率上界，由此计算 ``sidm_timestep``。
在累计 ``TotalTime`` 前，把 CMC 时间步与该上界取最小值。
散射步骤再次检查实际概率，越界会显式报错而不是悄悄截断。

输出与重启
----------

``result.sidm.dat`` 记录::

    tcount time dt N_DM pairs events expected_events max_probability relative_delta_KE

``expected_events`` 为本步所有已抽配对的概率之和，``relative_delta_KE``
只测量 SIDM 速度更新造成的相对动能变化，不代表整个 CMC 演化的总能量误差。
``time``、``dt`` 为 CMC 时间单位。截面为零时不创建该文件。

普通快照中的 ``startype=-200`` 可筛选 DM，``m_MSUN`` 为数值粒子质量。
原有全局质量、核心量等诊断通常包含全部引力粒子，不能一概解释为恒星统计量。
现有 ``se_k`` 与速度字段随排序、MPI 传输及 checkpoint 保存，不改变 ``star_t`` 布局。
SIDM 使用原有可重启随机状态，重启时重新检查 DM 类型与质量。
SIDM 参数仍从 ini 读取，恢复原模型时需保持这些参数不变。
诊断日志在重启后追加一个带表头的新段；从较早 checkpoint 恢复时可能存在重复时间，
分析时应按重启段选择，不能把所有行的事件数直接相加。

适用范围和验证
--------------

这是球壳内分布平滑的 DSMC 基础实现，不含速度依赖、角度依赖、非弹性散射或
不等质量 DM 数值粒子的加权散射。恒星质量可以不同于 DM 数值粒子质量。
MPI 版通过 MPI_Gatherv 只在 rank 0 汇集全局 DM，并用 MPI_Scatterv
把更新后的粒子分回所属进程，避免遗漏进程边界的相互作用。
根进程内存为 O(N_DM)，其他进程为 O(N_DM,local + N_rank)，
不在每个进程复制全体 DM。当前算法仍由根进程执行散射，尚非完全分布式散射，
超大 DM 数量应评估根进程内存与通信开销。
该实现兼容现有节点共享引力数组与 CMC_SHARED_GLOBALS=0 的复制模式。
MPI 本地 HDF5 读取器仅为 DM 接受负类型 -200，其他无效负类型仍会被拒绝。

保留 CMC 的球对称与轨道平均假设。SIDM 时间步限制控制散射概率，不能保证
散射时间短于轨道周期的强碰撞系统仍满足轨道平均假设。
用于物理结论前需检查轨道/散射时间尺度，并对 DM 数量、球壳大小、概率上限做收敛检验。
恒星强相互作用的背景矩估计仍沿用 CMC，DM 占主导时其恒星专属率需进一步验证。

测试命令::

    cc -std=c99 -Iinclude/cmc tests/sidm_kernel.c -lm -o /tmp/test_sidm_kernel
    /tmp/test_sidm_kernel
    .venv/bin/python tests/test_dm_sidm.py
    bash tests/node_shared/run_storage_tests.sh

散射核测试检查配对动量/能量、各向同性、零相对速度及奇偶球壳抽样归一化。
集成测试覆盖 DM 势、零截面、恒星演化隔离、强相互作用隔离、概率/事件率、
MPI 单/双/四进程、共享/复制存储一致性、跨进程 DM 配对、输入校验和 checkpoint 恢复。
