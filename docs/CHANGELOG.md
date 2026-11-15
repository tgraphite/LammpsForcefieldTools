# Changelog

本文件记录 LammpsForcefieldTools 的版本变更。格式参考 Keep a Changelog，版本号遵循语义化版本。

## [1.1.0] - 2026-11-15

首个带自动更新提示的公开版本；同时包含 0.2.0 的参数库、预优化与 LAMMPS/GROMACS 输出改进。

### Added

- `lft` 在交互终端启动时读取缓存，发现新版本就在 stderr 打印一行提示；缓存过期后在后台线程检查，
  单源超时 1.5 秒，不阻塞管线、不影响退出码。
- 多源检查：PyPI JSON、`PIP_INDEX_URL`、`LFT_UPDATE_URL`（JSON）与 `LFT_UPDATE_INDEX`
  （PEP 503 简单索引），取所有源中的最高版本。
- 缓存位于 `$XDG_CACHE_HOME/lft/update.json`，同一提示默认 24 小时内只打印一次。
- 新增 `lft --check-update`（显式同步检查）与 `lft --no-update-check`（单次跳过）。
- 关闭方式：`LFT_NO_UPDATE_CHECK=1`、`LFT_UPDATE_CHECK=0` 或 `lft.config` 中 `update_check = off`；
  非交互终端、CI、`--version`、`--help` 自动跳过。
- 单元测试覆盖 JSON/简单索引解析、多源取最高、缓存提示去重、超时/离线、禁用开关与 CLI 参数。

### Changed

- 版本号 0.1.0 → 1.1.0（包含此前未单独发布的 0.2.0 内容）；`pyproject.toml`、`lft.__version__` 与文档同步。
- 更新检查只读取版本号，不下载、不执行远程内容；升级仍由用户通过 pip/uv 完成。

### Validation

- `pytest` 全量回归 **501 passed**（含新增更新检查与文档一致性测试）。

## [0.2.0] - 2026-11-15

本轮聚焦"小分子稳定出品、有依据"：参数库离线化、缺失项显式化、几何预优化、DFT 电荷分层，
并把测试集收敛为可 1:1 对照的非周期分子。

### Added

- 内置 GAFF 1.81 参数库（`src/lft/forcefields/data/gaff_bonded.json`）：1085 键、5748 角、
  845 个二面角类型（含多项）、109 improper、249 LJ 类型；装 wheel 即用，不再依赖 `references/`。
- 几何预优化 `--preopt {auto,none,uff,mmff94,dft}`：默认 UFF；`mmff94` 电荷用 MMFF94 优化，
  `dft-*` 电荷用 pySCF/geometric；短程 steepest-descent + conjugate-gradient。
- 电荷新方法 `dft-resp`、`dft-resp2`、`dft-mk`、`dft-cm5`、`dft-raw_cm5`、`dft-adch`：
  先 DFT 优化几何，再算同一电荷模型。
- 电荷规模护栏：DFT 几何优化 ≤50 原子、DFT 单点电荷 ≤200、AM1-BCC ≤500，超过时给出明确错误。
- 多项二面角完整贯通：数据模型、LAMMPS `fourier` 的 `m K n d ...`、GROMACS 多行 functype 9、
  moltemplate coeff。
- 缺失项策略：proper 二面角缺参默认报错（`--allow-missing-torsions` 可省略；线性中心自动省略）；
  improper 缺参默认不构建（`--generic-improper amber|k10` 可选）。
- `--current-geometry`：平衡键长/键角取当前几何，保留库力常数；使用时打印 `INFO_CURRENT_GEOMETRY`。
- LAMMPS 类型去重：按 (类型元组, 参数) 合并，金刚石 216 原子从 432/1296/3888 个类型降到 1/1/1。
- 周期 LAMMPS 默认 PPPM（`lj/cut/coul/long` + `kspace_style pppm 1.0e-4`，kspace 在 `read_data` 之后）；
  非周期仍用 10 Å 直接截断。
- 正交晶胞不再写全零 tilt 行，避免被 LAMMPS 误判为 triclinic。
- 非周期 1:1 验证套件：22 个分子 + AmberTools atomtype 冻结类型参考（`tests/fixtures/validation/`）。

### Changed

- 默认 LAMMPS 产物改为 moltemplate（`<name>.lt` + `system.lt`）；`--lammpsdata` 写 `.data` + `.in`。
- 自动电荷分层改为 ≤50 `resp`、51–200 `am1-bcc`、>200 `mmff94`。
- Hessian 键投影改为完整 3×3 块（含 `H_ij`），并用经验因子标定到 GAFF 量级；角/二面角不再伪装成 Hessian 来源。
- 参数匹配改为索引 + 记忆化；18000 原子参数化从 142 s 降到 0.8 s。
- 文档同步：手册、README、模块现状、变更日志。

### Removed

- 周期/晶体测试对象（`test_periodic_reference.py`、periodic GT manifests、GT-35..41/43/44 目录）：
  暂无可用的 1:1 数值参考。

### Validation

- 将 `geometric` 纳入核心依赖，保证纯 wheel 安装下 `dft-*` 与 `--preopt dft` 可直接使用。
- `pytest` 全量回归 **487 passed**。
- 1:1 类型对照：22/22 分子与 AmberTools26 `atomtype` 完全一致。
- 1:1 参数对照：22 个分子的键/二面角 100% 命中 GAFF 库；仅 caffeine（2）与 formaldehyde（1）的角使用通用值。
- 跨 writer 对照：同一分子的 LAMMPS 与 GROMACS 键/角/二面角系数在单位换算后逐项一致。
- 性能：900 原子 MMFF94 电荷 1.5 s（含 MMFF94 预优化 21 s）；92 原子 AM1-BCC 4.8 s、
  152 原子 11.5 s；18000 原子拓扑 9 s；水分子 `dft-resp` 5.3 s、乙醇 25 s。
- moltemplate 与 `.data` 两条路径的周期 PPPM 总能量逐位一致（0.11770851 kcal/mol）。

### Known limitations

- UFF 键合项仍为规则占位值，仅作 GAFF 不支持元素的后备。
- `dft-*` 电荷的 DFT 优化成本高（20+ 原子常需十分钟级），默认上限 50 原子。
- 周期体系仍只接受显式电荷，暂不参与 1:1 测试。

---

## [0.1.0] - 2026-11-15

首个公开版本，项目更名为 **LammpsForcefieldTools**，PyPI 发行包名为 **`lammps-forcefield-tools`**，命令行入口保持 **`lft`**。

### Added

- 端到端管线：结构读入 → 连接性 → 原子类型 → 电荷 → 参数 → LAMMPS/GROMACS 产物。
- 结构格式：MOL2、PDB/ENT、PQR、GRO、XYZ、CIF/mmCIF、结构 JSON。
- 自动建键、角/二面角/improper/1-4 pair 枚举，以及带整数镜像位移的周期拓扑。
- 纯 Python GAFF 原子类型规则引擎，无外部二进制依赖。
- 纯 Python AM1 半经验 SCF 与 AM1-BCC 电荷。
- 内置 pySCF 量化后端：单点、PCM、Hessian、ESP。
- 电荷方法：`input`、`mmff94`、`am1-bcc`、`cm5`、`raw_cm5`、`adch`、`mk`、`resp`、`resp2`。
- LAMMPS 输出：`real` 单位 `.data` + `.in`，可选 moltemplate `.lt`/`system.lt`。
- GROMACS 输出：`.itp`、`.top`、`.gro`、`.mdp`。
- 统一的周期二面角表示：LAMMPS `fourier`、GROMACS functype 9。
- 全局 `lft.config` 配置、工具路径环境变量、结构化错误。
- 按需披露目录 `<name>.lft/`：`notes.txt`、`image_map.json`、`debug.txt`、`normalized_ir.json`。
- 手动类型与键合参数 JSON（`--manual-file`）。
- 引擎输入烟测（`--engine-smoke`）。
- 单元/回归测试、ground truth 清单、GAFF/电荷跨实现对照工具。

### Changed

- 项目名由工作名 `lammps_ff_topo` 变更为 **LammpsForcefieldTools**。
- PyPI 发行包名由 `lammps-ff-topo` 变更为 **`lammps-forcefield-tools`**。
- GitHub 仓库：<https://github.com/tgraphite/LammpsForcefieldTools>。
- 开发文档收敛为 [更新日志](CHANGELOG.md) 与 [模块现状](PROJECT_STATUS.md)；用户手册独立为 [使用手册](MANUAL.md)。
- README 重写为面向用户的安装、模块与参数说明。
- 许可证明确为 GPL-2.0。

### Removed

- CLI 参数 `--charge-file`（改用 `--charge <文件>`）、`--assemble`、`--mode`。
- 仓库主树与发行包中的开发参考数据、第三方二进制、临时产物、缓存与历史文档。

### Validation

- `pytest` 全量回归 **360 passed**。
- 文档/CLI 一致性测试 **9 passed**：手册覆盖全部命令行选项、电荷方法、输入格式、配置键与模块清单。
- `twine check` 对 wheel 与 sdist 均为 PASSED。
- Wheel 与 sdist 在独立虚拟环境安装成功，`pip check` 无冲突，`lft --help` 正常。
- 端到端样例：水分子 LAMMPS 输出可被 LAMMPS 2 Aug 2023 运行（退出码 0）；GROMACS 输出可被 `gmx grompp` 处理并生成 `.tpr`（退出码 0）。
- 历史参考验收保留：GAFF 类型 129/129、C 源码语义探针 3577/3577、GT-44 跨引擎单帧 10/10 ready。

### Known limitations

- 经典键合参数库不完整，未命中项使用确定性通用值并写警告。
- UFF/共轭二面角为占位参数。
- AM1-BCC 默认不做几何优化。
- 周期体系必须显式提供电荷。
- LAMMPS 固定 10 Å 直接截断，未启用 PPPM。

---

## 前期开发里程碑（未单独发布）

| 日期 | 里程碑 |
|---|---|
| 2026-09-01 | 项目初始化；统一 IR、拓扑、LAMMPS/GROMACS writer 与 GT 验收计划 |
| 2026-09-01 | 内部电荷引擎与纯 pySCF QC 落地 |
| 2026-09-10 | GAFF typing 与 AM1-BCC 纯 Python 化，去除产参数链路上的外部二进制 |
| 2026-09-13 | GAFF 与 AmberTools26 端到端差分 129/129；C 源码语义探针 3577/3577 |
| 2026-09-13 | CLI 易用性重构：默认 LAMMPS、自动电荷选择、`lft.config`、统一二面角形式 |
| 2026-09-13 | 晶体 GT-38/39/43/44 跨引擎单帧解阻，GT-44 ready 10/10 |
| 2026-09-17 | RESP 算法与 moltemplate 输出修复；AM1-BCC 默认关闭几何优化 |
| 2026-11-15 | 更名、文档重构、脚手架清理，发布 0.1.0 至 GitHub 与 PyPI |

[0.1.0]: https://github.com/tgraphite/LammpsForcefieldTools/releases/tag/v0.1.0
