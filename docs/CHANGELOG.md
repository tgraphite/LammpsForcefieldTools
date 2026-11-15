# Changelog

本文件记录 LammpsForcefieldTools 的版本变更。格式参考 Keep a Changelog，版本号遵循语义化版本。

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
