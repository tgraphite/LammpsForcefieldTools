# 模块现状

**项目**：LammpsForcefieldTools  
**发行包**：`lammps-forcefield-tools`  
**版本**：`0.1.0`  
**状态日期**：2026-11-15  
**命令行入口**：`lft`

本文是开发侧的唯一模块状态来源，记录每个模块的职责、成熟度和已知边界。用户安装、参数与用法见 [使用手册](MANUAL.md)；版本变更见 [更新日志](CHANGELOG.md)。

---

## 1. 总体状态

| 维度 | 状态 | 说明 |
|---|---|---|
| 结构读入 | 稳定 | MOL2/PDB/PQR/GRO/XYZ/CIF/mmCIF/结构 JSON |
| 连接性与拓扑 | 稳定 | 显式键归一化、距离建键、角/二面角/improper/1-4 pair、镜像提升 |
| GAFF 原子类型 | 稳定 | 纯 Python 移植 `atomtype.c`，无外部二进制 |
| 电荷 | 稳定 | 输入、MMFF94、AM1-BCC、RESP/RESP2、MK、CM5/raw-CM5、ADCH |
| 量化后端 | 稳定 | 纯 pySCF；单点、PCM、Hessian、ESP |
| 键合参数 | 部分 | 参考库覆盖不完整；未命中项使用确定性通用值并告警 |
| 非键参数 | 部分 | GAFF/UFF 元素级 LJ 表；离子与特殊类型覆盖有限 |
| LAMMPS 输出 | 稳定 | `real` 单位、`.data` + `.in`，可选 `.lt` |
| GROMACS 输出 | 稳定 | `.itp/.top/.gro/.mdp` |
| 周期体系 | 稳定（有限制） | 镜像感知 IR、winding 披露；必须显式电荷 |
| 审计与诊断 | 稳定 | 内存审计 + 按需 `<name>.lft/` 纯文本披露 |

---

## 2. 模块清单

状态含义：

- **稳定**：生产路径使用，有回归测试。
- **部分**：可用但覆盖范围未完整，输出会带警告。
- **兼容层**：保留旧导入路径，转发到规范实现。
- **适配/对照**：外部工具或验证专用，不进入默认生产路径。

### 2.1 顶层模块

| 模块 | 职责 | 状态 |
|---|---|---|
| `src/lft/__init__.py` | 版本与包元信息 | 稳定 |
| `src/lft/__main__.py` | `python -m lft` 入口 | 稳定 |
| `src/lft/cli.py` | 参数解析、配置合并、警告汇总 | 稳定 |
| `src/lft/config.py` | `PipelineConfig` 校验与默认值 | 稳定 |
| `src/lft/pipeline.py` | 端到端任务图、产物与披露写出 | 稳定 |
| `src/lft/audit.py` | run id、哈希、产物清单、警告 | 稳定 |
| `src/lft/report.py` | 人类可读运行摘要 | 稳定 |
| `src/lft/settings.py` | `lft.config` 读取与环境变量提升 | 稳定 |
| `src/lft/errors.py` | 结构化错误类型 | 稳定 |
| `src/lft/bonds.py` | 连接性/键合项兼容导入 | 兼容层 |
| `src/lft/hessian.py` | Hessian 加载/投影兼容导入 | 兼容层 |
| `src/lft/model.py` | 结构模型兼容导入 | 兼容层 |
| `src/lft/parameters.py` | 参数记录兼容导入 | 兼容层 |
| `src/lft/units.py` | 单位换算兼容导入 | 兼容层 |

### 2.2 `domain/`

| 模块 | 职责 | 状态 |
|---|---|---|
| `src/lft/domain/structure.py` | 规范结构、原子、晶胞 | 稳定 |
| `src/lft/domain/topology.py` | 键/角/二面角/improper/pair 记录 | 稳定 |
| `src/lft/domain/parameters.py` | 后端无关参数记录 | 稳定 |
| `src/lft/domain/image.py` | 最小镜像位移与提升 | 稳定 |
| `src/lft/domain/units.py` | 规范单位换算 | 稳定 |

### 2.3 `io/`

| 模块 | 职责 | 状态 |
|---|---|---|
| `src/lft/io/loaders.py` | MOL2/PDB/PQR/GRO/XYZ/CIF/JSON 读入 | 稳定 |
| `src/lft/io/serializers.py` | 确定性 JSON 读写 | 稳定 |
| `src/lft/io/writers.py` | writer 兼容选择层 | 兼容层 |
| `src/lft/io/qc_loaders.py` | QC 产物读取兼容层 | 兼容层 |

### 2.4 `topology/`

| 模块 | 职责 | 状态 |
|---|---|---|
| `src/lft/topology/infer_bonds.py` | 距离建键、镜像位移归一化 | 稳定 |
| `src/lft/topology/enumerate_terms.py` | 角、二面角、improper、1-4 pair 枚举 | 稳定 |
| `src/lft/topology/pbc.py` | 周期图 winding 不变量 | 稳定 |

### 2.5 `forcefields/`

| 模块 | 职责 | 状态 |
|---|---|---|
| `src/lft/forcefields/gaff_typer.py` | 纯 Python GAFF 类型规则引擎 | 稳定 |
| `src/lft/forcefields/matching.py` | GAFF/UFF/manual 分派与非键参数 | 稳定 |
| `src/lft/forcefields/manual.py` | 手动参数文件解析 | 稳定 |
| `src/lft/forcefields/tables.py` | 元素、质量、LJ 表 | 部分 |
| `src/lft/forcefields/reference.py` | 参考键合参数读取 | 部分 |
| `src/lft/forcefields/atomtype.py` | 外部 `atomtype` 适配 | 适配/对照 |
| `src/lft/forcefields/gaff.py` | GAFF typing 兼容导入 | 兼容层 |
| `src/lft/forcefields/uff.py` | UFF typing 兼容导入 | 兼容层 |

### 2.6 `charges/`

| 模块 | 职责 | 状态 |
|---|---|---|
| `src/lft/charges/dispatcher.py` | 电荷方法分派与守恒校验 | 稳定 |
| `src/lft/charges/input.py` | 电荷文件/结构电荷读入 | 稳定 |
| `src/lft/charges/empirical.py` | MMFF94 与 AM1-BCC 适配 | 稳定 |
| `src/lft/charges/resp.py` | RESP/RESP2 拟合 | 稳定 |
| `src/lft/charges/esp.py` | ESP 网格与约束拟合 | 稳定 |
| `src/lft/charges/density_charges.py` | Mulliken/Hirshfeld/CM5/ADCH | 稳定 |
| `src/lft/charges/_grid.py` | 确定性格点 | 稳定 |
| `src/lft/charges/_atomic_density.py` | 自由原子径向密度 | 稳定 |
| `src/lft/charges/_cm5_data.py` | CM5/ADCH 参考数据 | 稳定 |
| `src/lft/charges/semiempirical/am1.py` | 纯 Python AM1 RHF SCF | 稳定 |
| `src/lft/charges/semiempirical/bcc.py` | AM1-BCC 电荷生成 | 稳定 |
| `src/lft/charges/semiempirical/bondtype.py` | AmberTools bondtype 复现 | 稳定 |
| `src/lft/charges/acpype.py` | 外部 acpype 适配 | 适配/对照 |

### 2.7 `qc/`

| 模块 | 职责 | 状态 |
|---|---|---|
| `src/lft/qc/backend.py` | QC 协议与分派 | 稳定 |
| `src/lft/qc/pyscf_backend.py` | pySCF 单点/PCM/Hessian | 稳定 |
| `src/lft/qc/wavefunction.py` | 波函数数据容器 | 稳定 |
| `src/lft/qc/hessian.py` | Hessian 文件读取与单位归一 | 稳定 |

### 2.8 `parameterize/`

| 模块 | 职责 | 状态 |
|---|---|---|
| `src/lft/parameterize/classic.py` | 经典/手动键合参数准备 | 部分 |
| `src/lft/parameterize/hessian.py` | Hessian 键常数投影 | 部分 |

### 2.9 `backends/`

| 模块 | 职责 | 状态 |
|---|---|---|
| `src/lft/backends/moltemplate.py` | LAMMPS `.data`/`.in` 与 moltemplate 输出 | 稳定 |
| `src/lft/backends/gromacs.py` | GROMACS `.itp/.top/.gro/.mdp` | 稳定 |
| `src/lft/backends/function_map.py` | 跨引擎函数形式对照 | 稳定 |

### 2.10 `runners/`

| 模块 | 职责 | 状态 |
|---|---|---|
| `src/lft/runners/external.py` | 受控子进程执行 | 稳定 |
| `src/lft/runners/gmx.py` | `gmx grompp`/`mdrun` | 稳定 |
| `src/lft/runners/lammps.py` | `lmp` 执行 | 稳定 |
| `src/lft/runners/moltemplate.py` | moltemplate 执行 | 稳定 |
| `src/lft/runners/packmol.py` | Packmol 执行 | 稳定 |

### 2.11 `validation/`

| 模块 | 职责 | 状态 |
|---|---|---|
| `src/lft/validation/ground_truth.py` | 清单加载与计数校验 | 稳定 |
| `src/lft/validation/normalized.py` | 与文件布局无关的结构/电荷校验 | 稳定 |
| `src/lft/validation/differential.py` | 数组差分报告 | 稳定 |
| `src/lft/validation/physical.py` | 势能公式点检 | 稳定 |
| `src/lft/validation/gaff_oracle.py` | AmberTools26 oracle 隔离执行 | 对照/验证专用 |

---

## 3. 能力矩阵

| 能力 | 输入/选项 | 产物/行为 |
|---|---|---|
| 结构格式 | `.mol2`、`.pdb`、`.ent`、`.pqr`、`.gro`、`.xyz`、`.cif`、`.mmcif`、`.json` | 统一镜像感知 IR |
| 原子类型 | `gaff`、`gaff+uff`、`uff`、`manual` | GAFF 规则引擎 / UFF 回退 / 显式映射 |
| 电荷 | `input`、`mmff94`、`am1-bcc`、`cm5`、`raw_cm5`、`adch`、`mk`、`resp`、`resp2` | 原子电荷 + 守恒校验 |
| 量化 | `--qc method/basis`、`--qc-backend pyscf` | 单点、PCM、Hessian、ESP |
| 参数模式 | `classic`、`hessian` | 参考/几何/投影参数，带来源标记 |
| LAMMPS | `--output lammps` | `.data`、`.in`，可选 `.lt`/`system.lt` |
| GROMACS | `--output gromacs` | `.itp`、`.top`、`.gro`、`.mdp` |
| 周期 | 晶体 CIF/PDB/MOL2 | 镜像提升、winding 披露、显式电荷强制 |
| 诊断 | `--debug` | `debug.txt`、`normalized_ir.json` |
| 披露 | 任何警告 | `<name>.lft/notes.txt` |
| 引擎烟测 | `--engine-smoke` | `grompp` 或 `lmp` 输入检查 |

---

## 4. 验证矩阵

以下为 0.1.0 发布前的实测结果。测试环境：Python 3.11、Linux x86_64、pySCF 2.14、NumPy 1.26。

| 检查 | 命令 | 最近结果（2026-11-15） |
|---|---|---|
| 单元/回归测试 | `python -m pytest -q` | **360 passed**，42.55 s |
| 文档/CLI 一致性 | `pytest -q tests/test_docs_consistency.py` | **9 passed** |
| 命令行帮助 | `lft --help` / `lft -V` | `lft 0.1.0`，帮助正常渲染 |
| 包元数据 | `twine check dist/*` | wheel 与 sdist 均 **PASSED** |
| Wheel 安装 | 独立 venv 安装 `*.whl` | 安装成功，`pip check` 无冲突，入口 `lft` 可用 |
| Sdist 安装 | 独立 venv 从 `*.tar.gz` 构建安装 | 构建安装成功，`lft --version` 正常 |
| 端到端 LAMMPS | wheel 安装后 `lft -f examples/water.mol2` | 生成 `.data`+`.in`；LAMMPS 2 Aug 2023 运行退出码 0 |
| 端到端 GROMACS | wheel 安装后 `lft -f examples/water.mol2 -o gromacs` | 生成 4 个文件；`gmx grompp` 生成 `.tpr`，退出码 0 |
| 确定性 | `tests/test_idempotency.py` | 通过（含重复运行 SHA-256 一致） |
| GAFF 端到端差分 | `scripts/validate_gaff_typer_vs_oracle.py` | 129/129（历史保留） |
| GAFF C 语义探针 | `scripts/validate_gaff_source_semantics.py` | 3577/3577（历史保留） |
| 电荷 vs Multiwfn | `scripts/validate_charges_vs_multiwfn.py` | 密度电荷 ~5e-9 e；RESP ~2e-5–4e-5 e（历史保留） |
| GT-44 跨引擎单帧 | `scripts/gt44_runner.py` | 10/10 ready（历史保留） |
| 全量交接验收 | `scripts/verify_handoff.sh` | 依赖 `references/` 与 `thirdparty/`，发布包不含 |

> 上表“历史保留”表示使用仓库内参考数据复跑过的结果；PyPI 用户环境不包含这些参考数据与外部引擎。

---

## 5. 已知边界

1. **经典键合参数**：GAFF 原子类型完整，但键/角/二面角参考库覆盖有限。未命中项使用确定性默认值（`classic:geometry_guess`、`classic:generic_periodic`）并写 `PARAMETER_GENERIC_GUESS` 警告。
2. **UFF/共轭二面角**：使用 k=0.50、phase=0、n=2 的占位周期项；函数形式与单位正确，数值不是完整 UFF 扭转库。
3. **AM1-BCC**：默认不做几何优化，电荷依赖输入构象；与 AmberTools `sqm` 为化学级一致（~1e-2 e），非逐位一致。
4. **RESP**：与 Multiwfn 参考逐原子一致到 ~2e-5–4e-5 e；MK 为单阶段无约束拟合。
5. **周期电荷**：只接受显式电荷，不做气相 ESP 拟合；winding 图必须读取 `image_map.json`。
6. **LAMMPS 输出**：固定 `real` 单位、`lj/cut/coul/cut 10.0`、`special_bonds lj 0 0 0.5 / coul 0 0 0.8333333333`；未实现 PPPM。
7. **GROMACS 输出**：非周期输入补兼容盒；部分 PBC 掩码写成全周期盒并披露。
8. **外部引擎**：`--engine-smoke`、moltemplate、Packmol 需要用户环境提供可执行文件；缺失时报结构化错误。
9. **参考数据**：`references/`、`thirdparty/`、`research/`、`_scratch/`、`_archive/` 仅保留在开发工作区，不进入 GitHub 主树或发行包。
10. **范围**：不做构象搜索、体系组装、MD 采样、力场重优化或 GUI。

---

## 6. 兼容入口

以下旧导入路径保留，内部转发到规范模块：

- `lft.bonds` → `lft.topology`
- `lft.hessian` → `lft.qc.hessian` / `lft.parameterize.hessian`
- `lft.model` → `lft.domain`
- `lft.parameters` → `lft.domain.parameters`
- `lft.units` → `lft.domain.units`
- `lft.io.writers` → `lft.backends`
- `lft.io.qc_loaders` → `lft.qc`
- `lft.forcefields.gaff` / `lft.forcefields.uff` → `lft.forcefields.matching`

已移除的 CLI 入口：`--charge-file`（改用 `--charge <file>`）、`--assemble`、`--mode`。

---

## 7. 维护重点

| 优先级 | 事项 |
|---|---|
| 高 | 补齐 GAFF/UFF 键合参数库，消除 generic 占位 |
| 高 | 扩大周期与晶体回归覆盖，逐项比对 LJ/Coulomb |
| 中 | 固定 lint/类型检查版本与规则集 |
| 中 | 增加非周期大分子与金属配合物样例 |
| 低 | 发布流程自动化（GitHub Actions → PyPI trusted publishing） |
