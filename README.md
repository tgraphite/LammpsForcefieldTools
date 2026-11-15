# LammpsForcefieldTools

[![PyPI](https://img.shields.io/pypi/v/lammps-forcefield-tools.svg)](https://pypi.org/project/lammps-forcefield-tools/)
[![Python](https://img.shields.io/pypi/pyversions/lammps-forcefield-tools.svg)](https://pypi.org/project/lammps-forcefield-tools/)
[![License: GPL-2.0](https://img.shields.io/badge/license-GPL--2.0-blue.svg)](LICENSE)

把分子/晶体结构自动转换为可运行的 **LAMMPS** 力场与拓扑输入，同时提供 **GROMACS** 兼容输出。命令行入口为 **`lft`**。

```bash
lft -f molecule.mol2          # 生成 molecule.data + molecule.in
lmp -in molecule.in           # 直接运行
```

- GitHub：<https://github.com/tgraphite/LammpsForcefieldTools>
- PyPI：<https://pypi.org/project/lammps-forcefield-tools/>
- 使用手册：[docs/MANUAL.md](docs/MANUAL.md)

---

## 功能特性

| 能力 | 说明 |
|---|---|
| 结构读入 | MOL2、PDB/ENT、PQR、GRO、XYZ、CIF/mmCIF、结构 JSON |
| 连接性 | 显式键优先；无键结构按共价半径与价键上限自动建键 |
| 拓扑 | 键、角、二面角、improper、1-4 pair，支持周期镜像 |
| 原子类型 | GAFF 规则引擎；`gaff+uff` 自动回退；支持纯 UFF 与手动映射 |
| 电荷 | 输入、MMFF94、AM1-BCC、RESP/RESP2、MK、CM5/raw-CM5、ADCH；`dft-` 前缀先用 DFT 优化几何 |
| 量化 | 内置 pySCF：单点、PCM、Hessian、ESP；geometric 几何优化，无外部量化程序依赖 |
| 几何预优化 | 默认 UFF（全元素可用）；`mmff94` 用同名优化器，`dft` 用 pySCF |
| 参数化 | 内置 GAFF 1.81 参数库（键/角/多项二面角/improper/LJ），离线可用；Hessian 键投影；手动参数 JSON |
| LAMMPS 输出 | 默认 moltemplate（`.lt` + `system.lt`）；`--lammpsdata` 写 `.data` + `.in`（周期体系启用 PPPM） |
| GROMACS 输出 | `.itp`、`.top`、`.gro`、`.mdp` |
| 周期体系 | 镜像感知键合项；晶体必须显式提供电荷；LAMMPS 周期默认 PPPM |
| 诊断 | 重要警告直接显示；完整披露写入 `<name>.lft/` |

---

## 安装

要求 **Python 3.11+**。

```bash
python -m pip install --upgrade pip
python -m pip install lammps-forcefield-tools
```

检查安装：

```bash
lft --version
lft --help
```

从源码安装：

```bash
git clone https://github.com/tgraphite/LammpsForcefieldTools.git
cd LammpsForcefieldTools
python -m pip install .
```

开发/测试安装：

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

安装包会自动拉取：`numpy`、`scipy`、`pyscf`、`dftd3`、`cclib`、`openbabel-wheel`、`PyYAML`。

---

## 快速开始

仓库自带一个带电荷的水分子示例：

```bash
cd examples
lft -f water.mol2
```

输出：

```text
water.lt
system.lt
```

用 moltemplate 生成 LAMMPS 输入：

```bash
moltemplate.sh system.lt
lmp -in run.in          # 由 run.in.EXAMPLE 改写，或自行 include system.in.init/data/settings
```

要传统的 data/in 文件：

```bash
lft -f water.mol2 --lammpsdata
```

运行 LAMMPS：

```bash
lmp -in water.in
```

生成 GROMACS 输入：

```bash
lft -f water.mol2 --output gromacs
# water.itp  water.top  water.gro  water.mdp
```

生成后立即做引擎输入烟测：

```bash
lft -f water.mol2 --engine-smoke
```

---

## 模块

程序按处理阶段组织，命令行会把各阶段自动串联：

| 阶段 | Python 包 | 职责 |
|---|---|---|
| 结构 | `lft.io` | 读取常见分子/晶体格式，输出统一结构模型 |
| 领域模型 | `lft.domain` | 原子、晶胞、拓扑项、参数、单位 |
| 拓扑 | `lft.topology` | 建键、键合项枚举、周期镜像处理 |
| 力场 | `lft.forcefields` | GAFF/UFF/manual 原子类型与非键参数 |
| 电荷 | `lft.charges` | AM1-BCC、RESP、密度电荷、电荷文件 |
| 量化 | `lft.qc` | pySCF 单点、PCM、Hessian、ESP |
| 参数化 | `lft.parameterize` | 经典参数与 Hessian 投影 |
| 后端 | `lft.backends` | LAMMPS 与 GROMACS 文件生成 |
| 外部程序 | `lft.runners` | gmx、lmp、moltemplate、Packmol 受控调用 |
| 校验 | `lft.validation` | 结构、电荷、能量与 ground truth 校验 |

---

## 通用参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `-f`, `--structure` | 必填 | 输入结构文件 |
| `-o`, `--output` | `lammps` | `lammps` 或 `gromacs` |
| `--ff` | `gaff+uff` | `gaff`、`gaff+uff`、`uff`、`manual` |
| `--charge` | 自动 | 电荷方法名或电荷文件路径 |
| `--net-charge` | `0` | 分子形式电荷 |
| `--mult`, `--multiplicity` | `1` | 自旋多重度 |
| `--outdir` | 当前目录 | 产物目录 |
| `--lammpsdata` | 关闭 | 写 `.data` + `.in`，而不是 moltemplate 文件 |
| `--preopt` | `auto` | 几何预优化：`auto`/`none`/`uff`/`mmff94`/`dft` |
| `--current-geometry` | 关闭 | 平衡值取当前几何（构象专用，会打印 INFO） |
| `--allow-missing-torsions` | 关闭 | 无参二面角省略而非报错 |
| `--generic-improper` | `omit` | 无参 improper：`omit`/`amber`/`k10` |
| `--force` | 关闭 | 覆盖同名产物 |
| `--debug` | 关闭 | 写出 `<name>.lft/debug.txt` 诊断包 |
| `--check-update` | — | 立即检查 PyPI/配置镜像并退出 |
| `--no-update-check` | 关闭 | 本次运行跳过自动更新提示 |
| `-V`, `--version` | — | 显示版本 |
| `-h`, `--help` | — | 显示帮助 |

### 电荷自动选择

未指定 `--charge` 时：

1. 结构自带电荷 → 直接使用；
2. 非周期且 ≤50 原子 → `resp`；
3. 非周期且 51–200 原子 → `am1-bcc`；
4. 非周期且 >200 原子 → `mmff94`；
5. 周期结构 → 必须显式提供电荷，否则报错。

`--charge` 也可以直接给文件路径：

```bash
lft -f molecule.xyz --charge charges.txt
lft -f crystal.pdb --charge charges.txt --ff uff
```

### 高级参数

以下参数默认值通常正确，完整说明见[使用手册](docs/MANUAL.md)：

`--parameter-mode`、`--qc`、`--qc-backend`、`--hessian-file`、`--hessian-method`、`--manual-file`、`--no-qc`、`--engine-smoke`、`--moltemplate`。

---

## 输入与输出

### 输入格式

| 后缀 | 连接性 | 周期性 |
|---|---|---|
| `.mol2` | `@<TRIPOS>BOND`（推荐） | `@<TRIPOS>CRYSIN` |
| `.pdb`、`.ent`、`.pqr` | `CONECT` 或距离推断 | `CRYST1` |
| `.gro` | 距离推断 | box 行 |
| `.xyz` | 距离推断 | 注释行晶格（可选） |
| `.cif`、`.mmcif` | 距离推断 | 晶胞参数 |
| `.json` | 文件内 `bonds` | 文件内 `cell` |

### 输出文件

LAMMPS（默认）：

```text
<name>.data   坐标、拓扑、原子类型与系数
<name>.in     units / pair_style / bond_style / 运行设置
```

GROMACS：

```text
<name>.itp    原子、键、角、二面角、1-4 pair
<name>.top    力场默认项与系统定义
<name>.gro    坐标
<name>.mdp    最小运行参数
```

按需出现：

```text
<name>.lft/notes.txt          警告与参数来源披露
<name>.lft/image_map.json     周期图镜像映射
<name>.lft/debug.txt          --debug 诊断
<name>.lft/normalized_ir.json --debug 规范化结构
```

---

## 配置

可选全局配置文件 `lft.config`，格式为 `key = value`：

```ini
output = lammps
ff = gaff+uff
charge = resp
qc = b3lyp-d3bj/def2-svp
gmx = /usr/local/gromacs/bin/gmx
lmp = /opt/lammps/bin/lmp
```

查找顺序：`$LFT_CONFIG` → 安装根目录 → Python 包目录。优先级：**命令行 > 配置文件 > 内置默认**。

常用环境变量：

| 变量 | 作用 |
|---|---|
| `LFT_CONFIG` | 配置文件路径 |
| `LFT_GMX` | GROMACS 可执行文件 |
| `LFT_LMP` | LAMMPS 可执行文件 |
| `LFT_OBABEL` | OpenBabel 可执行文件 |
| `LFT_AM1BCC_OPTIMIZE` | AM1-BCC 几何优化：`yes`/`no`/`auto` |
| `LFT_ATOMTYPE_EXTERNAL` | 用外部 `atomtype` 做对照 |
| `LFT_NO_UPDATE_CHECK` | 设为 `1` 关闭自动更新检查 |
| `LFT_UPDATE_CHECK` | 设为 `0` 关闭自动更新检查 |
| `LFT_UPDATE_URL` | 自定义 JSON 版本端点 |
| `LFT_UPDATE_INDEX` | 自定义 PEP 503 简单索引 |
| `LFT_UPDATE_INTERVAL_HOURS` | 覆盖检查间隔（默认 24） |
| `LFT_UPDATE_TIMEOUT_S` | 覆盖单源超时（默认 1.5 秒） |

### 更新检查

`lft` 在交互终端启动时做一次**只提示、不安装**的版本检查：

- 默认每 24 小时最多检查一次；结果缓存在 `$XDG_CACHE_HOME/lft/update.json`。
- 检查在后台线程执行，单源超时 1.5 秒，不阻塞管线，也不影响退出码。
- 同时查询 PyPI、`PIP_INDEX_URL`、`LFT_UPDATE_URL` / `LFT_UPDATE_INDEX`，取最高版本。
- 发现新版本时只在 stderr 打印一行提示，例如：
  `[lft NOTICE] new version 1.2.0 available (current 1.1.0); upgrade with: pip install -U lammps-forcefield-tools`
- 非交互终端、`CI`、`--version` / `--help` 自动跳过。
- 关闭：`LFT_NO_UPDATE_CHECK=1` 或 `update_check = off`；手动检查：`lft --check-update`；
  单次跳过：`lft --no-update-check -f ...`。
- `lft` 不会自动下载或执行更新；升级始终由你通过 pip/uv 完成。

---

## 文档

| 文档 | 内容 |
|---|---|
| [使用手册](docs/MANUAL.md) | 安装、完整 CLI、电荷、周期、产物、排错 |
| [模块现状](docs/PROJECT_STATUS.md) | 模块成熟度、能力矩阵、验证结果与已知边界 |
| [更新日志](docs/CHANGELOG.md) | 版本变更 |

---

## 已知限制

- 内置 GAFF 键合参数覆盖 GAFF 1.81 的键/角/二面角；极少数角（如 `h4-c-h4`）仍会用几何通用值，`notes.txt` 会标注。
- UFF 键合项仍是规则占位值；UFF 仅作为 GAFF 不支持元素的后备，定量工作建议核对或 `--manual-file` 覆盖。
- 纯经验电荷的预优化用 UFF（毫秒级）；`dft-` 电荷会做 DFT 几何优化，成本随原子数快速增长（默认上限 50 原子）。
- 周期体系只接受显式电荷，不做 ESP/密度电荷拟合；周期结构暂不参与 1:1 测试。
- 周期 LAMMPS 输出默认 PPPM（`kspace_style pppm 1.0e-4`）；非周期使用 10 Å 直接截断。
- 不提供构象搜索、体系组装、MD 采样或力场重优化。

---

## 许可证

GNU General Public License v2.0，见 [LICENSE](LICENSE)。
