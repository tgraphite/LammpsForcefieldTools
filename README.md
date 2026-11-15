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
| 电荷 | 输入、MMFF94、AM1-BCC、RESP/RESP2、MK、CM5/raw-CM5、ADCH |
| 量化 | 内置 pySCF：单点、PCM、Hessian、ESP，无外部量化程序依赖 |
| 参数化 | 经典参数 + Hessian 投影；手动参数 JSON |
| LAMMPS 输出 | `real` 单位、`atom_style full`、`.data` + `.in`，可选 moltemplate |
| GROMACS 输出 | `.itp`、`.top`、`.gro`、`.mdp` |
| 周期体系 | 镜像感知键合项；晶体必须显式提供电荷 |
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
water.data
water.in
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
| `--force` | 关闭 | 覆盖同名产物 |
| `--debug` | 关闭 | 写出 `<name>.lft/debug.txt` 诊断包 |
| `-V`, `--version` | — | 显示版本 |
| `-h`, `--help` | — | 显示帮助 |

### 电荷自动选择

未指定 `--charge` 时：

1. 结构自带电荷 → 直接使用；
2. 非周期且 ≤100 原子 → `resp`；
3. 非周期且 >100 原子 → `am1-bcc`；
4. 周期结构 → 必须显式提供电荷，否则报错。

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

---

## 文档

| 文档 | 内容 |
|---|---|
| [使用手册](docs/MANUAL.md) | 安装、完整 CLI、电荷、周期、产物、排错 |
| [模块现状](docs/PROJECT_STATUS.md) | 模块成熟度、能力矩阵、验证结果与已知边界 |
| [更新日志](docs/CHANGELOG.md) | 版本变更 |

---

## 已知限制

- 经典键合参数库尚不完整；未命中项使用确定性通用值，并在 `notes.txt` 中标注。
- UFF/共轭二面角当前使用占位参数，定量工作前应核对或用 `--manual-file` 覆盖。
- AM1-BCC 默认不做几何优化，电荷依赖输入构象。
- 周期体系只接受显式电荷，不做 ESP/密度电荷拟合。
- LAMMPS 输出固定 10 Å 直接截断，未启用 PPPM/PME。
- 不提供构象搜索、体系组装、MD 采样或力场重优化。

---

## 许可证

GNU General Public License v2.0，见 [LICENSE](LICENSE)。
