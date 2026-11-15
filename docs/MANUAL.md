# LammpsForcefieldTools 使用手册

**项目名**：LammpsForcefieldTools  
**Python 发行包**：`lammps-forcefield-tools`  
**命令行入口**：`lft`（安装后全局可用，也可用 `python -m lft`）

`lft` 把「结构 → 连接性 → 原子类型 → 电荷 → 参数 → 后端文件」整条链路自动化，默认生成可直接运行的 LAMMPS 输入，同时提供 GROMACS 兼容输出。

---

## 1. 安装

需要 Python 3.11 或更高版本。

```bash
python -m pip install --upgrade pip
python -m pip install lammps-forcefield-tools

lft --version
lft --help
```

源码安装：

```bash
git clone https://github.com/tgraphite/LammpsForcefieldTools.git
cd LammpsForcefieldTools
python -m pip install .
```

开发安装与测试：

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

依赖由发行包自动安装：`numpy`、`scipy`、`pyscf`、`dftd3`、`cclib`、`openbabel-wheel`、`PyYAML`。

---

## 2. 一句话用法

```bash
lft -f molecule.mol2
```

在当前目录写出 `molecule.data` 和 `molecule.in`，可直接运行：

```bash
lmp -in molecule.in
```

常用变体：

```bash
lft -f molecule.xyz --output gromacs          # .itp/.top/.gro/.mdp
lft -f molecule.xyz --charge charges.txt      # 显式电荷文件
lft -f crystal.pdb --charge charges.txt --ff uff
lft -f molecule.mol2 --debug                  # 额外写出 <name>.lft/debug.txt
```

---

## 3. 命令行参考

### 3.1 完整帮助

```text
usage: lft [-h] -f STRUCTURE [-o {lammps,gromacs}]
           [--ff {gaff,gaff+uff,uff,manual}] [--charge METHOD|FILE]
           [--net-charge NET_CHARGE] [--mult MULTIPLICITY] [--outdir OUTDIR]
           [--force] [--debug] [-V]

Generate ready-to-run LAMMPS or GROMACS force field and topology files.
```

### 3.2 常用参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `-f`, `--structure` | 必填 | 输入结构文件 |
| `-o`, `--output` | `lammps` | 输出后端：`lammps` 或 `gromacs` |
| `--ff` | `gaff+uff` | 原子类型策略：`gaff`、`gaff+uff`、`uff`、`manual` |
| `--charge` | 自动 | 电荷方法名或电荷文件路径 |
| `--net-charge` | `0` | 分子形式电荷（整数） |
| `--mult`, `--multiplicity` | `1` | 自旋多重度 |
| `--outdir` | 当前目录 | 产物目录 |
| `--force` | 关闭 | 覆盖已存在的同名产物 |
| `--debug` | 关闭 | 在 `<name>.lft/` 写纯文本诊断包 |
| `-h`, `--help` | — | 显示帮助 |
| `-V`, `--version` | — | 显示版本 |

### 3.3 高级参数（默认值通常正确）

帮助信息中不展开，但仍受支持：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--parameter-mode` | `classic` | 键合参数模式：`classic` 或 `hessian` |
| `--qc` | `b3lyp-d3bj/def2-svp` | 量化方法/基组，格式 `method/basis`；支持 `hf`、`b3lyp`、`pbe0`、`pbe`、`blyp` 等 |
| `--qc-backend` | `pyscf` | 量化后端：`pyscf` 或 `auto` |
| `--hessian-file` | 无 | 外部 Hessian；支持 `.npy`、`.json`、`.fchk`、`.hess`、`.out`、`.log` |
| `--hessian-method` | `seminario` | 键常数投影：`seminario`、`mseminario`、`m2seminario`、`drih` |
| `--manual-file` | 无 | 手动类型/参数 JSON（见 7.4 节） |
| `--no-qc` | 关闭 | 禁止隐式量化；所需数据必须已由输入提供 |
| `--engine-smoke` | 关闭 | 生成后调用 `gmx grompp` 或 `lmp` 做输入烟测 |
| `--moltemplate` | 关闭 | 额外写出 `<name>.lt` 与 `system.lt` |

示例：

```bash
# Hessian 驱动的键合参数
lft -f ethanol.mol2 --parameter-mode hessian --hessian-file ethanol.hess \
    --hessian-method seminario --charge charges.txt

# 手动类型与参数
lft -f fragment.xyz --ff manual --manual-file manual.json --charge input

# 生成后立即用本机 LAMMPS 做输入烟测
lft -f molecule.mol2 --engine-smoke
```

---

## 4. 默认行为与自动选择

| 项目 | 行为 |
|---|---|
| 输出后端 | `lammps` |
| 力场 | `gaff+uff`：GAFF 优先，缺失类型回退 UFF 并记录警告 |
| 电荷 | 结构自带电荷 → 直接用；否则 ≤100 原子用 `resp`，>100 原子用 `am1-bcc` 并提示 |
| 产物目录 | 当前工作目录，文件名取输入文件主名 |
| 覆盖 | 默认拒绝；同名产物已存在时要求 `--force` 或换 `--outdir` |
| 周期体系 | 必须提供显式电荷；未提供直接报错 |
| 日志 | 仅重要警告打印到 stderr；全部披露写入 `<name>.lft/notes.txt` |

显式 `--charge` 始终覆盖自动选择。`--charge` 的值若匹配已知方法名则按方法处理，否则按电荷文件路径处理。

---

## 5. 输入格式与拓扑来源

| 格式 | 后缀 | 连接性来源 | 周期性来源 |
|---|---|---|---|
| Tripos MOL2 | `.mol2` | `@<TRIPOS>BOND`，推荐 | `@<TRIPOS>CRYSIN` |
| PDB / PQR | `.pdb`、`.ent`、`.pqr` | 显式 `CONECT`，否则按距离推断 | `CRYST1` |
| GROMACS | `.gro` | 按距离推断 | box 行 |
| XYZ | `.xyz` | 按共价半径与价键上限推断 | 注释行晶格（可选） |
| CIF / mmCIF | `.cif`、`.mmcif` | 按距离推断 | 晶胞参数 |
| 结构 JSON | `.json` | 文件内 `bonds`（可选） | 文件内 `cell` |

说明：

- 有显式键的文件优先信任输入键序；键序 `1/2/3/ar` 会影响 GAFF 原子类型与键参数。
- 无键结构按元素共价半径与价键上限自动建键。
- 复杂共轭、芳香、金属配合物建议提供带键的 MOL2。
- 周期结构必须同时提供电荷；晶胞信息不会触发电荷拟合。

---

## 6. 电荷模型

`--charge` 接受以下方法名，或一个电荷文件路径。

| 方法 | 需要 | 说明 |
|---|---|---|
| `input` | 结构自带电荷或电荷文件 | 最快；周期体系必须使用 |
| `mmff94` | OpenBabel | 快速经验电荷，适合有机小分子 |
| `am1-bcc` | 内置纯 Python AM1 | 大分子默认；较慢 |
| `cm5` | 内置 pySCF 单点 | Hirshfeld + CM5 校正；中性分子做 1.2 倍缩放 |
| `raw_cm5` | 内置 pySCF 单点 | 不做 1.2 倍缩放的 CM5 |
| `adch` | 内置 pySCF 单点 | ADCH 密度电荷 |
| `mk` | 内置 pySCF + ESP 拟合 | 单阶段无约束最小二乘 |
| `resp` | 内置 pySCF + ESP 拟合 | 两阶段 RESP；小分子默认 |
| `resp2` | 内置 pySCF，含水相单点 | 两阶段 RESP 的气相/水相混合，最慢 |

电荷文件格式：

- JSON 数组：`[0.1, -0.2, 0.1]`
- JSON 对象：`{"charges": [0.1, -0.2, 0.1]}`
- 纯文本：每行一个数值；首行可写原子数；`#` 开头为注释

电荷守恒容差为 `1e-5 e`。计算所得净电荷与 `--net-charge` 不一致时给出结构化错误。

---

## 7. 力场与参数

### 7.1 `--ff` 语义

| 名称 | 行为 | 缺失类型时 |
|---|---|---|
| `gaff` | 只用 GAFF | 直接报错，不静默回退 |
| `gaff+uff` | GAFF 优先 | 回退 UFF，并在 `notes.txt` 披露每个原子 |
| `uff` | 全部 UFF | 不调用 GAFF typing |
| `manual` | 全部显式 | 缺少类型或参数直接报错 |

GAFF 覆盖元素：H、B、C、N、O、F、Si、P、S、Cl、Br、I。其他元素在 `gaff+uff` 下回退 UFF。

### 7.2 参数模式

- `classic`（默认）：键长/键角取自输入几何，力常数使用确定性默认值；能匹配参考库的项使用参考值。未匹配的键合项标记为 `classic:` 来源并写入警告 `PARAMETER_GENERIC_GUESS`。
- `hessian`：由外部 Hessian 或内置 pySCF Hessian 投影得到键常数；键角仍使用确定性默认值。必须提供 `--hessian-file` 或允许 QC。

> 当前版本的经典键合参数库尚不完整，二面角对 UFF/共轭项使用占位符。用于定量生产前请检查 `<name>.lft/notes.txt`，或用 `--manual-file` 覆盖关键参数。

### 7.3 二面角表示

所有 proper 二面角统一为单一周期形式：

```text
E = K [1 + cos(n φ − δ)]
```

- LAMMPS：`dihedral_style fourier`，K 单位 kcal/mol
- GROMACS：functype 9，K 单位 kJ/mol

### 7.4 手动参数文件

`--manual-file` 接受 JSON：

```json
{
  "atoms": {"1": "Cmanual", "2": "Hmanual"},
  "types": {
    "Cmanual": {"mass": 12.0, "sigma": 3.4, "epsilon": 0.1},
    "Hmanual": {"mass": 1.0, "sigma": 2.5, "epsilon": 0.02}
  },
  "bonds": {"Cmanual-Hmanual": {"r0": 1.1, "k": 300.0}},
  "angles": {"Hmanual-Cmanual-Hmanual": {"theta0": 109.5, "k": 50.0}},
  "dihedrals": {"Cmanual-Cmanual-Cmanual-Cmanual": {"phase": 0.0, "k": 0.15, "periodicity": 3}},
  "impropers": {"Cmanual-Nmanual-Omanual-Hmanual": {"phase": 0.0, "k": 10.0, "periodicity": 2}}
}
```

键名使用原子类型；无序项可反序匹配。`manual` 模式下缺少任何键合参数都会报错。

---

## 8. 周期性/晶体规则

- 含晶胞的结构只接受显式电荷：`--charge input` 加结构自带电荷，或 `--charge charges.txt`。
- 周期结构若指定 `resp`、`am1-bcc` 等拟合方法会直接报错，避免把气相 ESP 拟合套到晶胞。
- 跨胞面的键、角、二面角带整数镜像位移；当周期图存在无法用单一全局镜像表示的 winding 时，会写出 `<name>.lft/image_map.json` 并给出 `LAMMPS_PERIODIC_LIFT_SIDECAR` 警告。
- GROMACS 侧跨镜像键由 PBC 处理；非周期输入会补一个有限兼容盒并记录警告。

---

## 9. 输出产物

### 9.1 LAMMPS（默认）

| 文件 | 内容 |
|---|---|
| `<name>.data` | `real` 单位、`atom_style full`、坐标、键/角/二面角/improper、原子类型与系数 |
| `<name>.in` | `units`、`pair_style lj/cut/coul/cut 10.0`、`bond_style harmonic`、`angle_style harmonic`、`dihedral_style fourier`、`improper_style cvff`、`special_bonds` 与运行设置 |
| `<name>.lt`、`system.lt` | 仅在 `--moltemplate` 时生成 |

### 9.2 GROMACS

| 文件 | 内容 |
|---|---|
| `<name>.itp` | `[moleculetype]`、`[atoms]`、`[bonds]`、`[angles]`、`[dihedrals]`、`[pairs]` |
| `<name>.top` | 力场默认项、`#include` 与 `[system]` |
| `<name>.gro` | 坐标 |
| `<name>.mdp` | 可运行的最小 `mdp` |

### 9.3 披露目录 `<name>.lft/`

仅在需要时出现：

| 文件 | 触发条件 |
|---|---|
| `notes.txt` | 存在任何警告（UFF 回退、通用参数、周期缠绕等） |
| `image_map.json` | 周期图 winding 或 `--debug` |
| `debug.txt` | `--debug` |
| `normalized_ir.json` | `--debug`，可再次作为结构 JSON 读入 |
| `work/` | QC 或外部工具需要落盘的中间文件 |

---

## 10. 全局配置与环境变量

### 10.1 `lft.config`

配置文件为 `key = value`，`#` 与 `;` 开头为注释。查找顺序：

1. `$LFT_CONFIG` 指定的文件
2. 仓库/安装根目录的 `lft.config`
3. Python 包目录内的 `lft.config`

不做项目级配置查找。优先级：**命令行 > 配置文件 > 内置默认**；工具路径环境变量优先于配置文件。

| 键 | 说明 |
|---|---|
| `output` | 默认输出后端 |
| `ff` | 默认力场策略 |
| `charge` | 默认电荷方法 |
| `qc` | 默认量化方法/基组 |
| `gmx` | GROMACS 可执行文件 |
| `lmp` | LAMMPS 可执行文件 |
| `moltemplate` | moltemplate 脚本 |
| `obabel` | OpenBabel 可执行文件 |
| `atomtype` | 外部 AmberTools `atomtype`（仅对照） |
| `orca` | ORCA 可执行文件（保留） |
| `param_dir` | 参考参数目录 |
| `external_atomtype` | 是否调用外部 `atomtype` 对照 |
| `am1bcc_external` | 是否调用外部 AM1-BCC 对照 |

### 10.2 环境变量

| 变量 | 作用 |
|---|---|
| `LFT_CONFIG` | 指定配置文件路径 |
| `LFT_GMX` | GROMACS 可执行文件 |
| `LFT_LMP` | LAMMPS 可执行文件 |
| `LFT_MOLTEMPLATE` | moltemplate 脚本 |
| `LFT_OBABEL` | OpenBabel 可执行文件 |
| `LFT_ATOMTYPE` | 外部 `atomtype` 可执行文件 |
| `LFT_ATOMTYPE_DEF` | 外部 `atomtype` 定义文件 |
| `LFT_ATOMTYPE_EXTERNAL` | 设为 `1` 时用外部 `atomtype` 做对照 |
| `LFT_AM1BCC_EXTERNAL` | 设为 `1` 时用外部 AM1-BCC 做对照 |
| `LFT_AM1BCC_OPTIMIZE` | AM1-BCC 几何优化：`yes` / `no`（默认）/ `auto`（≤200 原子） |
| `LFT_APP_ORCA` | ORCA 路径（兼容保留） |
| `LFT_PARAM_DIR` | 参考参数目录 |

---

## 11. 常见错误

| 现象 | 原因 | 处理 |
|---|---|---|
| `input charge mode requires charges` | `--charge input` 但无电荷 | 使用带电荷的 MOL2，或 `--charge charges.txt` |
| `periodic structures require explicit charges` | 有晶胞但未给电荷 | `--charge charges.txt` |
| `--charge value '...' is neither a known method nor an existing charge file` | 方法名拼错或文件不存在 | 检查拼写/路径 |
| `output already exists: ...` | 同名产物已存在 | `--force` 或换 `--outdir` |
| `unsupported structure format` | 后缀不在支持列表 | 转成 MOL2/PDB/CIF/XYZ 等 |
| `OpenBabel MMFF94 is unavailable` | 缺少 OpenBabel | 改用 `--charge input` 或 `am1-bcc` |
| 出现 UFF 回退 | 元素没有 GAFF 类型 | 正常；查看 `<name>.lft/notes.txt`，或用 `--manual-file` |
| 运行很慢 | 自动触发 RESP/AM1-BCC | 先用 `--charge mmff94` 或自带电荷验证管线 |

所有可预期错误返回非零退出码，并输出 `[ERROR_CODE] message` 与可选 `Hint:`。

---

## 12. 验证

安装后可做的最小自检：

```bash
lft --version
lft --help
```

源码检出中的完整测试：

```bash
python -m pytest -q
```

结构示例：

```bash
cd examples
lft -f water.mol2
```

---

## 13. 已知边界

- 经典键合参数库不完整；未匹配项使用确定性通用值并写入警告。
- UFF/共轭二面角使用占位参数，定量使用前应核对或被 `--manual-file` 覆盖。
- AM1-BCC 默认不做几何优化，结果依赖输入构象。
- 不提供构象搜索、体系组装、MD 采样或力场重优化。
- LAMMPS 输出固定为 `real` 单位、10 Å 直接截断；未启用 PPPM/PME。
- 周期体系不做 ESP/密度电荷拟合，必须显式给电荷。
- GROMACS 输出为非周期输入补兼容盒；部分 PBC 掩码会写成全周期盒并披露。

更多能力与验证记录见 [模块现状](PROJECT_STATUS.md) 与 [更新日志](CHANGELOG.md)。
