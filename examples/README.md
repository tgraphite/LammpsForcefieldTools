# 示例

## water.mol2

带 TIP3P 风格点电荷的水分子，用于演示最小 LAMMPS/GROMACS 输出：

```bash
cd examples
lft -f water.mol2
lft -f water.mol2 --output gromacs --outdir water_gromacs
```

LAMMPS 产物可直接运行：

```bash
lmp -in water.in
```

由于 MOL2 自带电荷且采用 `USER_CHARGES`，管线会自动走 `input` 电荷模式，不需要量化计算。
