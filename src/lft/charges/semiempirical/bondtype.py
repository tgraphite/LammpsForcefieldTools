"""Bond-type assignment reproducing AmberTools ``bondtype -j full``.

This is a pure-Python port of ``AmberTools/src/antechamber/bondtype.c``
(including the ``ncsu-penalties`` valence-state enumeration used by
``process()``).  It reassigns bond orders (1/2/3) from atomic valences so that
e.g. an ester written with aromatic ``C(=O)-O`` bonds becomes ``C=O`` (2) +
``C-O`` (1), and aromatic ring carbons end up alternating 7/8.

The pipeline steps mirror ``bondtype.c``:

* ``assignav``  - per-atom penalty table (``APS.DAT``) with functional-group
  corrections (carbonyl C, phosphate P, sulfone S, N1-/N2+/NO2, ...).
* ring detection + aromatic classification (``ring.c``).
* ``process``   - enumerate valence states in ascending total-penalty order
  (``ncsu_penalties``) and run the first state that satisfies every atom.
* ``judgebt``   - assign concrete bond orders via ``jbo_induce`` /
  ``jbo_iteration``.
* ``finalize``  - map aromatic ring bonds 1->7 / 2->8 plus a few
  delocalization rewrites (types 6/9).

Only ``src/lft/charges/semiempirical/bondtype.py`` is created here; the rest of
the AM1-BCC pipeline is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Iterable, Sequence

PSCUTOFF = 10

# ---------------------------------------------------------------------------
# APS.DAT
# ---------------------------------------------------------------------------

_ELEMENT_Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57,
    "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64,
    "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
    "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
    "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85,
    "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92,
    "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99,
    "Fm": 100, "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105,
    "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111,
    "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117,
    "Og": 118,
}


@dataclass(frozen=True)
class ApsRow:
    """One parsed row of ``APS.DAT``.

    ``av`` holds the 8 atomic-penalty values indexed by valence (sum of bond
    orders); ``9999`` encodes the ``*`` (not allowed) entry.
    """

    prefix: str
    element: str
    atomicnum: int
    con: int  # -1 means "*" (any connectivity)
    av: tuple[int, ...]  # APS penalties indexed by valence 0..7


def _load_aps_table() -> tuple[ApsRow, ...]:
    text = Path(str(resources.files("lft.charges.data").joinpath("APS.DAT"))).read_text(
        encoding="ascii", errors="replace"
    )
    rows: list[ApsRow] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        toks = line.split()
        if len(toks) < 11:
            continue
        prefix = toks[0]
        try:
            atomicnum = int(toks[2])
        except ValueError:
            continue
        con_str = toks[3]
        av: list[int] = []
        ok = True
        for i in range(4, 12):
            tok = toks[i]
            if tok == "*":
                av.append(9999)
            else:
                try:
                    av.append(int(tok))
                except ValueError:
                    ok = False
                    break
        if not ok or len(av) != 8:
            continue
        rows.append(
            ApsRow(
                prefix=prefix,
                element=toks[1],
                atomicnum=atomicnum,
                con=-1 if con_str == "*" else int(con_str),
                av=tuple(av),
            )
        )
    return tuple(rows)


APS_TABLE: tuple[ApsRow, ...] = _load_aps_table()


# ---------------------------------------------------------------------------
# Small internal containers
# ---------------------------------------------------------------------------


class _Ring:
    __slots__ = ("num", "atomno")

    def __init__(self) -> None:
        self.num = 0
        self.atomno: list[int] = []


class _Arom:
    __slots__ = ("nr", "ar1", "ar2", "ar3", "ar4", "ar5", "rg")

    def __init__(self) -> None:
        self.nr = 1
        self.ar1 = 0
        self.ar2 = 0
        self.ar3 = 0
        self.ar4 = 0
        self.ar5 = 0
        self.rg = [0] * 12


class _Molecule:
    """Mutable state of the bond-type engine, mirroring bondtype.c globals."""

    def __init__(self, anums: list[int], connum: list[int], con: list[list[int]],
                 bi: list[int], bj: list[int], btype: list[int]) -> None:
        self.anums = anums
        self.connum = connum
        self.con = con  # padded to >= 6 entries, trailing -1
        self.bi = bi
        self.bj = bj
        self.btype = btype
        self.jflag = [-1] * len(btype)
        self.natom = len(anums)
        self.nbond = len(btype)

        self.av = [[9999] * 8 for _ in range(self.natom)]
        self.valence = [0] * self.natom
        self.va_best = [0] * self.natom
        self.apstype = [-1] * self.natom
        self.conjatom = [0] * self.natom

        self.ring: list[_Ring] = []
        self.arom = [_Arom() for _ in range(self.natom)]


# ---------------------------------------------------------------------------
# assignav: build the per-atom valence-penalty tables
# ---------------------------------------------------------------------------


def _assignav(mol: _Molecule) -> None:
    """Port of ``assignav()`` (bondtype.c lines 97-396)."""
    for i in range(mol.natom):
        an = mol.anums[i]
        cn = mol.connum[i]
        con = mol.con[i]
        idx1 = idx2 = idx3 = idx4 = idx5 = 0
        idx6 = idx7 = idx8 = idx9 = idx10 = 0

        # carbonyl C (CO2- / CS2- etc)
        if an == 6 and cn == 3:
            for j in range(3):
                nb = mol.anums[con[j]]
                if (nb == 8 or nb == 16) and mol.connum[con[j]] == 1:
                    idx1 += 1
        # phosphate P
        if an == 15 and cn == 4:
            for j in range(4):
                nb = mol.anums[con[j]]
                if (nb == 8 or nb == 16) and mol.connum[con[j]] == 1:
                    idx2 += 1
        # sulfone S
        if an == 16 and cn == 4:
            for j in range(4):
                nb = mol.anums[con[j]]
                if (nb == 8 or nb == 16) and mol.connum[con[j]] == 1:
                    idx3 += 1
        # N1- (terminal N bonded to a divalent N)
        if an == 7 and cn == 1:
            if mol.anums[con[0]] == 7 and mol.connum[con[0]] == 2:
                idx4 = 1
        # N2+ (divalent N with a terminal C/N neighbour)
        if an == 7 and cn == 2:
            for j in range(2):
                nb = mol.anums[con[j]]
                if (nb == 7 or nb == 6) and mol.connum[con[j]] == 1:
                    idx5 = 1
        # NO2 (trivalent N with >= 2 terminal O/S)
        if an == 7 and cn == 3:
            for j in range(3):
                nb = mol.anums[con[j]]
                if (nb == 8 or nb == 16) and mol.connum[con[j]] == 1:
                    idx6 += 1
        # N3+ (trivalent N with exactly one terminal O/S, not nitro)
        if an == 7 and cn == 3 and idx6 < 2:
            for j in range(3):
                nb = mol.anums[con[j]]
                if (nb == 8 or nb == 16) and mol.connum[con[j]] == 1:
                    idx7 = 1
        # O1- (terminal O on a nitro-like N)
        if an == 8 and cn == 1 and mol.anums[con[0]] == 7 and mol.connum[con[0]] == 3:
            nn = con[0]
            count = 0
            for j in range(3):
                nb = mol.anums[mol.con[nn][j]]
                if (nb == 8 or nb == 16) and mol.connum[mol.con[nn][j]] == 1:
                    count += 1
            idx8 = 1 if count <= 1 else 0
        # S1- (terminal S on a nitro-like N)
        if an == 16 and cn == 1 and mol.anums[con[0]] == 7 and mol.connum[con[0]] == 3:
            nn = con[0]
            count = 0
            for j in range(3):
                nb = mol.anums[mol.con[nn][j]]
                if (nb == 8 or nb == 16) and mol.connum[mol.con[nn][j]] == 1:
                    count += 1
            idx9 = 1 if count <= 1 else 0
        # C1+ (terminal C bonded to a divalent N, e.g. C#N-)
        if an == 6 and cn == 1:
            if mol.anums[con[0]] == 7 and mol.connum[con[0]] == 2:
                idx10 = 1

        mol.conjatom[i] = 0
        if idx1 >= 2 or idx2 >= 2:
            mol.conjatom[i] = 1
        if idx3 >= 3 or idx6 >= 2:
            mol.conjatom[i] = 1

        all_small = (
            idx1 < 2 and idx2 < 2 and idx3 < 2 and idx4 < 1 and idx5 < 1
            and idx6 < 2 and idx7 < 1 and idx8 < 1 and idx9 < 1 and idx10 < 1
        )
        special = None
        if idx1 >= 2:
            special = "APSCO2"
        elif idx2 == 2:
            special = "APSPO2"
        elif idx2 > 2:
            special = "APSPO3"
        elif idx3 == 2:
            special = "APSSO2"
        elif idx3 == 3:
            special = "APSSO3"
        elif idx3 == 4:
            special = "APSSO4"
        elif idx4 == 1:
            special = "APSN1-"
        elif idx5 == 1:
            special = "APSN2+"
        elif idx6 >= 2:
            special = "APSNO2"
        elif idx7 == 1:
            special = "APSN3+"
        elif idx8 == 1:
            special = "APSO1-"
        elif idx9 == 1:
            special = "APSS1-"
        elif idx10 == 1:
            special = "APSC1+"

        # Scan rows in file order; the plain "APS" case matches any row whose
        # atomic number / connectivity line up (mirrors strncmp("APS", ...)).
        for row in APS_TABLE:
            if all_small:
                pass
            elif special is not None and row.prefix != special:
                continue
            elif special is None:
                continue
            if row.atomicnum != an:
                continue
            if row.con != -1 and row.con != cn:
                continue
            mol.av[i] = list(row.av)
            break

    # derive best valence / apstype and the global maxaps (informational)
    maxaps = 0
    for i in range(mol.natom):
        best_found = False
        for j in range(8):
            if mol.av[i][j] == 0:
                best_found = True
                mol.apstype[i] = 1
                mol.valence[i] = j
                mol.va_best[i] = j
            if mol.av[i][j] > PSCUTOFF:
                continue
            if maxaps < mol.av[i][j]:
                maxaps = mol.av[i][j]
        _ = best_found  # atoms without any zero-penalty valence stay apstype -1


# ---------------------------------------------------------------------------
# Ring detection + aromatic classification (ring.c)
# ---------------------------------------------------------------------------


def _ring_capable(anums: list[int], connum: list[int], a: int) -> bool:
    an = anums[a]
    if an not in (6, 7, 8, 15, 16):
        return False
    if an == 6 and connum[a] <= 2:
        return False
    if an == 8 and connum[a] == 1:
        return False
    if an == 16 and connum[a] == 1:
        return False
    return True


def _ring_detect_cycle(
    mol: _Molecule,
    selectarom: list[int],
    selectnum: int,
    startnum: int,
) -> int:
    """Port of ``ring_detect_cycle()``; returns the updated ring count."""
    selectarom[selectnum] = startnum
    selectnum += 1
    ringnum = len(mol.ring)
    for i in range(4):
        start = mol.con[startnum][i]
        if start == -1:
            return ringnum
        if not _ring_capable(mol.anums, mol.connum, start):
            continue
        if start in selectarom[:selectnum]:
            continue
        if selectnum > 10:
            return ringnum
        for j in range(2, 10):
            if selectnum == j:
                for k in range(4):
                    first_nei = mol.con[selectarom[0]][k]
                    if first_nei == -1:
                        break
                    if first_nei == start:
                        r = _Ring()
                        r.atomno = list(selectarom[:j]) + [start]
                        r.num = j + 1
                        mol.ring.append(r)
                        ringnum += 1
                        break
        ringnum = _ring_detect_cycle(mol, selectarom, selectnum, start)
    return ringnum


def _purify(mol: _Molecule) -> None:
    """Port of ``purify()``: sort, dedupe, drop invalid rings, compact."""
    if not mol.ring:
        return
    for r in mol.ring:
        r.atomno.sort()
    for i in range(len(mol.ring)):
        for j in range(i + 1, len(mol.ring)):
            if mol.ring[i].num == mol.ring[j].num and mol.ring[i].num != 0:
                if mol.ring[i].atomno == mol.ring[j].atomno:
                    mol.ring[j].num = 0
    for r in mol.ring:
        for a in r.atomno:
            index = 0
            for k in range(r.num):
                if r.atomno[k] in mol.con[a]:
                    index += 1
            if index == 3:
                r.num = 0
                break
    mol.ring = [r for r in mol.ring if r.num != 0]


def _ringproperty(mol: _Molecule) -> None:
    for r in mol.ring:
        for a in r.atomno:
            mol.arom[a].rg[0] += 1
            mol.arom[a].rg[r.num] += 1


def _aromatic(mol: _Molecule, bondi: list[int], bondj: list[int],
              btype: list[int]) -> None:
    """Port of ``aromatic()`` (ring.c): classify rings into ar1..ar5."""
    initarom = [0] * mol.natom
    for i in range(mol.natom):
        an = mol.anums[i]
        cn = mol.connum[i]
        if an == 6:
            if cn == 3:
                initarom[i] = 2
            elif cn == 4:
                initarom[i] = -2
        elif an == 7:
            if cn <= 3:
                initarom[i] = 2
        elif an == 8:
            if cn == 2:
                initarom[i] = 1
        elif an == 15:
            if cn == 2:
                initarom[i] = 2
            elif cn == 3:
                initarom[i] = 1
            elif cn >= 4:
                initarom[i] = -1
        elif an == 16:
            if cn == 2:
                initarom[i] = 1
            elif cn >= 3:
                initarom[i] = -1

    for i, r in enumerate(mol.ring):
        tmpint = sum(initarom[a] for a in r.atomno)
        # pure aliphatic rings
        if tmpint == -2 * r.num:
            for a in r.atomno:
                mol.arom[a].ar5 += 1
            continue
        # rings containing an sp3 carbon / heteroatom
        continue_index = 0
        for a in r.atomno:
            if initarom[a] < 0:
                for b in r.atomno:
                    mol.arom[b].ar4 += 1
                continue_index = 1
                break
        if continue_index:
            continue
        # planar rings formed "outside" double bonds
        if r.num <= tmpint <= 2 * r.num:
            hit = False
            for k in range(mol.nbond):
                index = 0
                for a in r.atomno:
                    if bondi[k] == a and mol.arom[bondj[k]].rg[0] == 0:
                        index += 1
                for a in r.atomno:
                    if bondj[k] == a and mol.arom[bondi[k]].rg[0] == 0:
                        index += 1
                if index == 1 and btype[k] in (2, 8):
                    for b in r.atomno:
                        mol.arom[b].ar3 += 1
                    hit = True
                    break
            if hit:
                continue
        # pure aromatic 6-membered rings
        if tmpint == 12 and r.num == 6:
            index = 0
            for a in r.atomno:
                if mol.anums[a] in (7, 15):
                    index0 = 0
                    for k in range(mol.nbond):
                        if bondi[k] == a and btype[k] in (2, 8, 10):
                            index0 = 1
                        if bondj[k] == a and btype[k] in (2, 8, 10):
                            index0 = 1
                    if index0 == 0:
                        index = 1
            if index == 0:
                for a in r.atomno:
                    mol.arom[a].ar1 += 1
                continue
        # other planar rings
        if tmpint >= r.num + 3:
            for a in r.atomno:
                mol.arom[a].ar2 += 1
            continue
        # remaining rings
        for a in r.atomno:
            mol.arom[a].ar4 += 1


def _ringdetect(mol: _Molecule) -> None:
    """Port of ``ringdetect(..., index=1)``: find rings + arom flags."""
    for i in range(mol.natom):
        if not _ring_capable(mol.anums, mol.connum, i):
            continue
        selectarom = [-1] * mol.natom
        _ring_detect_cycle(mol, selectarom, 0, i)
        _purify(mol)
    _purify(mol)
    _ringproperty(mol)
    _aromatic(mol, mol.bi, mol.bj, mol.btype)

    # am1-bcc: 5-membered rings in indole-like systems are not aromatic
    for i in range(mol.natom):
        if mol.arom[i].rg[5] >= 1 and mol.arom[i].rg[6] >= 1:
            if (mol.arom[i].ar1 + mol.arom[i].ar2) >= 2:
                for r in mol.ring:
                    if r.num == 5 and i in r.atomno:
                        for a in r.atomno:
                            mol.arom[a].ar2 -= 1


# ---------------------------------------------------------------------------
# Integer-partition and combination iterators (ncsu-penalties.c)
# ---------------------------------------------------------------------------


def _combination_next(n: int, k: int, c: list[int]) -> list[int] | None:
    """Lexicographic k-combinations of 0..n-1, faithful to combination.c."""
    if c[0] == n:
        for i in range(k):
            c[i] = i
        return c
    r = k - 1
    nn = n - 1
    mark = k
    while True:
        if c[r] == nn:
            if r != 0:
                mark = r - 1  # "mark = --r; ++r"
            else:
                c[0] = n
                return None
        else:
            if mark < k:
                nm = c[mark] + 1
                for i in range(mark, k):
                    c[i] = nm
                    nm += 1
                return c
            for i in range(n):
                if c[r] == i:
                    c[r] = i + 1
                    return c
        # for (;; --r, --n) runs after every iteration, including `continue`
        r -= 1
        nn -= 1


def _int_part_next(N: int, K: int, c: list[int]) -> list[int] | None:
    """Non-negative solutions of c[0]*1 + ... + c[K-1]*K = N (Stockmal 1962)."""
    if K == 1 and c[0] == N:
        c[0] = N + 1
        return None
    if c[0] == N + 1:
        # first partition: N*1 + 0*2 + ... + 0*K
        c[0] = N
        j = K
        for i in range(1, j):
            c[i] = 0
        return c
    j = 1
    a = c[0]
    while True:
        if a <= j:
            if j + 1 == K:
                c[0] = N + 1
                return None
            a += (j + 1) * c[j]
            j += 1
            continue
        c[j] += 1
        c[0] = a - j - 1
        for i in range(1, j):
            c[i] = 0
        return c


def _ncsu_penalties(N: int, sizes: Sequence[int], goal: int,
                    callback) -> None:
    """Port of ``ncsu_penalties()``: enumerate (set_id, element_id) pairs.

    ``sizes[i]`` is the number of points with penalty ``i + 1``; callback is
    invoked for each combination of points whose penalties sum to ``goal``.
    """
    c = [0] * N
    c[0] = N + 1  # sentinel, as int_part_setup does
    combos: list[list[int]] = []
    for _ in range(N):
        combos.append([])
    for tpl in iter(lambda: _int_part_next(goal, N, c), None):
        if any(tpl[i] > sizes[i] for i in range(goal)):
            continue
        last = 0
        active: list[tuple[int, list[int]]] = []
        for i in range(N):
            if tpl[i] != 0:
                comb = [0] * tpl[i]
                comb[0] = sizes[i]  # sentinel, as combination_setup does
                _combination_next(sizes[i], tpl[i], comb)
                active.append((i, comb))
                last = i
        if not active:
            continue
        while True:
            pairs: list[tuple[int, int]] = []
            for i, comb in active:
                for e in comb:
                    pairs.append((i, e))
            callback(pairs)
            # advance to next Cartesian-product element
            advanced = False
            for pos, (i, comb) in enumerate(active):
                if _combination_next(sizes[i], tpl[i], comb) is not None:
                    advanced = True
                    break
                if i == last:
                    break
                _combination_next(sizes[i], tpl[i], comb)
            if not advanced and i == last and comb[0] == sizes[i]:
                break


# ---------------------------------------------------------------------------
# Valence-state search (process / judgebt / jbo_*)
# ---------------------------------------------------------------------------


class _ResidueState:
    """Per-residue working state (the atom2/bond2 machinery in main())."""

    def __init__(self, mol: _Molecule) -> None:
        self.mol = mol
        self.anums = list(mol.anums)
        self.connum = list(mol.connum)
        self.con = [list(c) for c in mol.con]
        self.bi = list(mol.bi)
        self.bj = list(mol.bj)
        self.btype = list(mol.btype)
        self.jflag = list(mol.jflag)
        self.type2 = list(range(mol.nbond))
        self.av = [list(a) for a in mol.av]
        self.valence = list(mol.valence)
        self.va_best = list(mol.va_best)
        self.apstype = list(mol.apstype)
        self.natom = mol.natom
        self.nbond = mol.nbond

        self.con_num = [0] * self.natom
        self.ind_valence = [0] * self.natom
        self.ind_bondnum = [0] * self.natom
        self.bondindex = [0] * self.nbond

        self.va_num: list[int] = []
        self.va_penalty: list[int] = []
        self.va_addr: list[list[int]] = []
        self.va_valence: list[list[int]] = []


def _jbo_score(st: _ResidueState) -> int:
    violation = 0
    for i in range(st.natom):
        if st.apstype[i] == -1:
            continue
        if st.ind_bondnum[i] == 0 and st.ind_valence[i] != 0:
            violation += 1
        if st.ind_bondnum[i] > st.ind_valence[i]:
            violation += 1
    return violation


def _jbo_induce(st: _ResidueState) -> None:
    index = 1
    while index > 0:
        index = 0
        num = 0
        for i in range(st.natom):
            if (st.ind_bondnum[i] == 1 and st.ind_valence[i] > 0) or (
                st.ind_bondnum[i] == st.ind_valence[i] and st.ind_valence[i] != 0
            ):
                for j in range(st.connum[i]):
                    at1 = i
                    at2 = st.con[i][j]
                    for k in range(st.nbond):
                        if st.bondindex[k] == 1:
                            continue
                        if (st.bi[k] == at1 and st.bj[k] == at2) or (
                            st.bi[k] == at2 and st.bj[k] == at1
                        ):
                            tmpint = st.ind_valence[i] // st.ind_bondnum[i]
                            st.btype[k] = tmpint
                            st.bondindex[k] = 1
                            st.ind_valence[at1] -= tmpint
                            st.ind_valence[at2] -= tmpint
                            st.ind_bondnum[at1] -= 1
                            st.ind_bondnum[at2] -= 1
                            num += 1
                            break
        if num != 0:
            index = 1


def _jbo_iteration(st: _ResidueState) -> int:
    while_flag = 1
    score = 0
    while while_flag == 1:
        bondtype_bak = list(st.btype)
        bondindex_bak = list(st.bondindex)
        valence_bak = list(st.ind_valence)
        connum_bak = list(st.ind_bondnum)
        for i in range(st.nbond):
            if st.bondindex[i] == -1:
                at1 = st.bi[i]
                at2 = st.bj[i]

                # assume single first, then double, then triple
                st.ind_valence[at1] -= 1
                st.ind_valence[at2] -= 1
                st.ind_bondnum[at1] -= 1
                st.ind_bondnum[at2] -= 1
                st.btype[i] = 1
                st.bondindex[i] = 1
                _jbo_induce(st)
                score = _jbo_score(st)
                flag = sum(1 for t in st.btype if t == -1)
                if score == 0 and flag == 0:
                    return score
                if score > 0:
                    st.ind_valence = list(valence_bak)
                    st.ind_bondnum = list(connum_bak)
                    st.btype = list(bondtype_bak)
                    st.bondindex = list(bondindex_bak)

                    st.ind_valence[at1] -= 2
                    st.ind_valence[at2] -= 2
                    st.ind_bondnum[at1] -= 1
                    st.ind_bondnum[at2] -= 1
                    st.btype[i] = 2
                    st.bondindex[i] = 1
                    _jbo_induce(st)
                    score = _jbo_score(st)
                    flag = sum(1 for t in st.btype if t == -1)
                    if score == 0 and flag == 0:
                        return score
                    if score > 0:
                        st.ind_valence = list(valence_bak)
                        st.ind_bondnum = list(connum_bak)
                        st.btype = list(bondtype_bak)
                        st.bondindex = list(bondindex_bak)

                        st.ind_valence[at1] -= 3
                        st.ind_valence[at2] -= 3
                        st.ind_bondnum[at1] -= 1
                        st.ind_bondnum[at2] -= 1
                        st.btype[i] = 3
                        st.bondindex[i] = 1
                        _jbo_induce(st)
                        score = _jbo_score(st)
                        flag = sum(1 for t in st.btype if t == -1)
                        if score == 0 and flag == 0:
                            return score
                        if score > 0:
                            st.ind_valence = list(valence_bak)
                            st.ind_bondnum = list(connum_bak)
                            st.btype = list(bondtype_bak)
                            st.bondindex = list(bondindex_bak)
                            return score
        while_flag = 0
        for k in range(st.nbond):
            if st.btype[k] == -1:
                while_flag = 1
                break
    return score


def _judgebt(st: _ResidueState, mol: _Molecule) -> int:
    nstates = len(st.va_num)
    for si in range(-1, nstates):
        valence2 = list(st.va_best)
        if si != -1:
            for addr, val in zip(st.va_addr[si], st.va_valence[si]):
                valence2[addr - 1] = val
        for j in range(st.natom):
            st.con_num[j] = st.connum[j]
            st.ind_valence[j] = valence2[j]
            st.ind_bondnum[j] = st.con_num[j]
        for j in range(st.nbond):
            if st.jflag[j] == 1:
                st.bondindex[j] = 1
                st.con_num[st.bi[j]] -= 1
                st.con_num[st.bj[j]] -= 1
                st.ind_valence[st.bi[j]] -= st.btype[j]
                st.ind_valence[st.bj[j]] -= st.btype[j]
                st.ind_bondnum[st.bi[j]] = st.con_num[st.bi[j]]
                st.ind_bondnum[st.bj[j]] = st.con_num[st.bj[j]]
            else:
                st.btype[j] = -1
                st.bondindex[j] = -1
        _jbo_induce(st)
        fail_flag = 0
        for j in range(st.nbond):
            if st.bondindex[j] == -1:
                fail_flag = 1
                break
        if fail_flag == 1:
            fail_flag = _jbo_iteration(st)
        for j in range(st.natom):
            if st.ind_valence[j] != 0:
                fail_flag = 1
                break
        for j in range(st.natom):
            if st.ind_bondnum[j] != 0:
                fail_flag = 1
                break
        if fail_flag == 0:
            for j in range(st.nbond):
                if st.type2[j] != -1:
                    mol.btype[st.type2[j]] = st.btype[j]
            return 1
    return 0


def _process(st: _ResidueState, mol: _Molecule) -> None:
    """Port of ``process()``: penalty-ordered valence-state enumeration."""

    def va_arrays_filler(pairs: Iterable[tuple[int, int]]) -> None:
        addr_list: list[int] = []
        val_list: list[int] = []
        penalty = 0
        for set_id, elem_id in pairs:
            p = set_id + 1
            point = points[p][elem_id]
            addr = point >> 3
            val = point & 0x7
            addr_list.append(addr)
            val_list.append(val)
            penalty += p
        st.va_num.append(len(addr_list))
        st.va_addr.append(addr_list)
        st.va_valence.append(val_list)
        st.va_penalty.append(penalty)

    maxaps = 0
    for i in range(st.natom):
        for j in range(8):
            p = st.av[i][j]
            if p > PSCUTOFF:
                continue
            if maxaps < p:
                maxaps = p

    if maxaps <= 0:
        _judgebt(st, mol)
        return

    npoints = [0] * (maxaps + 1)
    for i in range(st.natom):
        for j in range(8):
            p = st.av[i][j]
            if 0 < p <= maxaps:
                npoints[p] += 1

    points: list[list[int]] = [None] * (maxaps + 1)  # type: ignore[list-item]
    points[0] = []
    for m in range(1, maxaps + 1):
        points[m] = [0] * npoints[m]
    ii = [0] * (maxaps + 1)
    for i in range(st.natom):
        for j in range(8):
            p = st.av[i][j]
            if 0 < p <= maxaps:
                addr = i + 1
                points[p][ii[p]] = (addr << 3) | (j & 0x7)
                ii[p] += 1

    for m in range(1, maxaps + 1):
        _ncsu_penalties(maxaps, npoints[1:], m, va_arrays_filler)

    _judgebt(st, mol)


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------


def _finalize(mol: _Molecule) -> None:
    """Port of ``finalize()`` (bondtype.c lines 718-901)."""
    for i in range(mol.nbond):
        if mol.btype[i] != 10:
            continue
        bondi = mol.bi[i]
        bondj = mol.bj[i]
        flag0 = 0
        for j in range(mol.nbond):
            if i == j:
                continue
            if (mol.bi[j] == bondi or mol.bj[j] == bondi
                    or mol.bi[j] == bondj or mol.bj[j] == bondj):
                if mol.btype[j] in (2, 8):
                    flag0 = 1
                    break
        if flag0 == 0:
            mol.btype[i] = 8
        else:
            mol.btype[i] = 7

    for i in range(mol.nbond):
        bondi = mol.bi[i]
        bondj = mol.bj[i]

        # part1: aromatic ring bonds -> 7/8
        if ((mol.arom[bondi].ar1 > 0 and mol.arom[bondj].ar1 > 0)
                or (mol.arom[bondi].ar2 > 0 and mol.arom[bondj].ar2 > 0)):
            for r in mol.ring:
                if r.num >= 7 or r.num <= 4:
                    continue
                flag = 0
                for a in r.atomno:
                    if mol.arom[a].ar1 <= 0 and mol.arom[a].ar2 <= 0:
                        flag = 1
                        break
                if flag == 1:
                    continue
                if bondi in r.atomno and bondj in r.atomno:
                    if mol.btype[i] == 1:
                        mol.btype[i] = 7
                    if mol.btype[i] == 2:
                        mol.btype[i] = 8
                    break

        # part2: delocalized C=O / C=S on conjugated atoms
        if mol.btype[i] == 2:
            if mol.conjatom[bondi] == 1:
                if mol.connum[bondj] == 1 and mol.anums[bondj] in (8, 16):
                    mol.btype[i] = 9
                    continue
            if mol.conjatom[bondj] == 1:
                if mol.connum[bondi] == 1 and mol.anums[bondi] in (8, 16):
                    mol.btype[i] = 9
                    continue

        # part3: N-O / N-S single/delocalized
        if (mol.connum[bondi] in (2, 3) and mol.anums[bondi] == 7
                and ((mol.connum[bondj] == 1 and mol.anums[bondj] == 8)
                     or (mol.connum[bondj] == 1 and mol.anums[bondj] == 16))):
            for j in range(6):
                if mol.con[bondi][j] < 0:
                    break
                if mol.con[bondi][j] == bondj:
                    continue
                if (mol.connum[mol.con[bondi][j]] == 1
                        and mol.anums[mol.con[bondi][j]] in (8, 16)):
                    mol.btype[i] = 6
                break
            if mol.btype[i] == 6:
                continue
        if (mol.connum[bondj] in (2, 3) and mol.anums[bondj] == 7
                and ((mol.connum[bondi] == 1 and mol.anums[bondi] == 8)
                     or (mol.connum[bondi] == 1 and mol.anums[bondi] == 16))):
            for j in range(6):
                if mol.con[bondj][j] < 0:
                    break
                if mol.con[bondj][j] == bondi:
                    continue
                if (mol.connum[mol.con[bondj][j]] == 1
                        and mol.anums[mol.con[bondj][j]] in (8, 16)):
                    mol.btype[i] = 6
                break
            mol.btype[i] = 6
            continue

        # part4: single bond to terminal O/S
        if mol.btype[i] == 1:
            if (mol.anums[bondi] == 8 and mol.connum[bondi] == 1) or (
                mol.anums[bondj] == 8 and mol.connum[bondj] == 1
            ) or (mol.anums[bondi] == 16 and mol.connum[bondi] == 1) or (
                mol.anums[bondj] == 16 and mol.connum[bondj] == 1
            ):
                mol.btype[i] = 9
                continue

    # part5: sulfone / phosphate terminal bonds
    for i in range(mol.natom):
        if mol.connum[i] == 3 and mol.anums[i] == 16:
            num = 0
            index = [0] * 6
            for j in range(3):
                bi = mol.con[i][j]
                if mol.connum[bi] == 1 and mol.anums[bi] in (8, 16):
                    num += 1
                    index[j] = 1
            if num == 2:
                for b in range(mol.nbond):
                    bi = mol.bi[b]
                    bj = mol.bj[b]
                    for k in range(3):
                        if (index[k] == 1
                                and ((bi == i and bj == mol.con[i][k])
                                     or (bj == i and bi == mol.con[i][k]))):
                            mol.btype[b] = 9
        if mol.connum[i] == 4 and mol.anums[i] == 15:
            num = 0
            index = [0] * 6
            for j in range(4):
                bi = mol.con[i][j]
                if mol.connum[bi] == 1 and mol.anums[bi] in (8, 16):
                    num += 1
                    index[j] = 1
            if num >= 2:
                for b in range(mol.nbond):
                    bi = mol.bi[b]
                    bj = mol.bj[b]
                    for k in range(4):
                        if (index[k] == 1
                                and ((bi == i and bj == mol.con[i][k])
                                     or (bj == i and bi == mol.con[i][k]))):
                            mol.btype[b] = 9


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assign_bond_types(
    atom_types: Sequence[str],
    elements: Sequence[str],
    bonds: Sequence[tuple[int, int, int]],
    formal_charges: Sequence[int] | None = None,
) -> tuple[list[int], dict]:
    """Reassign bond orders as ``bondtype -j full``.

    Parameters
    ----------
    atom_types:
        BCC atom-type names per atom (informational; the assignment only uses
        element and connectivity, matching ``bondtype.c``).
    elements:
        Element symbols per atom.
    bonds:
        ``(i, j, type)`` tuples; pairs may be 0- or 1-based.  Type is 1/2/3 for
        explicit single/double/triple bonds and 7 (or 8) for aromatic bonds,
        which are treated as "could become single or double".

    Returns
    -------
    (new_bond_types, metadata)
        ``new_bond_types[k]`` in 1/2/3/7/8/9 for bond ``k`` (same order as the
        input ``bonds``).
    """
    if len(elements) == 0:
        return [], {}

    # 1-based index detection
    flat = [x for i, j, _ in bonds for x in (i, j)]
    max_idx = max(flat) if flat else 0
    offset = 0 if max_idx < len(elements) else 1

    anums: list[int] = []
    unknown: list[str] = []
    for el in elements:
        z = _ELEMENT_Z.get(el)
        if z is None:
            unknown.append(el)
        anums.append(z if z is not None else 0)
    if unknown:
        raise ValueError(f"Unrecognized element symbols: {sorted(set(unknown))}")

    nb = len(bonds)
    con: list[list[int]] = [[] for _ in range(len(elements))]
    bi: list[int] = []
    bj: list[int] = []
    btype: list[int] = []
    for i, j, t in bonds:
        i0 = i - offset
        j0 = j - offset
        bi.append(i0)
        bj.append(j0)
        # mol2 semantics: aromatic (7) -> 10, aromatic double (8) -> 2
        if t == 7:
            btype.append(10)
        elif t == 8:
            btype.append(2)
        else:
            btype.append(t)
        con[i0].append(j0)
        con[j0].append(i0)

    connum = [len(c) for c in con]
    con_pad = [c + [-1] * (6 - len(c)) for c in con]

    mol = _Molecule(anums, connum, con_pad, bi, bj, btype)
    _assignav(mol)
    _ringdetect(mol)

    # Single-residue handling (whole molecule).  The residue machinery only
    # matters for multi-residue inputs with inter-residue cap-H replacement,
    # which the public contract does not exercise.
    st = _ResidueState(mol)

    for j in range(nb):
        if st.apstype[st.bi[j]] == -1 or st.apstype[st.bj[j]] == -1:
            st.jflag[j] = 1
        if st.jflag[j] == 1:
            if st.btype[j] == 7:
                st.btype[j] = 1
            if st.btype[j] == 8:
                st.btype[j] = 2

    for j in range(st.natom):
        tvalence = 0
        frozen = 0
        for k in range(nb):
            if st.bi[k] == j or st.bj[k] == j:
                tvalence += st.btype[k]
                if st.jflag[k] == 1:
                    frozen += 1
        if st.apstype[j] == -1:
            st.av[j][tvalence] = 0
            st.valence[j] = tvalence
            st.va_best[j] = tvalence
        elif st.connum[j] == frozen:
            st.av[j] = [9999] * 8
            st.av[j][tvalence] = 0
            st.valence[j] = tvalence
            st.va_best[j] = tvalence

    _process(st, mol)
    _finalize(mol)

    out_types = list(mol.btype)
    metadata: dict = {
        "valence": list(st.valence),
        "va_best": list(st.va_best),
        "apstype": list(st.apstype),
        "n_valence_states_tried": len(st.va_num),
        "ring_atoms": [
            [a + 1 for a in r.atomno] for r in mol.ring if r.num != 0
        ],
    }
    return out_types, metadata