"""Pure-Python port of AmberTools ``atomtype``.

This module reimplements ``AmberTools/src/antechamber/atomtype.c`` (GAFF/GFF
atom typing) entirely in Python so that no external ``atomtype`` binary is
required.  It reproduces the AmberTools 26 behaviour: DEF-rule interpretation,
ring/aromaticity detection, chemical-environment matching and post-adjustment.

The reference semantics are taken from the AmberTools 26 source tree
(``dat/antechamber/ATOMTYPE_GFF.DEF`` and ``src/antechamber/atomtype.c``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..domain.structure import Structure
from ..errors import ParameterError

# ---------------------------------------------------------------------------
# Atom / bond representation mirroring atomtype.c / atom.h
# ---------------------------------------------------------------------------


@dataclass
class _Atom:
    name: str
    element: str
    atomicnum: int
    connum: int
    con: list[int] = field(default_factory=list)  # neighbours (0-based), -1 padded
    ewd: int = 0
    saturate: int = -1
    improper: int = 0
    ambername: str = ""
    aa: str = "MOL"
    type2: int = 0
    type3: int = 0


@dataclass
class _Bond:
    bondi: int
    bondj: int
    type: int


@dataclass
class _Arom:
    nr: int = 1
    ar1: int = 0
    ar2: int = 0
    ar3: int = 0
    ar4: int = 0
    ar5: int = 0
    rg: list[int] = field(default_factory=lambda: [0] * 12)


@dataclass
class _ChemEnv:
    atconnum: int = 0
    apindex: int = 0
    ap: str = ""
    atname: str = ""
    cesname: str = ""


@dataclass
class _WildAtom:
    num: int
    atomicnum: list[int] = field(default_factory=list)
    connum: list[int] = field(default_factory=list)
    wildname: str = ""


@dataclass
class _Rule:
    type_name: str
    resname: str
    atomicnum: str
    connum: str
    nh: str
    ewg: str
    prop: str
    env: str
    ces_bond: str


# ---------------------------------------------------------------------------
# atomicnum() from common.c -- element symbol inferred from the atom NAME
# ---------------------------------------------------------------------------

# name[0] -> (name[1] test -> (atomicnum, element))
# The port mirrors the exact case-sensitive dispatch in common.c:atomicnum().


def _atomicnum_from_name(name: str) -> tuple[int, str]:
    """Return (atomicnum, element) for an atom name, exactly like common.c.

    The C code dispatches on the first two characters of the atom name; this
    is intentionally *name*-based (not element-based) to match the oracle.
    """
    if not name:
        raise ValueError("empty atom name")
    c0 = name[0]
    c1 = name[1] if len(name) > 1 else ""

    def alpha(ch: str) -> bool:
        return ch.isascii() and ch.isalpha()

    def lower(ch: str) -> bool:
        return ch.isascii() and ch.islower()

    # Each branch returns (atomicnum, element) or None to continue.
    if c0 == "A":
        table = {
            "c": (89, "Ac"),
            "g": (47, "Ag"),
            "l": (13, "Al"),
            "m": (95, "Am"),
            "r": (18, "Ar"),
            "s": (33, "As"),
            "t": (85, "At"),
            "u": (79, "Au"),
        }
        val = table.get(c1)
        if val is not None:
            return val
    elif c0 == "B":
        if c1 in ("r", "R"):
            return (35, "Br")
        if not alpha(c1):
            return (5, "B")
        table = {
            "a": (56, "Ba"),
            "e": (4, "Be"),
            "h": (107, "Bh"),
            "i": (83, "Bi"),
            "k": (97, "Bk"),
        }
        val = table.get(c1)
        if val is not None:
            return val
    elif c0 == "C":
        if c1 in ("l", "L"):
            return (17, "Cl")
        if not lower(c1):
            return (6, "C")
        table = {
            "a": (20, "Ca"),
            "d": (48, "Cd"),
            "e": (58, "Ce"),
            "f": (98, "Cf"),
            "m": (96, "Cm"),
            "o": (27, "Co"),
            "r": (24, "Cr"),
            "s": (55, "Cs"),
            "u": (29, "Cu"),
        }
        val = table.get(c1)
        if val is not None:
            return val
    elif c0 == "D":
        if not alpha(c1):
            return (1, "D")
        table = {"b": (105, "Db"), "s": (110, "Ds"), "y": (66, "Dy")}
        val = table.get(c1)
        if val is not None:
            return val
    elif c0 == "E":
        table = {"P": (0, "EP"), "r": (68, "Er"), "s": (99, "Es"), "u": (63, "Eu")}
        val = table.get(c1)
        if val is not None:
            return val
    elif c0 == "F":
        if not lower(c1):
            return (9, "F")
        table = {"e": (26, "Fe"), "m": (100, "Fm"), "r": (87, "Fr")}
        val = table.get(c1)
        if val is not None:
            return val
    elif c0 == "G":
        table = {"a": (31, "Ga"), "d": (64, "Gd"), "e": (32, "Ge")}
        val = table.get(c1)
        if val is not None:
            return val
    elif c0 == "H":
        if not lower(c1):
            return (1, "H")
        table = {
            "e": (2, "He"),
            "f": (72, "Hf"),
            "g": (80, "Hg"),
            "o": (67, "Ho"),
            "s": (108, "Hs"),
        }
        val = table.get(c1)
        if val is not None:
            return val
    elif c0 == "I":
        if not alpha(c1):
            return (53, "I")
        table = {"n": (49, "In"), "r": (77, "Ir")}
        val = table.get(c1)
        if val is not None:
            return val
    elif c0 == "K":
        if not alpha(c1):
            return (19, "K")
        if c1 == "r":
            return (36, "Kr")
    elif c0 == "l":
        if c1 == "p":
            return (0, "lp")
    elif c0 == "L":
        table = {
            "i": (3, "Li"),
            "a": (57, "La"),
            "r": (103, "Lr"),
            "u": (71, "Lu"),
            "P": (0, "LP"),
        }
        val = table.get(c1)
        if val is not None:
            return val
    elif c0 == "M":
        table = {
            "n": (25, "Mn"),
            "g": (12, "Mg"),
            "d": (101, "Md"),
            "o": (42, "Mo"),
            "t": (109, "Mt"),
        }
        val = table.get(c1)
        if val is not None:
            return val
    elif c0 == "N":
        if not lower(c1):
            return (7, "N")
        table = {
            "i": (28, "Ni"),
            "a": (11, "Na"),
            "b": (41, "Nb"),
            "d": (60, "Nd"),
            "e": (10, "Ne"),
            "o": (102, "No"),
            "p": (93, "Np"),
        }
        val = table.get(c1)
        if val is not None:
            return val
    elif c0 == "O":
        if not lower(c1):
            return (8, "O")
        if c1 == "s":
            return (76, "Os")
    elif c0 == "P":
        if not lower(c1):
            return (15, "P")
        table = {
            "d": (46, "Pd"),
            "t": (78, "Pt"),
            "b": (82, "Pb"),
            "a": (91, "Pa"),
            "m": (61, "Pm"),
            "o": (84, "Po"),
            "r": (59, "Pr"),
            "u": (94, "Pu"),
        }
        val = table.get(c1)
        if val is not None:
            return val
    elif c0 == "R":
        table = {
            "u": (44, "Ru"),
            "h": (45, "Rh"),
            "a": (88, "Ra"),
            "b": (37, "Rb"),
            "e": (75, "Re"),
            "f": (104, "Rf"),
            "n": (86, "Rn"),
        }
        val = table.get(c1)
        if val is not None:
            return val
    elif c0 == "S":
        if not lower(c1):
            return (16, "S")
        if c1 in ("i", "I"):
            return (14, "Si")
        table = {
            "c": (21, "Sc"),
            "e": (34, "Se"),
            "r": (38, "Sr"),
            "b": (51, "Sb"),
            "g": (106, "Sg"),
            "m": (62, "Sm"),
            "n": (50, "Sn"),
        }
        val = table.get(c1)
        if val is not None:
            return val
    elif c0 == "T":
        if not alpha(c1):
            return (1, "T")
        table = {
            "i": (22, "Ti"),
            "l": (81, "Tl"),
            "a": (73, "Ta"),
            "b": (65, "Tb"),
            "c": (43, "Tc"),
            "e": (52, "Te"),
            "h": (90, "Th"),
            "m": (69, "Tm"),
        }
        val = table.get(c1)
        if val is not None:
            return val
    elif c0 == "U":
        if not alpha(c1):
            return (92, "U")
    elif c0 == "V":
        if not alpha(c1):
            return (23, "V")
    elif c0 == "W":
        if not alpha(c1):
            return (74, "W")
    elif c0 == "X":
        if c1 == "e":
            return (54, "Xe")
    elif c0 == "Y":
        if not alpha(c1):
            return (39, "Y")
        if c1 == "b":
            return (70, "Yb")
    elif c0 == "Z":
        if c1 == "n":
            return (30, "Zn")
        if c1 == "r":
            return (40, "Zr")
    raise ValueError(f"Unrecognized case-sensitive atomic symbol ({name}).")


def _atomicnum_from_element(element: str) -> tuple[int, str]:
    """Element-symbol -> (atomicnum, symbol) via the common.c name dispatch."""
    anum, elem = _atomicnum_from_name(element)
    return anum, elem


# ---------------------------------------------------------------------------
# bond type bookkeeping (bondinfo in atomtype.c)
# ---------------------------------------------------------------------------


def _bondinfo(bonds: list[_Bond], natoms: int) -> dict[str, list[int]]:
    sb = [0] * natoms
    big_sb = [0] * natoms
    db = [0] * natoms
    big_db = [0] * natoms
    tb = [0] * natoms
    big_tb = [0] * natoms
    ab = [0] * natoms
    dl = [0] * natoms
    for b in bonds:
        bt = b.type
        if bt == 1:
            sb[b.bondi] += 1
            sb[b.bondj] += 1
            big_sb[b.bondi] += 1
            big_sb[b.bondj] += 1
        elif bt == 2:
            db[b.bondi] += 1
            db[b.bondj] += 1
            big_db[b.bondi] += 1
            big_db[b.bondj] += 1
        elif bt == 3:
            tb[b.bondi] += 1
            tb[b.bondj] += 1
            big_tb[b.bondi] += 1
            big_tb[b.bondj] += 1
        elif bt == 7:
            ab[b.bondi] += 1
            ab[b.bondj] += 1
            sb[b.bondi] += 1
            sb[b.bondj] += 1
        elif bt == 8:
            ab[b.bondi] += 1
            ab[b.bondj] += 1
            db[b.bondi] += 1
            db[b.bondj] += 1
        elif bt == 9:
            sb[b.bondi] += 1
            sb[b.bondj] += 1
            big_sb[b.bondi] += 1
            big_sb[b.bondj] += 1
            dl[b.bondi] += 1
            dl[b.bondj] += 1
        elif bt == 10:
            ab[b.bondi] += 1
            ab[b.bondj] += 1
    return {
        "sb": sb,
        "SB": big_sb,
        "db": db,
        "DB": big_db,
        "tb": tb,
        "TB": big_tb,
        "AB": ab,
        "DL": dl,
    }


def _jbond(bonds: list[_Bond], id1: int, id2: int, typ: int) -> int:
    for b in bonds:
        if (b.bondi == id1 and b.bondj == id2) or (b.bondi == id2 and b.bondj == id1):
            bt = b.type
            if typ == 1 and bt in (1, 7, 9, 10):
                return 1
            if typ == -1 and bt in (1, 9):
                return 1
            if typ == 2 and bt in (2, 8, 9, 10):
                return 1
            if typ == -2 and bt in (2, 9):
                return 1
            if typ == 3 and bt == 3:
                return 1
            if typ == -3 and bt == 3:
                return 1
            if typ == 4 and bt in (7, 8, 10):
                return 1
            if typ == 9 and bt == 9:
                return 1
            break
    return 0


# ---------------------------------------------------------------------------
# ring detection, aromaticity and EWD (ring.c)
# ---------------------------------------------------------------------------


def _ring_detect_cycle(
    atoms: list[_Atom],
    selectarom: list[int],
    selectnum: int,
    startnum: int,
    rings: list[list[int]],
    run_flag: int,
) -> int:
    selectarom[selectnum] = startnum
    selectnum += 1
    for i in range(4):
        if atoms[startnum].connum <= i:
            return 0
        start = atoms[startnum].con[i]
        if atoms[start].atomicnum not in (6, 7, 8, 16, 15):
            continue
        if atoms[start].atomicnum == 6 and atoms[start].connum <= 2:
            continue
        if atoms[start].atomicnum == 8 and atoms[start].connum == 1:
            continue
        if atoms[start].atomicnum == 16 and atoms[start].connum == 1:
            continue
        if start in selectarom[:selectnum]:
            continue
        if selectnum > 10:
            return 0
        for j in range(2, 10):
            if selectnum == j:
                for k in range(4):
                    if atoms[selectarom[0]].connum <= k:
                        break
                    if atoms[selectarom[0]].con[k] == start:
                        if run_flag == 1:
                            rings.append(selectarom[:j] + [start])
                        break
        _ring_detect_cycle(atoms, selectarom, selectnum, start, rings, run_flag)
    return 0


def _purify(atoms: list[_Atom], rings: list[list[int]]) -> None:
    if not rings:
        return
    # sort atom ids within each ring
    for r in rings:
        r.sort()
    # Keep the first occurrence, preserving the C discovery order.
    seen: set[tuple[int, ...]] = set()
    unique = []
    for ring in rings:
        key = tuple(ring)
        if key not in seen:
            unique.append(ring)
            seen.add(key)
    rings[:] = unique
    # drop "false" rings: any atom with 3 in-ring neighbours
    kept = []
    for r in rings:
        bad = False
        for a in r:
            cnt = sum(1 for nb in atoms[a].con[: atoms[a].connum] if nb in r)
            if cnt == 3:
                bad = True
                break
        if not bad:
            kept.append(r)
    rings[:] = kept


def _ringproperty(rings: list[list[int]], arom: list[_Arom]) -> None:
    for r in rings:
        for a in r:
            arom[a].rg[0] += 1
            arom[a].rg[len(r)] += 1


def _aromatic(
    atoms: list[_Atom],
    bonds: list[_Bond],
    rings: list[list[int]],
    arom: list[_Arom],
) -> None:
    initarom = [0] * len(atoms)
    for i, at in enumerate(atoms):
        arom[i].ar1 = arom[i].ar2 = arom[i].ar3 = arom[i].ar4 = arom[i].ar5 = 0
        at.saturate = -1
        z = at.atomicnum
        if z == 6:
            at.ewd = 0
            if at.connum == 3:
                initarom[i] = 2
            if at.connum == 4:
                initarom[i] = -2
                at.saturate = 1
        elif z == 7:
            at.ewd = 1
            if at.connum <= 3:
                initarom[i] = 2
            if at.connum >= 3:
                at.saturate = 1
        elif z == 8:
            at.ewd = 1
            if at.connum == 2:
                initarom[i] = 1
                at.saturate = 1
        elif z == 15:
            at.ewd = 0
            if at.connum == 2:
                initarom[i] = 2
            if at.connum == 3:
                initarom[i] = 1
            if at.connum >= 4:
                initarom[i] = -1
            if at.connum >= 3:
                at.saturate = 1
        elif z == 16:
            at.ewd = 1
            if at.connum == 2:
                initarom[i] = 1
            if at.connum >= 3:
                initarom[i] = -1
            if at.connum >= 3:
                at.saturate = 1
        elif z in (9, 17, 35, 53):
            at.ewd = 1
            at.saturate = 1
        elif z == 1:
            at.ewd = 0
            at.saturate = 1
        else:
            at.ewd = 0

    for r in rings:
        tmpint = sum(initarom[a] for a in r)
        if tmpint == -2 * len(r):  # pure aliphatic rings (sp3 carbon)
            for a in r:
                arom[a].ar5 += 1
            continue
        continue_index = False
        for a in r:
            if initarom[a] < 0:  # contains sp3 carbon
                for b in r:
                    arom[b].ar4 += 1
                continue_index = True
                break
        if continue_index:
            continue
        if len(r) <= tmpint <= 2 * len(r):  # planar ring w/ outside double bond
            for bd in bonds:
                index = 0
                for a in r:
                    if bd.bondi == a and arom[bd.bondj].rg[0] == 0:
                        index += 1
                    if bd.bondj == a and arom[bd.bondi].rg[0] == 0:
                        index += 1
                if index == 1 and bd.type in (2, 8):
                    for a in r:
                        arom[a].ar3 += 1
                    continue_index = True
                    break
            if continue_index:
                continue
        if tmpint == 12 and len(r) == 6:  # pure aromatic ring
            index = 0
            for a in r:
                if atoms[a].atomicnum in (7, 15):
                    tmpint1 = a
                    index0 = 0
                    for bd in bonds:
                        if bd.bondi == tmpint1 and bd.type in (8, 2, 10):
                            index0 = 1
                        if bd.bondj == tmpint1 and bd.type in (8, 2, 10):
                            index0 = 1
                    if index0 == 0:
                        index = 1
            if index == 0:
                for a in r:
                    arom[a].ar1 += 1
                continue
        if tmpint >= len(r) + 3:  # other planar rings
            for a in r:
                arom[a].ar2 += 1
            continue
        for a in r:  # the rest
            arom[a].ar4 += 1

    for i in range(len(atoms)):
        if (
            arom[i].ar1 > 0
            or arom[i].ar2 > 0
            or arom[i].ar3 > 0
            or arom[i].ar4 > 0
            or arom[i].ar5 > 0
        ):
            arom[i].nr = 0


def _ringdetect(
    atoms: list[_Atom], bonds: list[_Bond], arom: list[_Arom], bcc_ring: bool
) -> None:
    """Populate arom (rg, ar*) and atom ewd -- port of ring.c ringdetect()."""
    rings: list[list[int]] = []
    eligible = []
    for i, at in enumerate(atoms):
        z = at.atomicnum
        if z not in (6, 7, 8, 16, 15):
            continue
        if z == 6 and at.connum <= 2:
            continue
        if z == 8 and at.connum == 1:
            continue
        if z == 16 and at.connum == 1:
            continue
        eligible.append(i)

    for i in eligible:
        selectarom = [-1] * len(atoms)
        _ring_detect_cycle(atoms, selectarom, 0, i, rings, 1)
        _purify(atoms, rings)

    _ringproperty(rings, arom)
    _aromatic(atoms, bonds, rings, arom)

    # For AM1-BCC, a five-membered ring in an indole-like system is not aromatic.
    if bcc_ring:
        for i in range(len(atoms)):
            if (
                arom[i].rg[5] >= 1
                and arom[i].rg[6] >= 1
                and (arom[i].ar1 + arom[i].ar2) >= 2
            ):
                for r in rings:
                    if len(r) == 5 and i in r:
                        for a in r:
                            arom[a].ar2 -= 1
                break


# ---------------------------------------------------------------------------
# atomic property matching (apcheck in atomtype.c)
# ---------------------------------------------------------------------------


def _apcheck(
    atoms: list[_Atom],
    arom: list[_Arom],
    bondinfo: dict[str, list[int]],
    bonds: list[_Bond],
    atmid: int,
    preatmid: int,
    propstr: str,
) -> int:
    # strip '[' and split constraints on ',' and ']'; units are OR-separated by '.'
    s = propstr.replace("[", "")
    constraints = [c.split(".") for c in s.replace("]", ",").split(",") if c]
    a = atoms[atmid]
    ar = arom[atmid]
    bi = bondinfo
    for cons in constraints:
        suc = 0
        for unit in cons:
            # leading digits = count, rest = name
            k = 0
            while k < len(unit) and unit[k].isdigit():
                k += 1
            digits = unit[:k]
            name = unit[k:]
            propcount = int(digits) if digits else -1
            hit = False
            if name == "RG":
                hit = propcount == ar.rg[0] or (propcount == -1 and ar.rg[0] > 0)
            elif name == "RG3":
                hit = propcount == ar.rg[3] or (propcount == -1 and ar.rg[3] > 0)
            elif name == "RG4":
                hit = propcount == ar.rg[4] or (propcount == -1 and ar.rg[4] > 0)
            elif name == "RG5":
                hit = propcount == ar.rg[5] or (propcount == -1 and ar.rg[5] > 0)
            elif name == "RG6":
                hit = propcount == ar.rg[6] or (propcount == -1 and ar.rg[6] > 0)
            elif name == "RG7":
                hit = propcount == ar.rg[7] or (propcount == -1 and ar.rg[7] > 0)
            elif name == "RG8":
                hit = propcount == ar.rg[8] or (propcount == -1 and ar.rg[8] > 0)
            elif name == "RG9":
                hit = propcount == ar.rg[9] or (propcount == -1 and ar.rg[9] > 0)
            elif name == "RG10":
                hit = propcount == ar.rg[10] or (propcount == -1 and ar.rg[10] > 0)
            elif name == "NR":
                hit = propcount == ar.nr or (propcount == -1 and ar.nr > 0)
            elif name in ("AR1", "AR2", "AR3", "AR4", "AR5"):
                val = {
                    "AR1": ar.ar1,
                    "AR2": ar.ar2,
                    "AR3": ar.ar3,
                    "AR4": ar.ar4,
                    "AR5": ar.ar5,
                }[name]
                hit = propcount == val or (propcount == -1 and val > 0)
            elif name[:2] in ("SB", "sb", "DB", "db", "TB", "tb", "DL", "AB"):
                hit = _apcheck_bond(name, propcount, preatmid, atmid, a, bi, bonds)
            if hit:
                suc = 1
                break
        if suc == 0:
            return 0
    return 1


def _apcheck_bond(
    name: str,
    propcount: int,
    preatmid: int,
    atmid: int,
    a: _Atom,
    bi: dict[str, list[int]],
    bonds: list[_Bond],
) -> bool:
    key = name[:2]
    prime = name[2:].count("'")
    counts = bi[key]
    cnt = counts[atmid]
    if not (propcount == cnt or (propcount == -1 and cnt > 0)):
        return False
    if prime == 0:
        return True
    if preatmid == -1:
        return False
    if key == "SB":
        return _jbond(bonds, atmid, preatmid, -1) == (1 if prime == 1 else 0)
    if key == "sb":
        return _jbond(bonds, atmid, preatmid, 1) == (1 if prime == 1 else 0)
    if key == "DB":
        return _jbond(bonds, atmid, preatmid, -2) == (1 if prime == 1 else 0)
    if key == "db":
        return _jbond(bonds, atmid, preatmid, 2) == (1 if prime == 1 else 0)
    if key == "TB":
        return _jbond(bonds, atmid, preatmid, -3) == (1 if prime == 1 else 0)
    if key == "tb":
        return _jbond(bonds, atmid, preatmid, 3) == (1 if prime == 1 else 0)
    if key == "DL":
        return _jbond(bonds, atmid, preatmid, 9) == (1 if prime == 1 else 0)
    if key == "AB":
        return _jbond(bonds, atmid, preatmid, 4) == (1 if prime == 1 else 0)
    return False


# ---------------------------------------------------------------------------
# wildatom matching (wematch in atomtype.c)
# ---------------------------------------------------------------------------


def _wematch(
    atoms: list[_Atom], wildatoms: list[_WildAtom], name: str, atmid: int
) -> int:
    if not wildatoms:
        return 0
    for w in wildatoms:
        if name[:2] == w.wildname:
            for j in range(w.num):
                if atoms[atmid].atomicnum == w.atomicnum[j] and (
                    atoms[atmid].connum == w.connum[j] or w.connum[j] == 0
                ):
                    return 1
    return 0


# ---------------------------------------------------------------------------
# chemical environment matching (jatspecial / cematch / chain_check in atomtype.c)
# ---------------------------------------------------------------------------


class _EnvState:
    def __init__(self) -> None:
        self.ces: list[list[_ChemEnv]] = []
        self.ceslen: list[int] = []
        self.cesindex: list[int] = []
        self.schain: list[tuple[int, list[int]]] = []  # (chain_id, atom path)
        self.ces_bond_at1: list[str] = []
        self.ces_bond_at2: list[str] = []
        self.ces_bond_type: list[str] = []
        self.chain = 0


def _jatspecial(
    atoms: list[_Atom],
    arom: list[_Arom],
    bondinfo: dict[str, list[int]],
    bonds: list[_Bond],
    wildatoms: list[_WildAtom],
    caid: int,
    keyword: str,
    ces_bond_string: str,
) -> tuple[int, _EnvState]:
    layer = 0
    chain = 0
    maxbeed = 30
    atname = [""] * maxbeed
    atconnum = [0] * maxbeed
    apindex = [0] * maxbeed
    ap = [""] * maxbeed
    cesname = [""] * maxbeed
    ces: list[list[_ChemEnv]] = []
    ceslen: list[int] = []

    tmpapindex = 0
    tmpap = ""
    cesname_index = 0
    tmpcesname = ""
    cea_id = 1
    index0 = 0

    def push_chain(up_to: int) -> None:
        nonlocal chain
        while len(ces) <= chain:
            ces.append([])
        for j in range(up_to):
            e = _ChemEnv()
            e.atname = atname[j + 1]
            e.atconnum = atconnum[j + 1]
            e.apindex = apindex[j + 1]
            e.ap = ap[j + 1]
            e.cesname = cesname[j + 1]
            ces[chain].append(e)
        chain += 1
        ceslen.append(up_to)

    i = 0
    n = len(keyword)
    while i < n:
        ch = keyword[i]
        nxt = keyword[i + 1] if i + 1 < n else ""
        if tmpapindex == 0 and cesname_index == 0 and ch.isalpha() and nxt.isalpha():
            i += 1
            continue
        if ch == "(":
            layer += 1
            if layer >= maxbeed:
                raise ValueError("Too many beed levels in ces chain.")
            i += 1
            continue
        if ch == ")":
            layer -= 1
            if keyword[i - 1] != ")":
                push_chain(layer + 1)
            i += 1
            continue
        if tmpapindex == 0 and ch == "[":
            tmpapindex = 1
            tmpap = "["
            i += 1
            continue
        if tmpapindex == 1 and ch == "]":
            apindex[layer] = 1
            tmpap += "]"
            ap[layer] = tmpap
            tmpapindex = 0
            i += 1
            continue
        if tmpapindex == 1:
            tmpap += ch
            i += 1
            continue
        if cesname_index == 0 and ch == "<":
            tmpcesname = ""
            cesname_index = 1
            i += 1
            continue
        if cesname_index == 1 and ch == ">":
            cesname[layer] = tmpcesname
            cesname_index = 0
            i += 1
            continue
        if cesname_index == 1:
            tmpcesname += ch
            i += 1
            continue
        if ch == "," and keyword[i - 1] != ")":
            push_chain(layer)
            i += 1
            continue
        if ch.isalpha():
            index0 = 0
            atconnum[layer] = 0
            if keyword[i - 1].isalpha():
                atname[layer] = keyword[i - 1] + ch
            else:
                atname[layer] = ch
            cesname[layer] = f"cea_{cea_id}"
            cea_id += 1
            ap[layer] = ""
            apindex[layer] = 0
            i += 1
            continue
        if ch.isdigit():
            atconnum[layer] = int(ch)
            i += 1
            continue
        if index0 == 1:
            atconnum[layer] = 0
            index0 = 0
        i += 1

    state = _EnvState()
    state.ces = ces
    state.ceslen = ceslen
    state.chain = chain

    state.ces_bond_at1 = []
    state.ces_bond_at2 = []
    state.ces_bond_type = []
    if ces_bond_string and ces_bond_string != "&":
        for part in ces_bond_string.split(","):
            toks = part.split(":")
            if len(toks) == 3:
                state.ces_bond_at1.append(toks[0])
                state.ces_bond_at2.append(toks[1])
                state.ces_bond_type.append(toks[2])

    # cematch: DFS over the molecular graph starting at caid
    selectchain = [-1] * len(atoms)
    selectindex = [-1] * len(atoms)
    maxchain = max(ceslen) if ceslen else -1

    def cematch(selectnum: int, startnum: int) -> None:
        selectchain[selectnum] = startnum
        selectnum += 1
        selectindex[startnum] = 1
        for k in range(chain):
            if selectnum - 1 == ceslen[k]:
                index = 1
                for j in range(1, selectnum):
                    atm_j = atoms[selectchain[j]]
                    ce = ces[k][j - 1]
                    if atm_j.connum != ce.atconnum and ce.atconnum != 0:
                        index = 0
                        break
                    if ce.atname[:2] == "EW" and atm_j.ewd != 1:
                        index = 0
                        break
                    index1 = 1
                    if not atm_j.name or atm_j.name[0] != ce.atname[0]:
                        index1 = 0
                    if index1 == 1:
                        if len(ce.atname) >= 2 and atm_j.name[1:2] != ce.atname[1]:
                            index1 = 0
                        if len(atm_j.element) >= 2 and len(ce.atname) == 1:
                            index1 = 0
                    if (
                        index1 == 0
                        and _wematch(atoms, wildatoms, ce.atname, selectchain[j]) == 0
                    ):
                        index = 0
                        break
                    if ce.apindex == 1:
                        if j == 1:
                            ok = _apcheck(
                                atoms,
                                arom,
                                bondinfo,
                                bonds,
                                selectchain[j],
                                caid,
                                ce.ap,
                            )
                        else:
                            ok = _apcheck(
                                atoms,
                                arom,
                                bondinfo,
                                bonds,
                                selectchain[j],
                                selectchain[j - 1],
                                ce.ap,
                            )
                        if ok == 0:
                            index = 0
                            break
                if index == 1:
                    state.cesindex.append(k)
                    state.schain.append((k, list(selectchain[1:selectnum])))
        for i in range(6):
            if atoms[startnum].connum <= i:
                return
            start = atoms[startnum].con[i]
            if start in selectchain[:selectnum]:
                continue
            if selectnum > maxchain:
                return
            cematch(selectnum, start)

    cematch(0, caid)

    index = 1
    for k in range(chain):
        if state.cesindex.count(k) == 0:
            index = 0
            break
    if index == 1:
        index = _dccheck(atoms, state, caid, maxchain, bonds)
    return (index, state)


def _dccheck(
    atoms: list[_Atom],
    state: _EnvState,
    caid: int,
    maxchain: int,
    bonds: list[_Bond],
) -> int:
    chain = state.chain
    # collect candidate schain ids per chain
    by_chain: dict[int, list[int]] = {}
    for si, (k, path) in enumerate(state.schain):
        by_chain.setdefault(k, []).append(si)
    order = [k for k in range(chain)]
    assignment: dict[int, int] = {}

    def valid_assign() -> bool:
        # chain_check
        for ii in range(chain):
            for jj in range(ii + 1, chain):
                si = assignment[ii]
                sj = assignment[jj]
                if si == sj:
                    return False
                p1 = state.schain[si][1]
                p2 = state.schain[sj][1]
                m = min(len(p1), len(p2))
                if p1[:m] == p2[:m]:
                    return False
                for kk in range(m):
                    if p1[kk] == p2[kk]:
                        if state.ces[ii][kk].cesname != state.ces[jj][kk].cesname:
                            return False
                    else:
                        if state.ces[ii][kk].cesname == state.ces[jj][kk].cesname:
                            return False
        # ces bond constraints
        if state.ces_bond_at1:
            atomcesname = {a: "" for a in range(len(atoms))}
            for ii in range(chain):
                si = assignment[ii]
                path = state.schain[si][1]
                for kk, atom_id in enumerate(path):
                    atomcesname[atom_id] = state.ces[ii][kk].cesname
            atomcesname[caid] = "sa"
            for t in range(len(state.ces_bond_at1)):
                t1 = -1
                t2 = -1
                for a in range(len(atoms)):
                    if atomcesname[a] == state.ces_bond_at1[t]:
                        t1 = a
                    if atomcesname[a] == state.ces_bond_at2[t]:
                        t2 = a
                if t1 == t2 and t1 != -1:
                    return False
                if _cesbondcheck(bonds, state.ces_bond_type[t], t1, t2) == 0:
                    return False
        return True

    def dfs(depth: int) -> bool:
        if depth == chain:
            return valid_assign()
        k = order[depth]
        for si in by_chain.get(k, []):
            assignment[k] = si
            if dfs(depth + 1):
                return True
        return False

    return 1 if dfs(0) else 0


def _cesbondcheck(bonds: list[_Bond], btype: str, id1: int, id2: int) -> int:
    """Mirror the case-sensitive cesbondcheck table, including Any connectivity."""
    if id1 == -1 or id2 == -1:
        return 0
    accepted = {
        1: {"sb"},
        2: {"db"},
        3: {"tb"},
        7: {"sb", "SB", "ab", "AB"},
        8: {"db", "DB", "ab", "AB"},
        9: {"sb", "SB", "db", "DB"},
        10: {"ab", "AB"},
    }
    for bond in bonds:
        if {bond.bondi, bond.bondj} == {id1, id2} and (
            btype in {"any", "ANY", "Any"} or btype in accepted.get(bond.type, set())
        ):
            return 1
    return 0


# ---------------------------------------------------------------------------
# DEF rule parsing
# ---------------------------------------------------------------------------


def _assign_wildatom_atomicnum(name: str) -> tuple[int, int]:
    """Return (atomicnum, connum) for an element token in a WILDATOM line."""
    letters = "".join(ch for ch in name if ch.isalpha())
    digits = "".join(ch for ch in name if ch.isdigit())
    connum = int(digits) if digits else 0
    atomicnum, _ = _atomicnum_from_name(letters)
    return atomicnum, connum


def _parse_def(path: Path) -> tuple[list[_WildAtom], list[_Rule]]:
    wildatoms: list[_WildAtom] = []
    rules: list[_Rule] = []
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("WILDATOM"):
            tokens = line.split()
            w = _WildAtom(num=0, wildname=tokens[1])
            for tok in tokens[2:]:
                anum, cnum = _assign_wildatom_atomicnum(tok)
                w.atomicnum.append(anum)
                w.connum.append(cnum)
            w.num = len(w.atomicnum)
            wildatoms.append(w)
        elif line.startswith("ATD"):
            toks = line.split(None, 10)
            # toks: ATD type res atomnum connum nh ewg prop env [ces_bond]
            if len(toks) < 2:
                continue
            fields = [toks[1]]
            fields += [toks[i] if i < len(toks) else "" for i in range(2, 10)]
            while len(fields) < 9:
                fields.append("")
            rule = _Rule(
                fields[0],
                fields[1],
                fields[2],
                fields[3],
                fields[4],
                fields[5],
                fields[6],
                fields[7],
                fields[8],
            )
            rules.append(rule)
    return wildatoms, rules


def _resolve_def_file(fname: str) -> Path:
    """Locate a typing definition file in the package data directories."""
    from importlib import resources

    for pkg in ("lft.forcefields.data", "lft.charges.data"):
        try:
            cand = resources.files(pkg).joinpath(fname)
        except ModuleNotFoundError:
            continue
        if cand.is_file():
            return Path(str(cand))
    return Path(fname)


# ---------------------------------------------------------------------------
# post-adjustment (atadjust / cpadjust in atomtype.c)
# ---------------------------------------------------------------------------


def _atadjust(atoms: list[_Atom], bonds: list[_Bond]) -> None:
    groups = ("cc", "ce", "cg", "pc", "pe", "nc", "ne")
    index1 = [0] * len(atoms)
    index2 = [0] * len(atoms)
    index = 0
    num = 0
    for i, at in enumerate(atoms):
        if at.ambername in groups:
            index2[i] = 1
            if index == 0:
                index1[i] = 1
                index = 1
            num += 1
    if num == 0:
        return
    num -= 1
    while num > 0:
        num -= 1
        flag = 0
        for b in bonds:
            bi, bj = b.bondi, b.bondj
            if index2[bi] + index2[bj] != 2:
                continue
            if flag == 0 and index1[bi] == 0 and index1[bj] == 0:
                index1[bi] = 1
            if index1[bi] == 0 and index1[bj] != 0:
                flag = 1
                if b.type in (1, 7):
                    index1[bi] = index1[bj]
                if b.type in (2, 8, 3):
                    index1[bi] = -index1[bj]
            if index1[bj] == 0 and index1[bi] != 0:
                flag = 1
                if b.type in (1, 7):
                    index1[bj] = index1[bi]
                if b.type in (2, 8, 3):
                    index1[bj] = -index1[bi]
    rename = {
        "cc": "cd",
        "ce": "cf",
        "cg": "ch",
        "pc": "pd",
        "pe": "pf",
        "nc": "nd",
        "ne": "nf",
    }
    for i, at in enumerate(atoms):
        if index1[i] == -1 and at.ambername in rename:
            at.ambername = rename[at.ambername]


def _cpadjust(atoms: list[_Atom], bonds: list[_Bond]) -> None:
    index1 = [0] * len(atoms)
    index2 = [0] * len(atoms)
    index = 0
    num = 0
    for i, at in enumerate(atoms):
        if at.ambername == "cp":
            index2[i] = 1
            if index == 0:
                index1[i] = 1
                index = 1
            num += 1
    if num == 0:
        return
    num -= 1
    while num > 0:
        num -= 1
        for b in bonds:
            bi, bj = b.bondi, b.bondj
            if index2[bi] + index2[bj] != 2:
                continue
            if index1[bi] == 0 and index1[bj] != 0:
                index1[bi] = index1[bj] if b.type == 1 else -index1[bj]
            if index1[bj] == 0 and index1[bi] != 0:
                index1[bj] = index1[bi] if b.type == 1 else -index1[bi]
    for i, at in enumerate(atoms):
        if index1[i] == -1 and at.ambername == "cp":
            at.ambername = "cq"


def _improper(atoms: list[_Atom]) -> None:
    for i, at in enumerate(atoms):
        if at.atomicnum == 6 and at.connum == 3:
            at.improper = 1
        if (
            at.atomicnum == 7
            and at.connum == 3
            and at.ambername[:2]
            in ("ca", "cb", "cc", "cd", "ce", "cf", "cg", "ch", "cn", "cp", "cq")
        ):
            at.improper = 1
        if (
            at.ambername[:1] == "c"
            and len(at.ambername) >= 2
            and at.ambername[1] == " "
        ):
            pass


# ---------------------------------------------------------------------------
# main typing driver (jat in atomtype.c)
# ---------------------------------------------------------------------------

_AA = {
    "ALA",
    "GLY",
    "SER",
    "THR",
    "LEU",
    "ILE",
    "VAL",
    "ASN",
    "GLN",
    "ARG",
    "HID",
    "HIE",
    "HIP",
    "TRP",
    "PHE",
    "TYR",
    "GLU",
    "ASP",
    "LYS",
    "PRO",
    "CYS",
    "CYX",
    "MET",
    "ASH",
    "GLH",
    "LYH",
}
_NA = {"DADE", "RADE", "DTHY", "RURA", "DGUA", "RGUA", "DCYT", "RCYT"}


def _bond_type_from_order(order: float) -> int:
    # Canonical MOL2 uses lowercase ar: rmol2 maps it to type 10.
    if 1.3 <= float(order) <= 1.7:
        return 10
    value = round(float(order))
    if value not in {1, 2, 3, 7, 8}:
        raise ParameterError(f"unsupported GAFF bond order {order}")
    return {7: 1, 8: 2}.get(value, value)


def _build_graph(
    structure: Structure, *, mol2_semantics: bool = True
) -> tuple[list[_Atom], list[_Bond]]:
    atoms: list[_Atom] = []
    corrections = dict(
        line.split()[:2]
        for line in _resolve_def_file("CORR_NAME_TYPE.DAT").read_text().splitlines()
        if len(line.split()) >= 2
    )
    for atom in structure.atoms:
        # rmol2 truncates names, expands '*' from the input type, then main
        # calls atomicnum BEFORE adjustatomname. Never infer a different
        # element on an invalid name: the C oracle rejects it as well.
        name = atom.name[:5]
        if name.startswith("*"):
            name = atom.element.split(".")[0][:4]
        try:
            anum, elem = _atomicnum_from_name(name)
        except ValueError as exc:
            raise ParameterError(str(exc)) from exc
        name = name.replace(" ", "")
        prefix = corrections.get(atom.element)
        if prefix is not None:
            name = prefix + name[len(prefix) :]
        leading = len(name) - len(name.lstrip("0123456789"))
        if leading:
            name = name[leading:] + name[:leading] if leading < len(name) else ""
        atoms.append(
            _Atom(
                name=name,
                element=elem,
                atomicnum=anum,
                connum=0,
                ambername=atom.element,
                aa=atom.residue_name[:3],
            )
        )
    bonds: list[_Bond] = []
    for bond in structure.bonds:
        if mol2_semantics:
            btype = _bond_type_from_order(bond.order)
        else:
            # BCC consumes bondtype's internal integer codes (including 9).
            btype = 1 if 1.3 <= float(bond.order) <= 1.7 else round(bond.order)
        bonds.append(_Bond(bond.i - 1, bond.j - 1, btype))
    for b in bonds:
        if not (0 <= b.bondi < len(atoms) and 0 <= b.bondj < len(atoms)):
            raise ParameterError("GAFF bond refers to an atom outside the structure")
        atoms[b.bondi].con.append(b.bondj)
        atoms[b.bondj].con.append(b.bondi)
    for at in atoms:
        at.connum = len(at.con)
        if at.connum > 6:
            raise ParameterError(
                "GAFF atomtype supports at most six neighbours per atom"
            )
    return atoms, bonds


def assign_gaff_types(
    structure: Structure,
    definition: str | Path | None = None,
    *,
    pindex: int = 0,
    post_adjustment: bool = True,
) -> dict[int, str]:
    """Assign GAFF atom types to a structure entirely in Python.

    Returns a mapping of 1-based atom index -> GAFF type name.  ``pindex``
    selects the typing definition (0=gaff/GFF, 1=amber, 2=bcc, 3=gas, 4=sybyl,
    5=gaff2, 6=abcg2); ``post_adjustment`` mirrors the ``-a 1`` flag.
    """
    atoms, bonds = _build_graph(structure, mol2_semantics=pindex not in (2, 6))
    natoms = len(atoms)

    if definition is None:
        names = {
            0: "ATOMTYPE_GFF.DEF",
            1: "ATOMTYPE_AMBER.DEF",
            2: "ATOMTYPE_BCC.DEF",
            3: "ATOMTYPE_GAS.DEF",
            5: "ATOMTYPE_GFF2.DEF",
            6: "ATOMTYPE_ABCG2.DEF",
        }
        fname = names.get(pindex, "ATOMTYPE_GFF.DEF")
        definition = _resolve_def_file(fname)

    try:
        wildatoms, rules = _parse_def(Path(definition))
    except OSError as exc:
        raise ParameterError(
            f"cannot read atom typing definition: {definition}"
        ) from exc

    arom = [_Arom() for _ in range(natoms)]
    # ringdetect: bcc ring adjustment only for pindex 2 or 6
    _ringdetect(atoms, bonds, arom, bcc_ring=(pindex in (2, 6)))
    bi = _bondinfo(bonds, natoms)

    # main typing loop (jat)
    for i in range(natoms):
        for rule in rules:
            if rule.resname == "&":
                atoms[i].ambername = rule.type_name
                break
            resindex = 0
            if rule.resname == "*" or rule.atomicnum == atoms[i].aa:
                resindex = 1
            if (rule.resname == "AA" or rule.atomicnum == "BIO") and atoms[i].aa in _AA:
                resindex = 1
            if (rule.resname == "NA" or rule.atomicnum == "BIO") and atoms[i].aa in _NA:
                resindex = 1
            if resindex == 0:
                continue
            if rule.atomicnum == "&":
                atoms[i].ambername = rule.type_name
                break
            if (int(rule.atomicnum) if rule.atomicnum.isdecimal() else 0) != atoms[
                i
            ].atomicnum:
                continue
            if rule.connum == "&":
                atoms[i].ambername = rule.type_name
                break
            if rule.connum != "*" and int(rule.connum) != atoms[i].connum:
                continue
            if rule.nh == "&":
                atoms[i].ambername = rule.type_name
                break
            if rule.nh != "*":
                nh = sum(1 for nb in atoms[i].con if atoms[nb].atomicnum == 1)
                if int(rule.nh) != nh:
                    continue
            if rule.ewg == "&":
                atoms[i].ambername = rule.type_name
                break
            if rule.ewg != "*":
                if atoms[i].connum == 0:
                    continue
                first = atoms[i].con[0]
                cnt = sum(1 for nb in atoms[first].con if atoms[nb].ewd == 1)
                if int(rule.ewg) != cnt:
                    continue
            if rule.prop == "&":
                atoms[i].ambername = rule.type_name
                break
            if (
                rule.prop != "*"
                and _apcheck(atoms, arom, bi, bonds, i, -1, rule.prop) == 0
            ):
                continue
            if rule.env == "&":
                atoms[i].ambername = rule.type_name
                break
            ok, _ = _jatspecial(
                atoms, arom, bi, bonds, wildatoms, i, rule.env, rule.ces_bond
            )
            if ok == 1:
                atoms[i].ambername = rule.type_name
                break
            # else continue to next rule

    if post_adjustment:
        _atadjust(atoms, bonds)
        _cpadjust(atoms, bonds)

    result = {}
    for idx, at in enumerate(atoms):
        result[idx + 1] = (
            at.ambername if at.ambername else _element_fallback(atoms, idx)
        )
    return result


def _element_fallback(atoms: list[_Atom], idx: int) -> str:
    """Fallback matching the DEF element-symbol rules at the end of GFF.DEF."""
    elem = atoms[idx].element
    return elem


def _write_mol2_for_oracle(structure: Structure, path: str | Path) -> None:
    """Compatibility entry point for the shared canonical MOL2 serializer."""
    from .atomtype import _write_mol2

    _write_mol2(structure, Path(path))


# ---------------------------------------------------------------------------
# public helpers
# ---------------------------------------------------------------------------


def gaff_type_requires_external() -> bool:
    """Whether typing requires an external executable (always False)."""
    return False


def discover_atomtype() -> None:
    """Placeholder retained for API compatibility (no external binary)."""
    return
