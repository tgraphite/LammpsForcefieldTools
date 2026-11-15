"""Minimal pySCF smoke test: B3LYP-D3(BJ)/def2-SVP, PCM, Hessian."""

from __future__ import annotations

import numpy as np
from pyscf import gto, scf, solvent
import dftd3.pyscf as disp


def main() -> None:
    mol = gto.M(
        atom=[
            ["O", (0.0000, 0.0000, 0.1173)],
            ["H", (0.0000, 0.7572, -0.4692)],
            ["H", (0.0000, -0.7572, -0.4692)],
        ],
        basis="def2-svp",
        verbose=0,
    )

    mf = disp.d3_energy(
        scf.RKS(mol, xc="b3lyp"),
        version="d3bj",
    ).run()
    print(f"gas_energy {mf.e_tot:.14f}")

    hess = mf.Hessian().kernel()
    print(f"hessian_shape {hess.shape}")
    print(f"hessian_norm {np.linalg.norm(hess):.12f}")

    pcm = solvent.PCM(mf)
    pcm.eps = 78.39
    pcm.lebedev_order = 17
    pcm.method = "IEFPCM"
    pmf = pcm.run()
    print(f"pcm_energy {pmf.e_tot:.14f}")


if __name__ == "__main__":
    main()
