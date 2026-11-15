# Ground Truth Fixtures

Only non-periodic, 1:1-comparable cases live here.  A case is accepted only if
it has a distributable input, an independent reference, and a comparison that
can fail.  Cases without a numeric reference are removed rather than kept as
"aspirational" entries.

## Active fixtures

- `charges_multiwfn.json` - Multiwfn reference charges (water, methyl
  benzoate, phenol).  Used by `tests/test_charges_vs_reference.py`.
- `../fixtures/validation/` - 22 non-periodic molecules with frozen AmberTools
  `atomtype` GAFF-type references.  Used by
  `tests/test_validation_suite.py`; see `scripts/build_validation_suite.py`.

## Removed

The periodic/crystal cases (diamond, SiO2, FeCl2, MOF-5, crystalline GT-35..44)
were removed from the active set: they have no 1:1 numeric reference for
charges/energies, so they could only assert self-consistency.  The inputs and
the Sobtop reference `itp/top` files remain in `references/sobtop/examples/`
for future work once a REPEAT/DDEC charge reference exists.

`gt44_catalog.yaml` is now a non-periodic planning list only.
