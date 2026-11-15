# Ground Truth Fixtures

The first checked-in fixtures are the periodic topology references whose
inputs already live in `references/sobtop/examples/`.  They are intentionally
not copied or rewritten: the input hash and expected image-shift invariants
remain tied to the repository reference audit.

The remaining GT-01..GT-44 cases are added only with a distributable input,
normalized IR, and an independent comparison script.  A missing external
program must produce a blocked result, never a generated reference file.
