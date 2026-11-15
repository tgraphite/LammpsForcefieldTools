def test_package_import():
    import lft

    assert lft.__version__


def test_cli_contract():
    from lft.cli import build_parser

    args = build_parser().parse_args(
        [
            "-f",
            "molecule.mol2",
            "--parameter-mode",
            "hessian",
            "--charge",
            "resp2",
            "--ff",
            "gaff",
            "--net-charge",
            "1",
        ]
    )

    assert args.structure == "molecule.mol2"
    assert args.parameter_mode == "hessian"
    assert args.charge == "resp2"
    assert args.ff == "gaff"
    assert args.net_charge == 1
    # Defaults are resolved later (config file aware), so the parser leaves
    # output/ff/charge unset unless the user asks for them.
    defaults = build_parser().parse_args(["-f", "molecule.pdb"])
    assert defaults.output is None
    assert defaults.ff is None
    assert defaults.charge is None
    assert defaults.outdir is None

