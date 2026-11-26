"""The experiment-level parity cases: which experiment, with which arguments (tiny testing mode; everything else is the
paper's defaults), and which stage of which rung it covers. Shared by the golden maker and the gate."""

# case name -> (experiment, arguments). The paths of the models are the zoo's, taken from DEFOG_MEGAN_ZOO.
CASES = {
    "colorgraphs/probe": ("sample_color_joint", ["--__TESTING__", "True", "--STAGE", "'probe'"]),
    "colorgraphs/fk": ("sample_color_joint", ["--__TESTING__", "True", "--STAGE", "'fk'"]),
    "zinc-logp/single": ("sample_mol_joint", ["--__TESTING__", "True", "--STAGE", "'single'"]),
    "zinc-logp/probe": ("sample_mol_joint", ["--__TESTING__", "True", "--STAGE", "'probe'"]),
    "zinc-logp/fk": ("sample_mol_joint", ["--__TESTING__", "True", "--STAGE", "'fk'"]),
    "ames/single": ("sample_ames_joint", ["--__TESTING__", "True", "--STAGE", "'single'"]),
    "ames/probe": ("sample_ames_joint", ["--__TESTING__", "True", "--STAGE", "'probe'"]),
    "ames/fk": ("sample_ames_joint", ["--__TESTING__", "True", "--STAGE", "'fk'"]),
}
