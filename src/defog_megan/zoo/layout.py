"""Where the model zoo lives on this machine, and where a file of it is.

The experiments take their checkpoints, tables and request vectors as parameters. Their defaults point into the model
zoo, so that the documented commands stay short and no path of anyone's machine is written into the code. The zoo is a
folder with one subfolder per group (``colorgraphs``, ``zinc-logp``, ``ames``) laid out as in the manifest.

The root is the environment variable ``DEFOG_MEGAN_ZOO`` or, without it, ``~/.cache/defog-megan/zoo``.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "DEFOG_MEGAN_ZOO"


def zoo_root() -> Path:
    """The root of the zoo on this machine (it need not exist)."""
    configured = os.environ.get(ENV_VAR)
    return Path(configured).expanduser() if configured else Path.home() / ".cache" / "defog-megan" / "zoo"


def zoo_path(group: str, relative: str) -> str:
    """The path of one file of the zoo, as a string, resolved at the time of the call.

    Nothing is checked here: a missing file is reported by the code that opens it, with this path in the message.
    """
    return str(zoo_root() / group / relative)
