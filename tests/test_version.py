"""The version string is the installed metadata's, not a second copy that can drift."""
import re
from importlib import metadata

import defog_megan


def test_the_version_is_the_installed_distributions():
    assert defog_megan.__version__ == metadata.version("defog-megan")


def test_the_version_looks_like_a_version():
    assert re.match(r"^\d+\.\d+\.\d+", defog_megan.__version__)


def test_the_citation_file_names_the_same_version():
    from pathlib import Path
    cff = (Path(__file__).resolve().parents[1] / "CITATION.cff").read_text()
    assert f'version: "{defog_megan.__version__}"' in cff
