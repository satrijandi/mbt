"""No package reads YAML except through ``mbt.yamlio`` (FEEDBACK v6 A-4).

``yaml.safe_load`` keeps the last of two equal keys silently. Six loaders called
it directly when that was found, including the one that reads
``promotions.yml``; they all go through ``mbt.yamlio.safe_load`` now, and this
stops a seventh from appearing beside them.
"""

import re

from e2e_utils import REPO_ROOT

#: Any direct PyYAML load: safe_load, load, full_load, unsafe_load, *_all.
_DIRECT_LOAD = re.compile(r"\byaml\.(safe_load|load|full_load|unsafe_load)(_all)?\(")
_THE_ONE_READER = "packages/mbt-core/src/mbt/yamlio.py"


def test_every_yaml_read_goes_through_mbt_yamlio() -> None:
    sources = sorted(REPO_ROOT.glob("packages/*/src/**/*.py"))
    assert len(sources) > 100, "the glob found too little to be a real scan"
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{number}: {line.strip()}"
        for path in sources
        if path.relative_to(REPO_ROOT).as_posix() != _THE_ONE_READER
        and "/_scaffold/" not in path.as_posix()
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if _DIRECT_LOAD.search(line)
    ]
    assert not offenders, (
        "read YAML with mbt.yamlio.safe_load, which refuses duplicate keys:\n"
        + "\n".join(offenders)
    )
