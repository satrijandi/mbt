"""Core does not probe adapters for optional methods (FEEDBACK v7 C-1).

``CLAUDE.md``: "Never add a ``hasattr`` probe for an optional adapter method; add
a ``Capability`` and let ``capabilities_of`` answer." v5 B-1 did that for
training adapters and left eleven probes for the tracking, registry, compute and
data ones, one of them a test that named the probe as intended. A renamed method
then silently turned a feature off, and nothing could see it.
"""

import re
from pathlib import Path

CORE_SRC = Path(__file__).resolve().parents[1] / "src" / "mbt"

#: Lines that are not an adapter probe, each with the reason.
_ALLOWED = (
    "isoformat",  # a date or datetime check on a value, not an adapter
    "sysconf",  # os.sysconf exists only on some platforms
)


def test_core_never_probes_an_adapter_with_hasattr() -> None:
    offenders = [
        f"{path.relative_to(CORE_SRC)}:{number}: {line.strip()}"
        for path in sorted(CORE_SRC.rglob("*.py"))
        if "_scaffold" not in path.parts
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if re.search(r"\bhasattr\(", line) and not any(token in line for token in _ALLOWED)
    ]
    assert not offenders, (
        "ask `supports(adapter, Capability.X)` instead of probing (FEEDBACK v7 C-1); "
        "declare a new capability in mbt_adapter_base.capabilities if one is missing:\n"
        + "\n".join(offenders)
    )
