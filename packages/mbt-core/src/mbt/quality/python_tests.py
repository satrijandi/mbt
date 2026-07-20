"""Python data tests in ``tests/`` (TSD §5.7, FR-RES-05).

Discovery is static (AST only, no import) so ``mbt parse`` stays fast and
side-effect free; test modules are imported only when tests actually run,
inside the coordinator, against materialized datasets.
"""

import ast
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from mbt.contracts import DatasetSpec, TestResult
from mbt.events import get_bus
from mbt.events.models import TestEvaluated
from mbt.exceptions import MbtError

if TYPE_CHECKING:
    from mbt.parsing.errors import ParseReport

_SELECT_HEADER_RE = re.compile(r"^#\s*mbt:\s*select\s*=\s*(?P<selector>.+?)\s*$")


@dataclass(frozen=True)
class PythonTestFile:
    """One discovered test file and its binding selector."""

    path: Path
    rel: str
    selector: str | None  # from '# mbt: select=<selector>'; None = all datasets
    test_names: tuple[str, ...]


def discover_python_tests(
    project_dir: Path, test_paths: list[str], report: "ParseReport"
) -> list[PythonTestFile]:
    """Find test files, their ``test_*`` functions, and binding selectors."""
    found: list[PythonTestFile] = []
    for test_dir_name in test_paths:
        test_dir = project_dir / test_dir_name
        if not test_dir.is_dir():
            continue
        for path in sorted(test_dir.rglob("*.py")):
            if path.name.startswith("_"):
                continue
            rel = str(path.relative_to(project_dir))
            text = path.read_text()
            selector = _extract_selector(text)
            try:
                tree = ast.parse(text, filename=str(path))
            except SyntaxError as exc:
                report.error(f"invalid Python in data test: {exc}", file=rel)
                continue
            test_names = tuple(
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test_")
            )
            if not test_names:
                report.warning(
                    "no test_* functions found",
                    file=rel,
                    hint="data tests are functions like: def test_x(dataset, spec) -> TestResult",
                )
                continue
            found.append(
                PythonTestFile(path=path, rel=rel, selector=selector, test_names=test_names)
            )
    return found


def _extract_selector(text: str) -> str | None:
    for line in text.splitlines()[:10]:
        match = _SELECT_HEADER_RE.match(line.strip())
        if match:
            return match.group("selector")
    return None


def run_python_tests(
    test_file: PythonTestFile,
    dataset: pa.Table,
    spec: DatasetSpec,
    only: set[str] | None = None,
    resource: str | None = None,
) -> list[TestResult]:
    """Import a test module and run its test functions against a dataset."""
    bus = get_bus()
    module_name = f"_mbt_data_test_{test_file.path.stem}"
    spec_obj = importlib.util.spec_from_file_location(module_name, test_file.path)
    if spec_obj is None or spec_obj.loader is None:  # pragma: no cover - importlib guards
        raise MbtError(f"cannot import data test module {test_file.rel}")
    module = importlib.util.module_from_spec(spec_obj)
    sys.modules[module_name] = module
    try:
        spec_obj.loader.exec_module(module)
        results: list[TestResult] = []
        for name in test_file.test_names:
            if only is not None and name not in only:
                continue
            func: Any = getattr(module, name)
            try:
                outcome = func(dataset, spec)
            except Exception as exc:
                result = TestResult(name=name, passed=False, message=f"raised {exc!r}")
            else:
                if isinstance(outcome, TestResult):
                    result = outcome
                elif isinstance(outcome, bool) or outcome is None:
                    # Bare asserts / boolean returns are accepted for ergonomics.
                    passed = bool(outcome) if isinstance(outcome, bool) else True
                    result = TestResult(name=name, passed=passed)
                else:
                    result = TestResult(
                        name=name,
                        passed=False,
                        message=f"returned {type(outcome).__name__}, expected TestResult",
                    )
            results.append(result)
            # One event per test so the pass/fail stream is visible on the bus,
            # mirroring the built-in checks' CheckEvaluated.
            bus.emit(
                TestEvaluated(
                    unique_id=resource,
                    test=result.name,
                    passed=result.passed,
                    message=result.message,
                )
            )
        return results
    finally:
        sys.modules.pop(module_name, None)
