"""Parse-time issue collection (FR-PARSE-02).

Validation collects *all* errors in one pass; each is reported with file,
resource name, JSON-pointer field path, and an actionable message.
"""

from dataclasses import dataclass, field
from pathlib import Path

from mbt.exceptions import ConfigError


@dataclass(frozen=True)
class ParseIssue:
    """One validation finding with full location context."""

    severity: str  # "error" | "warning"
    file: str  # relative to the project dir
    resource: str | None
    field_path: str  # JSON pointer, "" for whole-resource issues
    message: str
    hint: str | None = None

    def format(self) -> str:
        location = self.file
        if self.resource:
            location += f" [{self.resource}]"
        if self.field_path:
            location += f" at {self.field_path}"
        text = f"{location}: {self.message}"
        if self.hint:
            text += f"\n    hint: {self.hint}"
        return text


@dataclass
class ParseReport:
    """Accumulates issues across the whole parse pass."""

    issues: list[ParseIssue] = field(default_factory=list)

    def error(
        self,
        message: str,
        *,
        file: str | Path = "",
        resource: str | None = None,
        field_path: str = "",
        hint: str | None = None,
    ) -> None:
        self.issues.append(
            ParseIssue(
                severity="error",
                file=str(file),
                resource=resource,
                field_path=field_path,
                message=message,
                hint=hint,
            )
        )

    def warning(
        self,
        message: str,
        *,
        file: str | Path = "",
        resource: str | None = None,
        field_path: str = "",
        hint: str | None = None,
    ) -> None:
        self.issues.append(
            ParseIssue(
                severity="warning",
                file=str(file),
                resource=resource,
                field_path=field_path,
                message=message,
                hint=hint,
            )
        )

    @property
    def errors(self) -> list[ParseIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ParseIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def raise_if_errors(self) -> None:
        errors = self.errors
        if errors:
            listing = "\n".join(f"  - {issue.format()}" for issue in errors)
            raise ConfigError(
                f"parsing failed with {len(errors)} error(s):\n{listing}",
                hint="fix the issues above and re-run 'mbt parse'",
            )
