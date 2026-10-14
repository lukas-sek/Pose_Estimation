from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Portable project locations derived from one repository root."""

    root: Path
    processed: Path
    raw: Path
    runs: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "ProjectPaths":
        root = Path(root).expanduser().resolve()
        return cls(
            root=root,
            processed=root / "preprocessed_data",
            raw=root / "cloth3d++_subset",
            runs=root / "runs",
        )

    @property
    def base_data(self) -> Path:
        nested = self.processed / "preprocessed_data"
        return nested if nested.exists() else self.processed

    def resized(self, size: int = 256, interpolation: str = "nearest") -> Path:
        return self.processed / "resized_variants" / f"{size}_{interpolation}"

    def run(self, task: str) -> Path:
        return self.runs / task
