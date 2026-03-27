from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import shutil
import subprocess

import numpy as np

_REQUIRED_KEYS = (
    "num_layers",
    "raw_damping",
    "raw_dual_step_mu",
    "raw_dual_step_lambda",
)


@dataclass(frozen=True)
class CheckpointRepair:
    name: str
    source: str
    destination: Path


@dataclass(frozen=True)
class CheckpointSourceReport:
    name: str
    destination: Path
    destination_exists: bool
    destination_valid: bool
    valid_train_candidates: tuple[Path, ...]
    git_candidates: tuple[str, ...]
    git_head_candidates_present: tuple[str, ...]
    git_available: bool
    git_repo_ok: bool


def _resolve_repo_root(repo_root: str | Path | None = None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    return Path.cwd().resolve()


def _resolve_path(path: str | Path, *, repo_root: str | Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return _resolve_repo_root(repo_root) / p


def checkpoint_roots_from_cfg(
    cfg: Mapping[str, Any], *, repo_root: str | Path | None = None
) -> tuple[Path, Path]:
    twc = cfg.get("twc", {}) or {}
    output_root_raw = Path(str(twc.get("output_root", "results_twc")))
    checkpoint_root_raw = Path(str(twc.get("checkpoint_root", output_root_raw / "checkpoints")))
    return (
        _resolve_path(output_root_raw, repo_root=repo_root),
        _resolve_path(checkpoint_root_raw, repo_root=repo_root),
    )


def is_valid_unfolding_checkpoint(path: str | Path) -> bool:
    p = Path(path)
    if not p.is_file():
        return False
    try:
        with np.load(p, allow_pickle=False) as data:
            files = set(data.files)
        return all(key in files for key in _REQUIRED_KEYS)
    except Exception:
        return False


def _latest_train_candidates(output_root: Path, name: str) -> list[Path]:
    candidates: list[Path] = []
    train_dirs = sorted(output_root.glob(f"train_{name}_*"), reverse=True)
    for train_dir in train_dirs:
        final_npz = train_dir / f"{name}_final.npz"
        if final_npz.is_file():
            candidates.append(final_npz)
        ckpt_dir = train_dir / "checkpoints"
        if ckpt_dir.is_dir():
            step_ckpts = sorted(ckpt_dir.glob(f"{name}_step*.npz"), reverse=True)
            candidates.extend(step_ckpts)
    return candidates


def _copy_if_valid(source: Path, dest: Path) -> bool:
    if not is_valid_unfolding_checkpoint(source):
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != dest.resolve():
        shutil.copy2(source, dest)
    return is_valid_unfolding_checkpoint(dest)


def _safe_run(cmd: list[str]) -> subprocess.CompletedProcess[bytes] | None:
    try:
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        return None


def _git_available(repo_root: Path) -> bool:
    proc = _safe_run(["git", "-C", str(repo_root), "--version"])
    return proc is not None and proc.returncode == 0


def _git_repo_ok(repo_root: Path) -> bool:
    proc = _safe_run(["git", "-C", str(repo_root), "rev-parse", "--is-inside-work-tree"])
    if proc is None or proc.returncode != 0:
        return False
    return proc.stdout.decode(errors="ignore").strip().lower() == "true"


def _git_head_has_path(repo_root: Path, repo_relpath: str) -> bool:
    proc = _safe_run(["git", "-C", str(repo_root), "cat-file", "-e", f"HEAD:{repo_relpath}"])
    return proc is not None and proc.returncode == 0


def _git_show_to_path(repo_root: Path, repo_relpath: str, dest: Path) -> bool:
    proc = _safe_run(["git", "-C", str(repo_root), "show", f"HEAD:{repo_relpath}"])
    if proc is None or proc.returncode != 0 or not proc.stdout:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(proc.stdout)
    if is_valid_unfolding_checkpoint(tmp):
        tmp.replace(dest)
        return True
    tmp.unlink(missing_ok=True)
    return False


def _git_checkout_to_path(repo_root: Path, repo_relpath: str, dest: Path) -> bool:
    """
    Restore a tracked file straight into the working tree.

    This is more robust than piping binary bytes through `git show` when the local
    repository already has the object and only the working-tree file is missing.
    """
    tracked_path = repo_root / repo_relpath
    tracked_path.parent.mkdir(parents=True, exist_ok=True)

    commands = [
        ["git", "-C", str(repo_root), "restore", "--source=HEAD", "--worktree", "--", repo_relpath],
        ["git", "-C", str(repo_root), "checkout", "HEAD", "--", repo_relpath],
        ["git", "-C", str(repo_root), "checkout", "--", repo_relpath],
    ]
    for cmd in commands:
        proc = _safe_run(cmd)
        if proc is None or proc.returncode != 0:
            continue
        if is_valid_unfolding_checkpoint(tracked_path):
            dest.parent.mkdir(parents=True, exist_ok=True)
            if tracked_path.resolve() != dest.resolve():
                shutil.copy2(tracked_path, dest)
            return is_valid_unfolding_checkpoint(dest)
    return False


def _repo_relpath(path: Path, repo_root: Path) -> str | None:
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
    except Exception:
        return None
    return rel.as_posix()


def checkpoint_recovery_report(
    *,
    checkpoint_root: str | Path,
    names: Sequence[str],
    output_root: str | Path = "results_twc",
    repo_root: str | Path | None = None,
) -> list[CheckpointSourceReport]:
    repo_root_p = _resolve_repo_root(repo_root)
    output_root_p = _resolve_path(output_root, repo_root=repo_root_p)
    checkpoint_root_p = _resolve_path(checkpoint_root, repo_root=repo_root_p)
    legacy_root = output_root_p / "checkpoints"

    git_available = _git_available(repo_root_p)
    git_repo_ok = _git_repo_ok(repo_root_p) if git_available else False

    reports: list[CheckpointSourceReport] = []
    for name in names:
        dest = checkpoint_root_p / f"{name}.npz"
        valid_train_candidates = tuple(
            p for p in _latest_train_candidates(output_root_p, str(name)) if is_valid_unfolding_checkpoint(p)
        )
        git_candidates: list[str] = []
        for path in [dest, legacy_root / f"{name}.npz"]:
            rel = _repo_relpath(path, repo_root_p)
            if rel is not None and rel not in git_candidates:
                git_candidates.append(rel)
        git_head_candidates_present: tuple[str, ...] = tuple(
            rel for rel in git_candidates if git_available and git_repo_ok and _git_head_has_path(repo_root_p, rel)
        )
        reports.append(
            CheckpointSourceReport(
                name=str(name),
                destination=dest,
                destination_exists=dest.exists(),
                destination_valid=is_valid_unfolding_checkpoint(dest),
                valid_train_candidates=valid_train_candidates,
                git_candidates=tuple(git_candidates),
                git_head_candidates_present=git_head_candidates_present,
                git_available=git_available,
                git_repo_ok=git_repo_ok,
            )
        )
    return reports


def ensure_checkpoint_files(
    *,
    checkpoint_root: str | Path,
    names: Sequence[str],
    output_root: str | Path = "results_twc",
    repo_root: str | Path | None = None,
    verbose: bool = False,
) -> list[CheckpointRepair]:
    repo_root_p = _resolve_repo_root(repo_root)
    output_root_p = _resolve_path(output_root, repo_root=repo_root_p)
    checkpoint_root_p = _resolve_path(checkpoint_root, repo_root=repo_root_p)
    legacy_root = output_root_p / "checkpoints"
    checkpoint_root_p.mkdir(parents=True, exist_ok=True)

    repairs: list[CheckpointRepair] = []

    for name in names:
        dest = checkpoint_root_p / f"{name}.npz"
        if is_valid_unfolding_checkpoint(dest):
            continue
        if dest.exists():
            dest.unlink(missing_ok=True)

        recovered = False

        for candidate in [legacy_root / f"{name}.npz", *(_latest_train_candidates(output_root_p, name))]:
            if candidate.resolve() == dest.resolve():
                continue
            if _copy_if_valid(candidate, dest):
                repairs.append(
                    CheckpointRepair(
                        name=str(name),
                        source=str(candidate),
                        destination=dest,
                    )
                )
                recovered = True
                break

        if recovered:
            if verbose:
                last = repairs[-1]
                print(
                    f"CHECKPOINT_RESTORED name={last.name} source={last.source} "
                    f"destination={last.destination}"
                )
            continue

        git_candidates: list[str] = []
        for path in [dest, legacy_root / f"{name}.npz"]:
            rel = _repo_relpath(path, repo_root_p)
            if rel is not None and rel not in git_candidates:
                git_candidates.append(rel)

        for rel in git_candidates:
            if _git_show_to_path(repo_root_p, rel, dest):
                repairs.append(
                    CheckpointRepair(
                        name=str(name),
                        source=f"git:HEAD:{rel}",
                        destination=dest,
                    )
                )
                recovered = True
                break
            if _git_checkout_to_path(repo_root_p, rel, dest):
                repairs.append(
                    CheckpointRepair(
                        name=str(name),
                        source=f"git-worktree:HEAD:{rel}",
                        destination=dest,
                    )
                )
                recovered = True
                break

        if verbose and recovered:
            last = repairs[-1]
            print(
                f"CHECKPOINT_RESTORED name={last.name} source={last.source} "
                f"destination={last.destination}"
            )

    return repairs


__all__ = [
    "CheckpointRepair",
    "CheckpointSourceReport",
    "checkpoint_recovery_report",
    "checkpoint_roots_from_cfg",
    "ensure_checkpoint_files",
    "is_valid_unfolding_checkpoint",
]
