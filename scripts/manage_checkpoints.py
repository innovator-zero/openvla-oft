#!/usr/bin/env python3
import argparse
import fnmatch
import os
import shutil
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

CHECKPOINTS_ROOT = Path(__file__).resolve().parent.parent / "experiments" / "checkpoints"


@dataclass(frozen=True)
class Checkpoint:
    step: int
    path: Path
    size_bytes: int
    mtime: float


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0


def compute_dir_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                fp = Path(root) / f
                total += fp.stat().st_size
            except FileNotFoundError:
                continue
    return total


def iter_experiments(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            yield entry


def parse_step_from_dirname(name: str) -> Optional[int]:
    # Expected pattern like "50000_chkpt" or "150000_chkpt"
    if name.endswith("_chkpt"):
        prefix = name[: -len("_chkpt")]
        if prefix.isdigit():
            try:
                return int(prefix)
            except ValueError:
                return None
    return None


def find_checkpoints(exp_dir: Path) -> List[Checkpoint]:
    checkpoints: List[Checkpoint] = []
    for entry in exp_dir.iterdir():
        if not entry.is_dir():
            continue
        step = parse_step_from_dirname(entry.name)
        if step is None:
            continue
        try:
            size_bytes = compute_dir_size(entry)
            mtime = entry.stat().st_mtime
        except FileNotFoundError:
            continue
        checkpoints.append(Checkpoint(step=step, path=entry, size_bytes=size_bytes, mtime=mtime))
    checkpoints.sort(key=lambda c: c.step)
    return checkpoints


def ensure_root(root_arg: Optional[str]) -> Path:
    root = Path(root_arg) if root_arg else CHECKPOINTS_ROOT
    return root.resolve()


def cmd_list(args: argparse.Namespace) -> int:
    root = ensure_root(args.root)
    if not root.exists():
        print(f"No checkpoints directory found at: {root}")
        return 1
    rows: List[Tuple[str, int, str, str]] = []
    for exp in iter_experiments(root):
        cps = find_checkpoints(exp)
        total_size = sum(c.size_bytes for c in cps)
        latest = cps[-1].step if cps else None
        rows.append((exp.name, len(cps), human_size(total_size), str(latest) if latest is not None else "-"))

    if not rows:
        print("No experiments found.")
        return 0

    print(f"Root: {root}")
    print("Experiment | #Checkpoints | Total Size | Latest Step")
    print("-" * 70)
    for name, n, size_str, latest in rows:
        print(f"{name} | {n:>3} | {size_str:>10} | {latest}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    root = ensure_root(args.root)
    exp_dir = (root / args.experiment).resolve()
    if not exp_dir.exists():
        print(f"Experiment not found: {exp_dir}")
        return 1
    cps = find_checkpoints(exp_dir)
    if not cps:
        print(f"No checkpoints in {exp_dir}")
        return 0
    print(f"Experiment: {exp_dir}")
    print("Step | Size | Modified")
    print("-" * 50)
    for c in cps:
        mod = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(c.mtime))
        print(f"{c.step:>6} | {human_size(c.size_bytes):>10} | {mod}")
    total = sum(c.size_bytes for c in cps)
    print("-" * 50)
    print(f"Total: {human_size(total)} across {len(cps)} checkpoints")
    return 0


def cmd_prune(args: argparse.Namespace) -> int:
    root = ensure_root(args.root)
    exp_dir = (root / args.experiment).resolve()
    if not exp_dir.exists():
        print(f"Experiment not found: {exp_dir}")
        return 1
    cps = find_checkpoints(exp_dir)
    if not cps:
        print(f"No checkpoints in {exp_dir}")
        return 0

    keep_steps: List[int] = []
    if args.keep_steps:
        for token in args.keep_steps.split(","):
            token = token.strip()
            if token:
                try:
                    keep_steps.append(int(token))
                except ValueError:
                    print(f"Warning: invalid step '{token}' in --keep-steps; ignoring")

    # Always keep the latest N
    latest_to_keep = set(c.step for c in cps[-args.keep_latest :]) if args.keep_latest > 0 else set()
    explicit_keep = set(keep_steps)
    to_keep = latest_to_keep | explicit_keep

    to_delete: List[Checkpoint] = []
    for c in cps:
        if c.step not in to_keep:
            to_delete.append(c)

    if not to_delete:
        print("Nothing to prune.")
        return 0

    print(f"Would delete {len(to_delete)} checkpoint(s):")
    total_reclaim = 0
    for c in to_delete:
        print(f"  {c.path} ({human_size(c.size_bytes)})")
        total_reclaim += c.size_bytes
    print(f"Reclaim: {human_size(total_reclaim)}")

    if args.dry_run:
        print("Dry run: no deletions performed.")
        return 0

    for c in to_delete:
        shutil.rmtree(c.path, ignore_errors=True)
    print("Deleted above checkpoints.")
    return 0


def cmd_prune_all(args: argparse.Namespace) -> int:
    root = ensure_root(args.root)
    if not root.exists():
        print(f"No checkpoints directory found at: {root}")
        return 1

    keep_steps: List[int] = []
    if args.keep_steps:
        for token in args.keep_steps.split(","):
            token = token.strip()
            if token:
                try:
                    keep_steps.append(int(token))
                except ValueError:
                    print(f"Warning: invalid step '{token}' in --keep-steps; ignoring")

    total_to_delete: List[Tuple[Path, int]] = []
    for exp in iter_experiments(root):
        cps = find_checkpoints(exp)
        if not cps:
            continue
        latest_to_keep = set(c.step for c in cps[-args.keep_latest :]) if args.keep_latest > 0 else set()
        explicit_keep = set(keep_steps)
        to_keep = latest_to_keep | explicit_keep
        for c in cps:
            if c.step not in to_keep:
                total_to_delete.append((c.path, c.size_bytes))

    if not total_to_delete:
        print("Nothing to prune across experiments.")
        return 0

    reclaim = sum(size for _, size in total_to_delete)
    print(f"Would delete {len(total_to_delete)} checkpoint directories across experiments:")
    for path, size in total_to_delete:
        print(f"  {path} ({human_size(size)})")
    print(f"Reclaim: {human_size(reclaim)}")

    if args.dry_run:
        print("Dry run: no deletions performed.")
        return 0

    for path, _ in total_to_delete:
        shutil.rmtree(path, ignore_errors=True)
    print("Deleted above checkpoints.")
    return 0


def make_tar_gz(source_path: Path, dest_tar_gz: Path) -> None:
    with tarfile.open(dest_tar_gz, "w:gz") as tar:
        tar.add(source_path, arcname=source_path.name)


def cmd_archive(args: argparse.Namespace) -> int:
    root = ensure_root(args.root)
    target = (root / args.target).resolve()
    if not target.exists():
        print(f"Target not found: {target}")
        return 1

    out_dir = Path(args.output).resolve() if args.output else target.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{target.name}.tar.gz"
    out_path = out_dir / out_name

    if out_path.exists() and not args.overwrite:
        print(f"Archive already exists: {out_path}. Use --overwrite to replace.")
        return 1

    print(f"Creating archive: {out_path}")
    if args.dry_run:
        print("Dry run: no archive created.")
        return 0
    make_tar_gz(target, out_path)
    print("Archive created.")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    root = ensure_root(args.root)
    if not root.exists():
        print(f"No checkpoints directory found at: {root}")
        return 1

    now = time.time()
    cutoff_mtime: Optional[float] = None
    if args.older_than_days is not None:
        cutoff_mtime = now - (args.older_than_days * 24 * 3600)

    paths_to_delete: List[Path] = []

    for exp in iter_experiments(root):
        # If pattern matches the experiment dir name, delete the whole experiment
        exp_match = fnmatch.fnmatch(exp.name, args.pattern)
        if exp_match and (cutoff_mtime is None or exp.stat().st_mtime < cutoff_mtime):
            paths_to_delete.append(exp)
            continue

        # Otherwise, check its checkpoint subdirs
        for c in find_checkpoints(exp):
            if fnmatch.fnmatch(c.path.name, args.pattern):
                if cutoff_mtime is None or c.mtime < cutoff_mtime:
                    paths_to_delete.append(c.path)

    if not paths_to_delete:
        print("Nothing matches deletion criteria.")
        return 0

    total = 0
    print("Will delete:")
    for p in paths_to_delete:
        try:
            total += compute_dir_size(p) if p.is_dir() else p.stat().st_size
        except FileNotFoundError:
            continue
        print(f"  {p}")
    print(f"Total reclaim: {human_size(total)}")

    if args.dry_run:
        print("Dry run: no deletions performed.")
        return 0

    for p in paths_to_delete:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            try:
                p.unlink(missing_ok=True)
            except TypeError:
                # Python <3.8 compatibility not required, but keep safe behavior
                if p.exists():
                    p.unlink()
    print("Deleted items above.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Manage experiment checkpoints under experiments/checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root", type=str, default=None, help="Override checkpoints root directory")

    sub = p.add_subparsers(dest="cmd", required=True)

    sp_list = sub.add_parser("list", help="List experiments and summary sizes")
    sp_list.set_defaults(func=cmd_list)

    sp_show = sub.add_parser("show", help="Show checkpoints within an experiment")
    sp_show.add_argument("experiment", type=str, help="Experiment directory name")
    sp_show.set_defaults(func=cmd_show)

    sp_prune = sub.add_parser("prune", help="Prune checkpoints within an experiment")
    sp_prune.add_argument("experiment", type=str, help="Experiment directory name")
    sp_prune.add_argument("--keep-latest", type=int, default=1, help="Keep N latest checkpoints")
    sp_prune.add_argument(
        "--keep-steps", type=str, default="", help="Comma-separated step numbers to keep regardless of recency"
    )
    sp_prune.add_argument("--dry-run", action="store_true", help="Preview deletions without applying")
    sp_prune.set_defaults(func=cmd_prune)

    sp_prune_all = sub.add_parser(
        "prune-all", help="Prune checkpoints in ALL experiments, keeping latest N and/or specific steps"
    )
    sp_prune_all.add_argument("--keep-latest", type=int, default=1, help="Keep N latest checkpoints per experiment")
    sp_prune_all.add_argument(
        "--keep-steps", type=str, default="", help="Comma-separated step numbers to keep in every experiment"
    )
    sp_prune_all.add_argument("--dry-run", action="store_true", help="Preview deletions without applying")
    sp_prune_all.set_defaults(func=cmd_prune_all)

    sp_archive = sub.add_parser("archive", help="Archive an experiment or a single checkpoint as tar.gz")
    sp_archive.add_argument("target", type=str, help="Experiment dir or specific checkpoint dir name")
    sp_archive.add_argument("--output", type=str, default=None, help="Output directory for archive")
    sp_archive.add_argument("--overwrite", action="store_true", help="Overwrite if archive exists")
    sp_archive.add_argument("--dry-run", action="store_true", help="Preview without creating archive")
    sp_archive.set_defaults(func=cmd_archive)

    sp_delete = sub.add_parser("delete", help="Delete experiments or checkpoints by pattern/age")
    sp_delete.add_argument("pattern", type=str, help="Glob pattern for experiment or checkpoint names")
    sp_delete.add_argument("--older-than-days", type=int, default=None, help="Only delete if modified before N days")
    sp_delete.add_argument("--dry-run", action="store_true", help="Preview deletions without applying")
    sp_delete.set_defaults(func=cmd_delete)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
