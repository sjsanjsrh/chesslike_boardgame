#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple build helper for this repo.

Commands:
  - ext         : Compile alphabeta.py to a CPython extension (.pyd) using Nuitka.
  - activate    : Copy the most recent built .pyd from build/ to project root for import override.
  - deactivate  : Remove any alphabeta*.pyd in project root to use pure .py again.
  - clean       : Remove built artifacts (.pyd in root/build, alphabeta.build folder).
  - status      : Show where compiled artifacts exist.

Usage examples:
  python build.py ext                 # build .pyd into ./build (no import override)
  python build.py activate            # copy built .pyd from ./build to project root
  python build.py deactivate          # delete root .pyd so .py is used
  python build.py clean               # clean artifacts
  python build.py status              # show current state
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD_DIR = ROOT / "build"
ALPHABETA_PY = ROOT / "alphabeta.py"
PYD_PATTERN = re.compile(r"^alphabeta\..*\.pyd$", re.IGNORECASE)


def _ensure_build_dir() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)


essential_msg = (
    "Nuitka is required. Install/update in your venv:\n"
    "  python -m pip install -U nuitka ordered-set zstandard\n"
)


def _find_pyd(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return [p for p in folder.iterdir() if p.is_file() and PYD_PATTERN.match(p.name)]


def cmd_status(_: argparse.Namespace) -> int:
    root_pyds = _find_pyd(ROOT)
    build_pyds = _find_pyd(BUILD_DIR)
    print("Root:")
    for p in root_pyds:
        print("  ", p.name)
    if not root_pyds:
        print("   (none)")
    print("Build:")
    for p in build_pyds:
        print("  ", p.name)
    if not build_pyds:
        print("   (none)")
    return 0


def cmd_ext(args: argparse.Namespace) -> int:
    if not ALPHABETA_PY.exists():
        print(f"ERROR: {ALPHABETA_PY} not found.")
        return 2
    try:
        import nuitka  # noqa: F401
    except Exception:
        print("ERROR: Nuitka is not installed in this environment.")
        print(essential_msg)
        return 3

    _ensure_build_dir()

    py_exe = sys.executable
    cmd = [
        py_exe,
        "-m",
        "nuitka",
        "--module",
        str(ALPHABETA_PY),
        f"--output-dir={BUILD_DIR.as_posix()}",
        "--assume-yes-for-downloads",
    ]
    if args.jobs:
        cmd.append(f"--jobs={int(args.jobs)}")

    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("ERROR: Build failed with exit code", e.returncode)
        return e.returncode or 1

    built = _find_pyd(BUILD_DIR)
    if not built:
        print("WARN: No .pyd found in build dir (unexpected).")
    else:
        print("Built:")
        for p in built:
            print("  ", p.name)

    if args.activate:
        return cmd_activate(args)
    return 0


def cmd_activate(_: argparse.Namespace) -> int:
    """Copy the latest build pyd from build/ to project root for import override."""
    candidates = sorted(_find_pyd(BUILD_DIR), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        print("No built .pyd found in ./build. Run 'python build.py ext' first.")
        return 0
    latest = candidates[0]
    # Clean existing root pyds
    for p in _find_pyd(ROOT):
        try:
            p.unlink()
        except Exception:
            pass
    target = ROOT / latest.name
    shutil.copy2(latest, target)
    print(f"Activated: {target.name} (root import will use compiled module)")
    return 0


def cmd_deactivate(_: argparse.Namespace) -> int:
    removed = 0
    for p in _find_pyd(ROOT):
        try:
            p.unlink()
            removed += 1
        except Exception:
            pass
    print("Deactivated: removed", removed, "file(s) from project root")
    return 0


def cmd_clean(_: argparse.Namespace) -> int:
    # Remove root and build pyds
    removed = 0
    for folder in (ROOT, BUILD_DIR):
        for p in _find_pyd(folder):
            try:
                p.unlink()
                removed += 1
            except Exception:
                pass
    # Remove Nuitka C build folder
    c_build = ROOT / "alphabeta.build"
    if c_build.exists():
        try:
            shutil.rmtree(c_build)
            print("Removed:", c_build)
        except Exception as e:
            print("WARN: Failed to remove", c_build, e)
    print("Cleaned .pyd files:", removed)
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build helper for alphabeta with Nuitka")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_status = sub.add_parser("status", help="Show current compiled artifacts")
    p_status.set_defaults(func=cmd_status)

    p_ext = sub.add_parser("ext", help="Compile alphabeta.py to .pyd in ./build")
    p_ext.add_argument("--jobs", type=int, default=None, help="Parallel C compile jobs")
    p_ext.add_argument("--activate", action="store_true", help="Copy built .pyd to project root after build")
    p_ext.set_defaults(func=cmd_ext)

    p_act = sub.add_parser("activate", help="Copy latest built .pyd from ./build to project root")
    p_act.set_defaults(func=cmd_activate)

    p_deact = sub.add_parser("deactivate", help="Remove root .pyd to use pure .py")
    p_deact.set_defaults(func=cmd_deactivate)

    p_clean = sub.add_parser("clean", help="Remove .pyds and Nuitka C build folder")
    p_clean.set_defaults(func=cmd_clean)

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(list(argv or sys.argv[1:]))
    return int(ns.func(ns) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
