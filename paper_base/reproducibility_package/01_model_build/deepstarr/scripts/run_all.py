#!/usr/bin/env python
# coding: utf-8
"""Run polished Figure 1 plotting scripts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPTS = ["fig1a.py", "fig1b.py", "fig1c.py"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-logo", action="store_true", help="Skip panels that require logomaker/h5py.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    for script in SCRIPTS:
        print(f"\n== {script} ==")
        subprocess.run([sys.executable, str(root / script)], check=True)


if __name__ == "__main__":
    main()
