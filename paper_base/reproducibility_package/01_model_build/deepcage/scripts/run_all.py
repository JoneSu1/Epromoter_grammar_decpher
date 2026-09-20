#!/usr/bin/env python
"""Run the package-local DeepCAGE Figure 2 scripts."""

from __future__ import annotations

import fig2c
import fig2f


def main() -> None:
    fig2c.main()
    fig2f.main()


if __name__ == "__main__":
    main()
