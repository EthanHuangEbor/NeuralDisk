from __future__ import annotations

import argparse

from hyb_nurbs.config import load_config
from hyb_nurbs.pipeline import run_pipeline


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="HYB topology density to NURBS reconstruction")
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
