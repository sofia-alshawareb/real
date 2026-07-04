"""Worker CLI entrypoint."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for _env_key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_env_key, "1")

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ML segmentation worker")
    parser.add_argument(
        "--config",
        default=str(_ROOT / "configs/ml_service.yaml"),
        help="Path to ml_service.yaml",
    )
    parser.add_argument(
        "--save-activations",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override dino.save_activations from config",
    )
    parser.add_argument(
        "--host", default=os.environ.get("ML_API_HOST", "0.0.0.0")
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("ML_API_PORT", "8000")),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["ML_SERVICE_CONFIG"] = args.config
    if args.save_activations is not None:
        os.environ["ML_SAVE_ACTIVATIONS"] = str(args.save_activations).lower()

    from services.api.main import main as run_api

    run_api()


if __name__ == "__main__":
    main()
