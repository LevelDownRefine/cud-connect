"""Allow `python -m cud_connect`."""

import os

# Cap nested numerical threads before numpy is imported: subjects run in separate processes.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

from .cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
