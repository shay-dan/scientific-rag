"""Local launcher that makes the packaged pipeline importable."""

from pathlib import Path
import sys

import uvicorn

sys.path.insert(0, str(Path(__file__).parent / "src"))

uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
