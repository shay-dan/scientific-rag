"""Run visible checks and save the full terminal output to a timestamped log file."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
import subprocess
import sys


project_root = Path(__file__).resolve().parents[1]
log_directory = project_root / "logs"
log_directory.mkdir(exist_ok=True)
log_path = log_directory / f"test-run-{datetime.now():%Y%m%d-%H%M%S}.log"
environment = os.environ.copy()
environment["PYTHONPATH"] = os.pathsep.join([str(project_root), str(project_root / "src"), environment.get("PYTHONPATH", "")])
command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]

print(f"Running: {' '.join(command)}")
print(f"Writing log: {log_path}")
with log_path.open("w", encoding="utf-8") as log_file:
    process = subprocess.Popen(command, cwd=project_root, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        log_file.write(line)
    exit_code = process.wait()

print(f"\nCheck result: {'PASS' if exit_code == 0 else 'FAIL'}")
sys.exit(exit_code)
