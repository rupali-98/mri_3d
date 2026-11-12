import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Tests don't write to MLflow unless they check the logging (they set HIPPO_MLFLOW=1 themselves).
os.environ["HIPPO_MLFLOW"] = "0"
