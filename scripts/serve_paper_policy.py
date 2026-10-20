"""Run a paper model from its own Python environment, without installing Isaac Sim."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "handson"))
from handson.policies.server import main

if __name__ == "__main__":
    main()
