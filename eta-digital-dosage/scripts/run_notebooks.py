from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


def run_notebook(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=600, kernel_name="python3")
    client.execute(cwd=path.parent)
    nbformat.write(notebook, path)
    print(f"executed {path}")


def main() -> None:
    notebook_dir = Path(__file__).resolve().parents[1] / "notebooks"
    for name in ["01_train_model.ipynb", "02_register_mlflow.ipynb", "03_promote_to_production.ipynb"]:
        run_notebook(notebook_dir / name)


if __name__ == "__main__":
    main()
