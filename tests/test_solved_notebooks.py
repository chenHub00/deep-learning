"""Execute exercise notebooks with their solutions filled in.

Exercise notebooks contain TODO blanks and cannot run as shipped. For each
exercise/solution pair below, the TODO cells (by index) are replaced with the
solution's code cells, in order, and the result is executed top to bottom.
"""
import json
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent

# folder, exercise, solution, indices of exercise cells to replace,
# {index: (prefix, suffix)} extra code needed because the solution is a fragment.
CASES = [
    ("linear-regression", "Regression", "RegressionSolutions", [3, 9, 12],
     {9: ("epochs = 1000\nlearning_rate = 0.001\n\n", "\nm, b = linear_regression(x, y)")}),
    ("gradient-descent", "GradientDescent", "GradientDescentSolutions", [5], {}),
    ("student-admissions", "StudentAdmissions", "StudentAdmissionsSolutions", [7, 9, 18], {}),
]


def solved_notebook(folder, exercise, solution, indices, wrap):
    ex = nbformat.read(ROOT / folder / f"{exercise}.ipynb", as_version=4)
    sol = json.loads((ROOT / folder / f"{solution}.ipynb").read_text())
    code = ["".join(c["source"]) for c in sol["cells"]
            if c["cell_type"] == "code" and "".join(c["source"]).strip()]
    assert len(code) == len(indices), "solution cells and TODO cells are out of step"
    for idx, src in zip(indices, code):
        pre, post = wrap.get(idx, ("", ""))
        ex.cells[idx].source = pre + src + post
    return ex


@pytest.mark.parametrize("folder,exercise,solution,indices,wrap", CASES,
                         ids=[c[0] for c in CASES])
def test_exercise_with_solutions(folder, exercise, solution, indices, wrap):
    nb = solved_notebook(folder, exercise, solution, indices, wrap)
    NotebookClient(nb, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT / folder)}}).execute()
