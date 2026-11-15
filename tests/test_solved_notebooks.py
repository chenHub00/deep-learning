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


# Exercise/solution pairs with identical cell layout: every exercise cell that
# contains "TODO" is replaced by the solution cell at the same index.
PAIRED = [
    ("intro-to-pytorch", "Digit_Recognition_PyTorch", "Digit_Recognition_PyTorch_Solution"),
    ("intro-to-pytorch", "Sentiment_Analysis_PyTorch", "Sentiment_Analysis_PyTorch_Solution"),
]


@pytest.mark.parametrize("folder,exercise,solution", PAIRED, ids=[p[1] for p in PAIRED])
def test_paired_exercise_with_solutions(folder, exercise, solution):
    ex = nbformat.read(ROOT / folder / f"{exercise}.ipynb", as_version=4)
    sol = nbformat.read(ROOT / folder / f"{solution}.ipynb", as_version=4)
    assert len(ex.cells) == len(sol.cells), "exercise and solution have different layouts"
    todo = [i for i, c in enumerate(ex.cells) if c.cell_type == "code" and "TODO" in c.source]
    assert todo, "no TODO cells found"
    for i in todo:
        ex.cells[i].source = sol.cells[i].source
    NotebookClient(ex, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT / folder)}}).execute()


@pytest.mark.parametrize("folder,exercise,solution,indices,wrap", CASES,
                         ids=[c[0] for c in CASES])
def test_exercise_with_solutions(folder, exercise, solution, indices, wrap):
    nb = solved_notebook(folder, exercise, solution, indices, wrap)
    NotebookClient(nb, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT / folder)}}).execute()
