import unittest
from pathlib import Path

import nbformat
from nbformat.validator import validate


class NotebookContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path("notebooks/ML_Tabular_Final.ipynb")
        cls.notebook = nbformat.read(cls.path, as_version=4)

    def test_notebook_is_valid_and_has_no_saved_errors(self):
        validate(self.notebook)
        errors = [
            output
            for cell in self.notebook.cells
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        self.assertEqual(errors, [])

    def test_all_code_cells_have_successful_saved_execution(self):
        code_cells = [cell for cell in self.notebook.cells if cell.cell_type == "code"]
        self.assertGreater(len(code_cells), 0)
        self.assertTrue(all(cell.execution_count is not None for cell in code_cells))

    def test_colab_bootstrap_is_self_contained(self):
        source = "\n".join(cell.source for cell in self.notebook.cells[:4])
        self.assertIn("pip", source)
        self.assertIn("requirements.txt", source)
        self.assertIn("DATA_PUBLIC_URL", source)
        self.assertIn("DATA_SHA256", source)
        self.assertIn("kaggle.com/datasets", source.lower())
        self.assertNotIn("drive.mount", source)


if __name__ == "__main__":
    unittest.main()
