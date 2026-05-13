import runpy
import sys
import unittest
from pathlib import Path


class PipelineEntrypointTests(unittest.TestCase):
    def test_pipeline_script_loads_without_module_import_error(self):
        pipeline_path = Path(__file__).resolve().parents[1] / "src" / "ingest" / "pipeline.py"
        original_path = list(sys.path)
        try:
            namespace = runpy.run_path(str(pipeline_path), run_name="__pipeline_test__")
        finally:
            sys.path[:] = original_path

        self.assertIn("main", namespace)


if __name__ == "__main__":
    unittest.main()
