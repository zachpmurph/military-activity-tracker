import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.db import init_db, resolve_db_path


class DatabasePathTests(unittest.TestCase):
    def test_resolve_db_path_defaults_to_src_data(self):
        expected = Path(__file__).resolve().parents[1] / "src" / "data" / "aircraft.db"
        self.assertEqual(resolve_db_path(), expected)

    def test_init_db_creates_missing_parent_directories_for_custom_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "nested" / "db" / "aircraft.db"
            conn, _cursor = init_db(db_path)
            conn.close()

            self.assertTrue(db_path.parent.exists())
            self.assertTrue(db_path.exists())


if __name__ == "__main__":
    unittest.main()
