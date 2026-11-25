"""Local UI assets must work without external CDN access (offline test)."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "app" / "templates" / "index.html"
HTMX = ROOT / "app" / "static" / "vendor" / "htmx-2.0.4.min.js"
LICENSE = ROOT / "app" / "static" / "vendor" / "htmx-LICENSE"


class UiAssetTests(unittest.TestCase):
    def test_research_ui_loads_local_htmx_instead_of_cdn(self):
        template = TEMPLATE.read_text(encoding="utf-8")
        self.assertIn('src="/static/vendor/htmx-2.0.4.min.js"', template)
        self.assertNotIn('src="https://unpkg.com/', template)
        self.assertNotIn('src="https://cdn.jsdelivr.net/', template)
        self.assertTrue(HTMX.exists())
        self.assertGreater(HTMX.stat().st_size, 45000)
        self.assertIn("var htmx=function()", HTMX.read_text(encoding="utf-8")[:80])
        self.assertIn("Zero-Clause BSD", LICENSE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
