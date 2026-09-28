import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing import clean_text

def test_clean_text():
    assert clean_text("Hello!!! CUSTOMER 123") == "hello customer"

def test_spaces():
    assert clean_text("  billing    issue  ") == "billing issue"
