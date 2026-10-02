from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def synthetic_xml() -> bytes:
    return (FIXTURES / "boe_synthetic.xml").read_bytes()
