# Configuration of the foundation backends compatibility tests
# ==============================================================================
# These tests need the backend library of one adapter installed, so they live
# outside the package (not in `testpaths`) and are launched by
# `tools/foundation_compat/run.py`, one adapter at a time.
import tomllib
from pathlib import Path
import pytest
from contracts import BACKEND_CONTRACTS

ADAPTERS_FILE = Path(__file__).resolve().parents[1] / "adapters.toml"


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        required=True,
        help="Entry of tools/foundation_compat/adapters.toml to test.",
    )


def pytest_generate_tests(metafunc):
    """
    Parametrize `contract_call` with the backend calls of the adapter selected
    with `--backend`.
    """
    if "contract_call" in metafunc.fixturenames:
        calls = BACKEND_CONTRACTS.get(metafunc.config.getoption("backend"), [])
        metafunc.parametrize(
            "contract_call", calls, ids=[call["target"] for call in calls]
        )


@pytest.fixture(scope="session")
def backend(request):
    """
    Name of the entry of `adapters.toml` selected with `--backend`.
    """
    return request.config.getoption("backend")


@pytest.fixture(scope="session")
def backend_config(backend):
    """
    Entry of `adapters.toml` selected with `--backend`.
    """
    with open(ADAPTERS_FILE, "rb") as f:
        adapters = tomllib.load(f)

    return adapters[backend]
