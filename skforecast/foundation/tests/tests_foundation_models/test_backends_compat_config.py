# Unit test configuration of the foundation backends compatibility check
# ==============================================================================
import re
from pathlib import Path
import pytest
from skforecast.foundation._adapters import _ADAPTER_REGISTRY, _resolve_adapter

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[4]
ADAPTERS_FILE = REPO_ROOT / "tools" / "foundation_compat" / "adapters.toml"
WORKFLOW_FILE = REPO_ROOT / ".github" / "workflows" / "foundation-backends.yml"

pytestmark = pytest.mark.skipif(
    not ADAPTERS_FILE.exists(),
    reason="tools/foundation_compat is only available in the repository.",
)


def load_adapters():
    """
    Return the content of `tools/foundation_compat/adapters.toml`.
    """
    with open(ADAPTERS_FILE, "rb") as f:
        return tomllib.load(f)


def test_adapters_toml_covers_every_registered_adapter():
    """
    Test that every adapter class in the registry has at least one entry in
    adapters.toml, so a new adapter cannot be left out of the compatibility
    check.
    """
    adapters = load_adapters()

    registered = {cls.__name__ for cls in _ADAPTER_REGISTRY.values()}
    configured = {config["adapter"] for config in adapters.values()}

    assert configured == registered


def test_adapters_toml_entries_match_the_adapter_classes():
    """
    Test that, for every entry in adapters.toml, `model_id` is routed to the
    adapter class named in `adapter`, `backend_package` is the one declared by
    that class, and the pinned variant is fully defined.
    """
    adapters = load_adapters()

    for name, config in adapters.items():
        adapter_cls = _resolve_adapter(config["model_id"])
        assert adapter_cls.__name__ == config["adapter"], name
        assert adapter_cls.backend_package == config["backend_package"], name
        assert config["pinned"], name
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", config["exclude_newer"]), name


def test_workflow_matrix_matches_adapters_toml():
    """
    Test that the matrix of the workflow foundation-backends.yml lists the
    same entries as adapters.toml.
    """
    adapters = load_adapters()

    match = re.search(r"^\s+adapter: \[(.*)\]$", WORKFLOW_FILE.read_text(), re.M)
    matrix = [name.strip() for name in match.group(1).split(",")]

    assert sorted(matrix) == sorted(adapters)
