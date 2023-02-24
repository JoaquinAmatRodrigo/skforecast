# Unit test check_optional_dependency
# ==============================================================================
from skforecast.utils import optional_dependencies

def test_skforecast_utils_optional_dependencies_match_requirements_optional():
    """
    Test that check_optional_dependency has the same dependencies than the file
    requirements_optional.txt
    """

    with open('../requirements_optional.txt') as f:
        requirements_optional = f.read()

    requirements_optional = {
        "sarimax": requirements_optional.split("\n\n")[0].splitlines(),
        "plotting": requirements_optional.split("\n\n")[1].splitlines()
    }

    requirements_optional = {k: v[1:] for k, v in requirements_optional.items()}
    assert requirements_optional == optional_dependencies