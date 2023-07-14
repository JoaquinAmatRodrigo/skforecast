# Unit test check_optional_dependency
# ==============================================================================
from skforecast.utils import optional_dependencies

def test_skforecast_utils_optional_dependencies_match_requirements_optional():
    """
    Test that check_optional_dependency has the same dependencies than the file
    requirements_optional.txt
    """

    with open('pyproject.toml') as f:
        pyproject = f.read().splitlines()

    # Find the element in the list that starts with sarimax
    for element in pyproject:
        if element.startswith('sarimax'):
            # Split the element into a list of strings
            sarimax_list = element.split(" = ")
            # Get the second element of the list, i.e. the version number
            sarimax_dependences = sarimax_list[1][2:-2]
            break
    sarimax_dependences

    # Find the element in the list that starts with sarimax

    requirements_optional = {
        "sarimax": sarimax_dependences,
        "plotting": 
    }

    requirements_optional = {k: v[1:] for k, v in requirements_optional.items()}
    assert requirements_optional == optional_dependencies