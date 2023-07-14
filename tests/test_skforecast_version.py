from skforecast import __version__

def test_version(version="0.9.2"):

    with open('pyproject.toml') as f:
        pyproject = f.read().splitlines()

    # Find the element in the list that starts with "version ="
    for element in pyproject:
        if element.startswith('version ='):
            # Split the element into a list of strings
            version_list = element.split()
            # Get the second element of the list, i.e. the version number
            version_in_tolm = version_list[2][1:-1]
        break

    assert __version__ ==  version_in_tolm == version


# Test are located inside each module's folder