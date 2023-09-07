import tomli
from itertools import chain

with open("./pyproject.toml", mode='rb') as fp:
    pyproject = tomli.load(fp)

dependencies_optional = [v for k, v in pyproject['project']['optional-dependencies'].items() if k != 'docs']
dependencies_optional = chain(*dependencies_optional)
dependencies_optional = list(set(dependencies_optional))
dependences = pyproject['project']['dependencies']
dependences_all = dependences + dependencies_optional
dependences_all

with open("requirements.txt", mode='w') as fp:
    for dependency in dependences_all:
        fp.write(f"{dependency}\n")
