import tomli
from itertools import chain

with open("./pyproject.toml", mode='rb') as fp:
    pyproject = tomli.load(fp)

dependencies = [v for k, v in pyproject['project']['optional-dependencies'].items() if k != 'docs']
dependencies = chain(*dependencies)
dependencies = set(dependencies)

with open("requirements.txt", mode='w') as fp:
    for dependency in dependencies:
        fp.write(f"{dependency}\n")