# conftest.py
import os


def pytest_generate_tests(metafunc):
    os.environ["KERAS_BACKEND"] = "torch"
