# Contributing to Skforecast

## How to Contribute

Skforecast is an open source project supported by a community that will gratefully and humbly accept any contribution you can make to the project. Big or small, any contribution makes a big difference; and if you have never contributed to an open source project before, we hope you will start with Skforecast!

Primarily, Skforecast development consists of adding and creating new *Forecasters*, new validation strategies or improving the performance of the current code. However, there are many other ways to contribute:


- Submit a bug report or feature request on [GitHub Issues](https://github.com/JoaquinAmatRodrigo/skforecast/issues).
- Contribute a Jupyter notebook to our [examples](https://joaquinamatrodrigo.github.io/skforecast/0.7.0/examples/examples.html).
- Write [unit or integration tests]() for our project.
- Answer questions on our issues, Stack Overflow, and elsewhere.
- Translate our documentation into another language.
- Write a blog post, tweet, or share our project with others.

As you can see, there are lots of ways to get involved and we would be very happy for you to join us! Before you start, please open an issue with with a brief description of the proposal so we can align all together.


## Testing

To run the test suite, first install the testing dependencies that are located in main folder:

```
$ pip install -r requirements_test.txt
```

All unit test can be run at once as follows from the project root:

```
$ pytest -vv
```

The tests do take a while to run, so during normal development it is recommended that you only run the tests for the test file you are writing:

```
$ pytest new_module/tests/test_module.py
```

This will help a lot to ensure that new code do not affect the already existent functionalities of the library.