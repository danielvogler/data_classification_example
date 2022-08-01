# data_classification_example
Exemplary data classification

## Setup

### Python environment
- Poetry is used for the virtual Python environment.
  - Install [python poetry](https://github.com/python-poetry/poetry):
    ```bash
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
    ```
    or
    ```bash
    pip install --user poetry
    ```

  - Make sure that the virutal environment is installed in the root folder of this projects:
    ```bash
    poetry config virtualenvs.in-project true
    ```

  - Install dependencies:
    ```bash
    poetry install --no-root
    ```

  - Add packages:
    To add further packages, run:
    ```bash
    poetry add <package-name>
    ```