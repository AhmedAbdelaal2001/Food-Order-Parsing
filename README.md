# Project Overview

This is our private repo for the NLP project. This README should contain summarized updates; check it regularly.

## Note

Please ensure that any changes or additions adhere to clean coding standards. Before pushing to the repository:

- **Testing:** Verify that all new methods are thoroughly tested in the `tests` directory.
- **Documentation:** Update this README or add docstrings to your code where necessary.
- **Clean Code:** Maintain clean, readable, and well-commented code.

## Table of Contents

- [Preparing the Dataset](#preparing-the-dataset)
- [Testing](#testing)
- [Preprocessing the Dataset](#preprocessing-the-dataset)
  - [Initializing the Preprocessor](#initializing-the-preprocessor)
  - [Cleaning the Dataset](#cleaning-the-dataset)
- [Analyzing the Dataset](#analyzing-the-dataset)
  - [Top 10 Deepest Trees](#top-10-deepest-trees)
  - [Top 10 Largest Trees](#top-10-largest-trees)
  - [Semantic Constructors Analysis](#semantic-constructors-analysis)
  - [Special Characters Analysis](#special-characters-analysis)

---

## Preparing the Dataset

Before starting, download the dataset from the project drive and place it in a directory called `dataset` at the root of the project. The `dataset` folder is included in the `.gitignore` file to prevent it from being pushed to the repository; do not change that.

---

## Testing

The `tests` directory contains all Jupyter notebooks used for testing the classes implemented in the root directory. Please ensure that:

- All implementations are encapsulated in clean, well-structured classes.
- Every method in each class is tested in a notebook within the `tests` directory before pushing to the repository.

---

## Preprocessing the Dataset

The `Preprocessor` class is responsible for cleaning and restructuring the dataset. Specifically, it:
1. Removes the `EXR` and `TOP-DECOUPLED` fields from each entry.
2. Removes the redundant `ORDER` semantic constructor, which is present in every entry and adds no useful information for future models.
3. Retains only the `SRC` and cleaned `TOP` fields.

### Initializing the Preprocessor

Initialize the `Preprocessor` class by providing the paths of the dataset files (`train`, `dev`, `test`) and the desired output paths for the preprocessed datasets. All paths are relative to the project's root directory.

```python
from preprocessor import Preprocessor

# Initialize the Preprocessor
preprocessor = Preprocessor(
    train_file='dataset/train.json',
    dev_file='dataset/dev.json',
    test_file='dataset/test.json',
    preprocessed_train_file='preprocessed/train_preprocessed.json',
    preprocessed_dev_file='preprocessed/dev_preprocessed.json',
    preprocessed_test_file='preprocessed/test_preprocessed.json'
)
```

### Cleaning the Dataset

To clean the dataset, use the `filter_and_clean_dataset` method. This method processes the dataset by removing unnecessary fields and simplifying the structure.

```python
# Preprocess the training dataset
preprocessor.filter_and_clean_dataset(dataset_type='train')
```

You can specify other dataset types (`"dev"` or `"test"`) depending on the dataset you wish to clean.

---

## Analyzing the Dataset

The `Analyzer` class provides tools for analyzing the hierarchical structure and content of datasets. Its features include:

1. Measuring tree properties such as **depth** and **size**.
2. Analyzing entries for the presence of special characters.
3. Extracting and categorizing **semantic constructors** associated with `PIZZAORDER` and `DRINKORDER`.

### Top 3 Deepest Trees

The deepest trees are those with the greatest depth in their hierarchical structure. The `ORDER` constructor is removed for clarity.

```python
# Get the top 10 deepest trees in the training dataset
from analyzer import Analyzer

analyzer = Analyzer(
    preprocessed_train_file='preprocessed/train_preprocessed.json',
    preprocessed_dev_file='preprocessed/dev_preprocessed.json',
    preprocessed_test_file='preprocessed/test_preprocessed.json'
)

deepest_trees = analyzer.get_top_trees(metric='depth', dataset_type='train', top_n=3)
```

#### Example Results:

1. `(PIZZAORDER (SIZE large ) pie with (TOPPING green pepper ) and with (COMPLEX_TOPPING (QUANTITY extra ) (TOPPING peperonni ) ) )`
2. `(PIZZAORDER (NUMBER a ) (STYLE stuffed crust ) pizza with (TOPPING american cheese ) and (COMPLEX_TOPPING (QUANTITY a little bit of ) (TOPPING peperonni ) ) )`
3. `(PIZZAORDER pie with (TOPPING banana pepper ) and (TOPPING peppperonis ) and (COMPLEX_TOPPING (QUANTITY extra ) (TOPPING low fat cheese ) ) )`

### Top 2 Largest Trees

The largest trees are those with the greatest number of nodes in their structure. The `ORDER` constructor is removed for clarity.

```python
# Get the top 2 largest trees in the training dataset
largest_trees = analyzer.get_top_trees(metric='size', dataset_type='train', top_n=2)
```

#### Example Results:

1. `(PIZZAORDER (NUMBER three ) pizzas no (NOT (TOPPING american cheese ) ) and also (DRINKORDER (NUMBER three ) (CONTAINERTYPE cans ) of (DRINKTYPE ice tea ) ) and (DRINKORDER (NUMBER three ) (SIZE medium ) (DRINKTYPE fantas ) ) and (DRINKORDER (NUMBER three ) (SIZE medium ) (DRINKTYPE sprites ) ) )`
2. `(PIZZAORDER (NUMBER three ) pizzas no (NOT (TOPPING american cheese ) ) and also (DRINKORDER (NUMBER a ) (CONTAINERTYPE bottle ) of (DRINKTYPE ice tea ) ) and (DRINKORDER (NUMBER three ) (SIZE medium ) (DRINKTYPE fantas ) ) and (DRINKORDER (NUMBER four ) (SIZE medium ) (DRINKTYPE sprites ) ) )`
---

### Semantic Constructors Analysis

Semantic constructors are unique tokens that represent structural elements in the dataset. They are categorized based on their association with `PIZZAORDER` or `DRINKORDER`. Defining them is important for finetuning BART.

#### Associated Semantic Constructors:

- **PIZZAORDER Constructors**:
  ```
  {'COMPLEX_TOPPING', 'NOT', 'NUMBER', 'QUANTITY', 'SIZE', 'STYLE', 'TOPPING'}
  ```

- **DRINKORDER Constructors**:
  ```
  {'CONTAINERTYPE', 'DRINKTYPE', 'NUMBER', 'SIZE', 'VOLUME'}
  ```

- **Union of Constructors**:
  ```
  {'COMPLEX_TOPPING', 'CONTAINERTYPE', 'DRINKORDER', 'DRINKTYPE', 'NOT', 'NUMBER', 'PIZZAORDER', 'QUANTITY', 'SIZE', 'STYLE', 'TOPPING', 'VOLUME'}
  ```

- **Total Possible Constructors**:
  12 unique constructors.

---

### Special Characters Analysis

Analyze the dataset for the presence of special characters.

#### Example Analysis:

- **Special Characters NOT Present**:
  ```
  ! @ # $ % ^ & * ( ) _ + = [ ] { } | ; : " , . < > ? / ` ~
  ```

- **Special Characters Present**:
  ```
  ' -
  ```
Note: the project document specified that we should remove special characters like @, but they do not exist in the first place in either the train or dev sets xD




  
