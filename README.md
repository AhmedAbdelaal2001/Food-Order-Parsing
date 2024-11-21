# Project Overview

This will be our private repository. The README contains essential notes, instructions, and examples to help you navigate and utilize the code effectively.

## Contributing

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
  - [Special Characters Analysis](#special-characters-analysis)

## Preparing the Dataset

Before starting, download the dataset from the project drive and place it in a directory called `dataset` at the root of the project. The `dataset` folder is included in the `.gitignore` file to prevent it from being pushed to the repository.

## Testing

The `tests` directory contains all Jupyter notebooks used for testing the classes implemented in the root directory. Please ensure that:

- All implementations are encapsulated in clean, well-structured classes.
- Every method in each class is tested in a notebook within the `tests` directory before pushing to the repository.

## Preprocessing the Dataset

To preprocess the dataset and remove the `EXR` and `TOP-DECOUPLED` entries, use the `Preprocessor` class.

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

To clean the dataset and retain only the `SRC` and `TOP` attributes of each entry, use the `filter_and_clean_dataset` method.

```python
# Preprocess the training dataset
preprocessor.filter_and_clean_dataset(dataset_type='train')
```

## Analyzing the Dataset

The `Preprocessor` class provides methods to analyze the dataset, including finding the deepest and largest parse trees and analyzing special characters.

### Top 10 Deepest Trees

The deepest trees are those with the greatest depth in their parse structures.

```python
# Get the top 10 deepest trees in the training dataset
deepest_trees = preprocessor.get_top_trees(metric='depth', dataset_type='train', top_n=10)
```

#### Results:

1. large pie with green pepper and with extra peperonni    
   `(ORDER (PIZZAORDER (SIZE large ) pie with (TOPPING green pepper ) and with (COMPLEX_TOPPING (QUANTITY extra ) (TOPPING peperonni ) ) ) )`

2. i want a stuffed crust pizza with american cheese and a little bit of peperonni  
   `(ORDER i want (PIZZAORDER (NUMBER a ) (STYLE stuffed crust ) pizza with (TOPPING american cheese ) and (COMPLEX_TOPPING (QUANTITY a little bit of ) (TOPPING peperonni ) ) ) )`

3. pie with banana pepper and peppperonis and extra low fat cheese   
   `(ORDER (PIZZAORDER pie with (TOPPING banana pepper ) and (TOPPING peppperonis ) and (COMPLEX_TOPPING (QUANTITY extra ) (TOPPING low fat cheese ) ) ) )`

4. can i have one party sized high rise dough pizza with american cheese and a lot of peperonni    
   `(ORDER can i have (PIZZAORDER (NUMBER one ) (SIZE party sized ) (STYLE high rise dough ) pizza with (TOPPING american cheese ) and (COMPLEX_TOPPING (QUANTITY a lot of ) (TOPPING peperonni ) ) ) )`

5. can i have a party - sized pie without any bean    
   `(ORDER can i have (PIZZAORDER (NUMBER a ) (SIZE party - sized ) pie without any (NOT (TOPPING bean ) ) ) )`

6. can i have one high rise dough pie with american cheese and a lot of meatball   
   `(ORDER can i have (PIZZAORDER (NUMBER one ) (STYLE high rise dough ) pie with (TOPPING american cheese ) and (COMPLEX_TOPPING (QUANTITY a lot of ) (TOPPING meatball ) ) ) )`

7. i want one regular pizza without any fried onions    
   `(ORDER i want (PIZZAORDER (NUMBER one ) (SIZE regular ) pizza without any (NOT (TOPPING fried onions ) ) ) )`

8. party size pie with chicken and mozzarella and with extra sauce  
   `(ORDER (PIZZAORDER (SIZE party size ) pie with (TOPPING chicken ) and (TOPPING mozzarella ) and with (COMPLEX_TOPPING (QUANTITY extra ) (TOPPING sauce ) ) ) )`

9. i'd like a lunch - sized pie without alfredo chicken  
   `(ORDER i'd like (PIZZAORDER (NUMBER a ) (SIZE lunch - sized ) pie without (NOT (TOPPING alfredo chicken ) ) ) )`

10. i'd like a party sized high rise dough pie with a lot of banana pepper and pecorino cheese  
    `(ORDER i'd like (PIZZAORDER (NUMBER a ) (SIZE party sized ) (STYLE high rise dough ) pie with (COMPLEX_TOPPING (QUANTITY a lot of ) (TOPPING banana pepper ) ) and (TOPPING pecorino cheese ) ) )`

### Top 10 Largest Trees

The largest trees are those with the greatest number of nodes in their parse structures.

```python
# Get the top 10 largest trees in the training dataset
largest_trees = preprocessor.get_top_trees(metric='size', dataset_type='train', top_n=10)
```

#### Results:

1. i'd like three pizzas no american cheese and also three cans of ice tea and three medium fantas and three medium sprites  
   `(ORDER i'd like (PIZZAORDER (NUMBER three ) pizzas no (NOT (TOPPING american cheese ) ) ) and also (DRINKORDER (NUMBER three ) (CONTAINERTYPE cans ) of (DRINKTYPE ice tea ) ) and (DRINKORDER (NUMBER three ) (SIZE medium ) (DRINKTYPE fantas ) ) and (DRINKORDER (NUMBER three ) (SIZE medium ) (DRINKTYPE sprites ) ) )`

2. three party sized pizzas with american cheese and a bottle of ice tea and three medium fantas and five medium sprites  
   `(ORDER (PIZZAORDER (NUMBER three ) (SIZE party sized ) pizzas with (TOPPING american cheese ) ) and (DRINKORDER (NUMBER a ) (CONTAINERTYPE bottle ) of (DRINKTYPE ice tea ) ) and (DRINKORDER (NUMBER three ) (SIZE medium ) (DRINKTYPE fantas ) ) and (DRINKORDER (NUMBER five ) (SIZE medium ) (DRINKTYPE sprites ) ) )`

3. three pizzas no american cheese and also a bottle of ice tea and three medium fantas and four medium sprites  
   `(ORDER (PIZZAORDER (NUMBER three ) pizzas no (NOT (TOPPING american cheese ) ) ) and also (DRINKORDER (NUMBER a ) (CONTAINERTYPE bottle ) of (DRINKTYPE ice tea ) ) and (DRINKORDER (NUMBER three ) (SIZE medium ) (DRINKTYPE fantas ) ) and (DRINKORDER (NUMBER four ) (SIZE medium ) (DRINKTYPE sprites ) ) )`

4. three party sized pizzas no american cheese and three cans of ice tea and a ginger ale and a medium san pellegrino   
   `(ORDER (PIZZAORDER (NUMBER three ) (SIZE party sized ) pizzas no (NOT (TOPPING american cheese ) ) ) and (DRINKORDER (NUMBER three ) (CONTAINERTYPE cans ) of (DRINKTYPE ice tea ) ) and (DRINKORDER (NUMBER a ) (DRINKTYPE ginger ale ) ) and (DRINKORDER (NUMBER a ) (SIZE medium ) (DRINKTYPE san pellegrino ) ) )`

5. three pizzas no american cheese and also a bottle of ice tea and three medium fantas and three eight ounce sprites   
   `(ORDER (PIZZAORDER (NUMBER three ) pizzas no (NOT (TOPPING american cheese ) ) ) and also (DRINKORDER (NUMBER a ) (CONTAINERTYPE bottle ) of (DRINKTYPE ice tea ) ) and (DRINKORDER (NUMBER three ) (SIZE medium ) (DRINKTYPE fantas ) ) and (DRINKORDER (NUMBER three ) (VOLUME eight ounce ) (DRINKTYPE sprites ) ) )`

6. four pizzas with not much american cheese and a bottle of ice tea and three fantas and three medium sprites   
   `(ORDER (PIZZAORDER (NUMBER four ) pizzas with (COMPLEX_TOPPING (QUANTITY not much ) (TOPPING american cheese ) ) ) and (DRINKORDER (NUMBER a ) (CONTAINERTYPE bottle ) of (DRINKTYPE ice tea ) ) and (DRINKORDER (NUMBER three ) (DRINKTYPE fantas ) ) and (DRINKORDER (NUMBER three ) (SIZE medium ) (DRINKTYPE sprites ) ) )`

7. three party sized pizzas no american cheese and also a bottle of ice tea and a ginger ale and a medium san pellegrino  
   `(ORDER (PIZZAORDER (NUMBER three ) (SIZE party sized ) pizzas no (NOT (TOPPING american cheese ) ) ) and also (DRINKORDER (NUMBER a ) (CONTAINERTYPE bottle ) of (DRINKTYPE ice tea ) ) and (DRINKORDER (NUMBER a ) (DRINKTYPE ginger ale ) ) and (DRINKORDER (NUMBER a ) (SIZE medium ) (DRINKTYPE san pellegrino ) ) )`

8. three party sized pizzas no american cheese and also three cans of ice tea and a ginger ale and one medium san pellegrino   
   `(ORDER (PIZZAORDER (NUMBER three ) (SIZE party sized ) pizzas no (NOT (TOPPING american cheese ) ) ) and also (DRINKORDER (NUMBER three ) (CONTAINERTYPE cans ) of (DRINKTYPE ice tea ) ) and (DRINKORDER (NUMBER a ) (DRINKTYPE ginger ale ) ) and (DRINKORDER (NUMBER one ) (SIZE medium ) (DRINKTYPE san pellegrino ) ) )`

9. three party sized pizzas no american cheese and also a sprite and three medium fantas and five medium sprites  
   `(ORDER (PIZZAORDER (NUMBER three ) (SIZE party sized ) pizzas no (NOT (TOPPING american cheese ) ) ) and also (DRINKORDER (NUMBER a ) (DRINKTYPE sprite ) ) and (DRINKORDER (NUMBER three ) (SIZE medium ) (DRINKTYPE fantas ) ) and (DRINKORDER (NUMBER five ) (SIZE medium ) (DRINKTYPE sprites ) ) )`

10. three pizzas no american cheese and three cans of ice tea and a medium ginger ale and a medium san pellegrino  
    `(ORDER (PIZZAORDER (NUMBER three ) pizzas no (NOT (TOPPING american cheese ) ) ) and (DRINKORDER (NUMBER three ) (CONTAINERTYPE cans ) of (DRINKTYPE ice tea ) ) and (DRINKORDER (NUMBER a ) (SIZE medium ) (DRINKTYPE ginger ale ) ) and (DRINKORDER (NUMBER a ) (SIZE medium ) (DRINKTYPE san pellegrino ) ) )`

### Special Characters Analysis

Analysis of the dataset for the presence of special characters.

- **Special Characters NOT Present in the train/dev sets:**

  ```
  ! @ # $ % ^ & * ( ) _ + = [ ] { } | ; : " , . < > ? / ` ~
  ```

- **Special Characters Present in the train/dev sets:**

  ```
  ' -
  ```


