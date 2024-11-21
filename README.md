# Overview
This will be our private repo, any notes of interest will be written in this README file.

## Preparing the Dataset
Before doing anything, you must download the dataset given on the project drive and place it in a directory called "dataset". I placed this folder in the .gitignore file, so it will never be pushed here.

## Testing
You will find a directory called "tests". This is important; it will contain all Jupyter notebooks used for testing the classes implemented in the root directory. Please, make sure that all your implementations are in clean classes, and that every method in every class is tested in a notebook in the "tests" directory before pushing to this repo.

## Preprocessing the Dataset
If you wish to preprocess the dataset and remove the EXR and TOP-DECOUPLED entries, you can just use the Preprocessor class. Initialize the class by giving its constructor the paths of each file in the dataset (train, dev, test), as well as the paths that you wish to save the output preprocessed datasets at (all paths are from the root directory of the project).

## Analyzing the Dataset
The Preprocessor class contains some methods that can be used to analyze the dataset; including finding the deepest/largest trees, and analyzing special characters. I shall list some notable results down here:

### Top 10 Deepest Trees:
[{'train.SRC': 'large pie with green pepper and with extra peperonni',
  'train.TOP': '(ORDER (PIZZAORDER (SIZE large ) pie with (TOPPING green pepper ) and with (COMPLEX_TOPPING (QUANTITY extra ) (TOPPING peperonni ) ) ) )'},
 {'train.SRC': 'i want a stuffed crust pizza with american cheese and a little bit of peperonni',
  'train.TOP': '(ORDER i want (PIZZAORDER (NUMBER a ) (STYLE stuffed crust ) pizza with (TOPPING american cheese ) and (COMPLEX_TOPPING (QUANTITY a little bit of ) (TOPPING peperonni ) ) ) )'},
 {'train.SRC': 'pie with banana pepper and peppperonis and extra low fat cheese',
  'train.TOP': '(ORDER (PIZZAORDER pie with (TOPPING banana pepper ) and (TOPPING peppperonis ) and (COMPLEX_TOPPING (QUANTITY extra ) (TOPPING low fat cheese ) ) ) )'},
 {'train.SRC': 'can i have one party sized high rise dough pizza with american cheese and a lot of peperonni',
  'train.TOP': '(ORDER can i have (PIZZAORDER (NUMBER one ) (SIZE party sized ) (STYLE high rise dough ) pizza with (TOPPING american cheese ) and (COMPLEX_TOPPING (QUANTITY a lot of ) (TOPPING peperonni ) ) ) )'},
 {'train.SRC': 'can i have a party - sized pie without any bean',
  'train.TOP': '(ORDER can i have (PIZZAORDER (NUMBER a ) (SIZE party - sized ) pie without any (NOT (TOPPING bean ) ) ) )'},
 {'train.SRC': 'can i have one high rise dough pie with american cheese and a lot of meatball',
  'train.TOP': '(ORDER can i have (PIZZAORDER (NUMBER one ) (STYLE high rise dough ) pie with (TOPPING american cheese ) and (COMPLEX_TOPPING (QUANTITY a lot of ) (TOPPING meatball ) ) ) )'},
 {'train.SRC': 'i want one regular pizza without any fried onions',
  'train.TOP': '(ORDER i want (PIZZAORDER (NUMBER one ) (SIZE regular ) pizza without any (NOT (TOPPING fried onions ) ) ) )'},
 {'train.SRC': 'party size pie with chicken and mozzarella and with extra sauce',
  'train.TOP': '(ORDER (PIZZAORDER (SIZE party size ) pie with (TOPPING chicken ) and (TOPPING mozzarella ) and with (COMPLEX_TOPPING (QUANTITY extra ) (TOPPING sauce ) ) ) )'},
 {'train.SRC': "i'd like a lunch - sized pie without alfredo chicken",
  'train.TOP': "(ORDER i'd like (PIZZAORDER (NUMBER a ) (SIZE lunch - sized ) pie without (NOT (TOPPING alfredo chicken ) ) ) )"},
 {'train.SRC': "i'd like a party sized high rise dough pie with a lot of banana pepper and pecorino cheese",
  'train.TOP': "(ORDER i'd like (PIZZAORDER (NUMBER a ) (SIZE party sized ) (STYLE high rise dough ) pie with (COMPLEX_TOPPING (QUANTITY a lot of ) (TOPPING banana pepper ) ) and (TOPPING pecorino cheese ) ) )"}]


### Top 10 Largest Trees:
[{'train.SRC': "i'd like three pizzas no american cheese and also three cans of ice tea and three medium fantas and three medium sprites",
  'train.TOP': "(ORDER i'd like (PIZZAORDER (NUMBER three ) pizzas no (NOT (TOPPING american cheese ) ) ) and also (DRINKORDER (NUMBER three ) (CONTAINERTYPE cans ) of (DRINKTYPE ice tea ) ) and (DRINKORDER (NUMBER three ) (SIZE medium ) (DRINKTYPE fantas ) ) and (DRINKORDER (NUMBER three ) (SIZE medium ) (DRINKTYPE sprites ) ) )"},
 {'train.SRC': 'three party sized pizzas with american cheese and a bottle of ice tea and three medium fantas and five medium sprites',
  'train.TOP': '(ORDER (PIZZAORDER (NUMBER three ) (SIZE party sized ) pizzas with (TOPPING american cheese ) ) and (DRINKORDER (NUMBER a ) (CONTAINERTYPE bottle ) of (DRINKTYPE ice tea ) ) and (DRINKORDER (NUMBER three ) (SIZE medium ) (DRINKTYPE fantas ) ) and (DRINKORDER (NUMBER five ) (SIZE medium ) (DRINKTYPE sprites ) ) )'},
 {'train.SRC': 'three pizzas no american cheese and also a bottle of ice tea and three medium fantas and four medium sprites',
  'train.TOP': '(ORDER (PIZZAORDER (NUMBER three ) pizzas no (NOT (TOPPING american cheese ) ) ) and also (DRINKORDER (NUMBER a ) (CONTAINERTYPE bottle ) of (DRINKTYPE ice tea ) ) and (DRINKORDER (NUMBER three ) (SIZE medium ) (DRINKTYPE fantas ) ) and (DRINKORDER (NUMBER four ) (SIZE medium ) (DRINKTYPE sprites ) ) )'},
 {'train.SRC': 'three party sized pizzas no american cheese and three cans of ice tea and a ginger ale and a medium san pellegrino',
  'train.TOP': '(ORDER (PIZZAORDER (NUMBER three ) (SIZE party sized ) pizzas no (NOT (TOPPING american cheese ) ) ) and (DRINKORDER (NUMBER three ) (CONTAINERTYPE cans ) of (DRINKTYPE ice tea ) ) and (DRINKORDER (NUMBER a ) (DRINKTYPE ginger ale ) ) and (DRINKORDER (NUMBER a ) (SIZE medium ) (DRINKTYPE san pellegrino ) ) )'},
 {'train.SRC': 'three pizzas no american cheese and also a bottle of ice tea and three medium fantas and three eight ounce sprites',
  'train.TOP': '(ORDER (PIZZAORDER (NUMBER three ) pizzas no (NOT (TOPPING american cheese ) ) ) and also (DRINKORDER (NUMBER a ) (CONTAINERTYPE bottle ) of (DRINKTYPE ice tea ) ) and (DRINKORDER (NUMBER three ) (SIZE medium ) (DRINKTYPE fantas ) ) and (DRINKORDER (NUMBER three ) (VOLUME eight ounce ) (DRINKTYPE sprites ) ) )'},
 {'train.SRC': 'four pizzas with not much american cheese and a bottle of ice tea and three fantas and three medium sprites',
  'train.TOP': '(ORDER (PIZZAORDER (NUMBER four ) pizzas with (COMPLEX_TOPPING (QUANTITY not much ) (TOPPING american cheese ) ) ) and (DRINKORDER (NUMBER a ) (CONTAINERTYPE bottle ) of (DRINKTYPE ice tea ) ) and (DRINKORDER (NUMBER three ) (DRINKTYPE fantas ) ) and (DRINKORDER (NUMBER three ) (SIZE medium ) (DRINKTYPE sprites ) ) )'},
 {'train.SRC': 'three party sized pizzas no american cheese and also a bottle of ice tea and a ginger ale and a medium san pellegrino',
  'train.TOP': '(ORDER (PIZZAORDER (NUMBER three ) (SIZE party sized ) pizzas no (NOT (TOPPING american cheese ) ) ) and also (DRINKORDER (NUMBER a ) (CONTAINERTYPE bottle ) of (DRINKTYPE ice tea ) ) and (DRINKORDER (NUMBER a ) (DRINKTYPE ginger ale ) ) and (DRINKORDER (NUMBER a ) (SIZE medium ) (DRINKTYPE san pellegrino ) ) )'},
 {'train.SRC': 'three party sized pizzas no american cheese and also three cans of ice tea and a ginger ale and one medium san pellegrino',
  'train.TOP': '(ORDER (PIZZAORDER (NUMBER three ) (SIZE party sized ) pizzas no (NOT (TOPPING american cheese ) ) ) and also (DRINKORDER (NUMBER three ) (CONTAINERTYPE cans ) of (DRINKTYPE ice tea ) ) and (DRINKORDER (NUMBER a ) (DRINKTYPE ginger ale ) ) and (DRINKORDER (NUMBER one ) (SIZE medium ) (DRINKTYPE san pellegrino ) ) )'},
 {'train.SRC': 'three party sized pizzas no american cheese and also a sprite and three medium fantas and five medium sprites',
  'train.TOP': '(ORDER (PIZZAORDER (NUMBER three ) (SIZE party sized ) pizzas no (NOT (TOPPING american cheese ) ) ) and also (DRINKORDER (NUMBER a ) (DRINKTYPE sprite ) ) and (DRINKORDER (NUMBER three ) (SIZE medium ) (DRINKTYPE fantas ) ) and (DRINKORDER (NUMBER five ) (SIZE medium ) (DRINKTYPE sprites ) ) )'},
 {'train.SRC': 'three pizzas no american cheese and three cans of ice tea and a medium ginger ale and a medium san pellegrino',
  'train.TOP': '(ORDER (PIZZAORDER (NUMBER three ) pizzas no (NOT (TOPPING american cheese ) ) ) and (DRINKORDER (NUMBER three ) (CONTAINERTYPE cans ) of (DRINKTYPE ice tea ) ) and (DRINKORDER (NUMBER a ) (SIZE medium ) (DRINKTYPE ginger ale ) ) and (DRINKORDER (NUMBER a ) (SIZE medium ) (DRINKTYPE san pellegrino ) ) )'}]

### Special Characters NOT Present in the train/dev sets: !@#$%^&*()_+=[]{}|;:\",.<>?/`~
### Special Characters Present in the train/dev sets: '-

