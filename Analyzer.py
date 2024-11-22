import heapq
import json
import os

class Analyzer:
    def __init__(self, preprocessed_train_file, preprocessed_dev_file, preprocessed_test_file):
        self.preprocessed_train_file = preprocessed_train_file
        self.preprocessed_dev_file = preprocessed_dev_file
        self.preprocessed_test_file = preprocessed_test_file

    def _calculate_depth(self, tree):
        """
        Helper function to calculate the depth of a parse tree.
        """
        max_depth, current_depth = 0, 0
        for char in tree:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        return max_depth

    def _calculate_size(self, tree):
        """
        Helper function to calculate the size of a parse tree.
        """
        return tree.count('(')

    def get_top_trees(self, metric="depth", dataset_type="train", top_n=10):
        """
        Returns the top_n instances with the deepest or largest parse trees.
        """
        preprocessed_file = self._get_dataset_file(dataset_type)

        if not os.path.exists(preprocessed_file):
            raise FileNotFoundError(f"Preprocessed file '{preprocessed_file}' not found.")

        min_heap = []
        processed_count = 0

        with open(preprocessed_file, 'r') as infile:
            for index, line in enumerate(infile):
                instance = json.loads(line)
                top = instance.get(f"{dataset_type}.TOP")
                if not top:
                    continue

                if metric == "depth":
                    value = self._calculate_depth(top)
                elif metric == "size":
                    value = self._calculate_size(top)
                else:
                    raise ValueError(f"Invalid metric '{metric}'. Must be 'depth' or 'size'.")

                if len(min_heap) < top_n:
                    heapq.heappush(min_heap, (value, index, instance))
                elif value > min_heap[0][0]:
                    heapq.heapreplace(min_heap, (value, index, instance))

                processed_count += 1
                if processed_count % 1000 == 0:
                    print(f"Processed {processed_count} entries for top trees...")

        print(f"Finished processing {processed_count} entries.")
        return [item[2] for item in sorted(min_heap, key=lambda x: -x[0])]

    def get_samples_with_special_characters(self, dataset_type="train", n=10, special_characters=None):
        """
        Returns the first n samples containing special characters in SRC from preprocessed files.
        """
        if not special_characters:
            print("No special characters provided. Returning an empty list.")
            return []

        preprocessed_file = self._get_dataset_file(dataset_type)

        if not os.path.exists(preprocessed_file):
            raise FileNotFoundError(f"Preprocessed file '{preprocessed_file}' not found.")

        results = []
        processed_count = 0

        with open(preprocessed_file, 'r') as infile:
            for line in infile:
                processed_count += 1
                instance = json.loads(line)
                src = instance.get(f"{dataset_type}.SRC", "")
                if any(char in special_characters for char in src):
                    results.append(instance)
                if len(results) >= n:
                    break

                if processed_count % 1000 == 0:
                    print(f"Processed {processed_count} entries for special characters...")

        print(f"Finished processing {processed_count} entries. Found {len(results)} matching samples.")
        return results

    def find_semantic_constructors(self, dataset_type="train"):
        """
        Finds all unique semantic constructors in the TOP fields of a dataset,
        categorized by their nearest ancestor (PIZZAORDER or DRINKORDER).
        """
        preprocessed_file = self._get_dataset_file(dataset_type)

        if not os.path.exists(preprocessed_file):
            raise FileNotFoundError(f"Preprocessed file '{preprocessed_file}' not found.")

        pizzaorder_constructors = set()
        drinkorder_constructors = set()
        processed_count = 0

        with open(preprocessed_file, 'r') as infile:
            for line in infile:
                processed_count += 1
                instance = json.loads(line)
                top = instance.get(f"{dataset_type}.TOP", "")

                if not top:
                    continue

                stack = []  
                tokens = top.split()

                for token in tokens:
                    if token.startswith("("): 
                        constructor = token[1:] 
                        stack.append(constructor)
                        wrapper = stack[0]
                        if wrapper == "PIZZAORDER": pizzaorder_constructors.add(constructor)
                        elif wrapper == "DRINKORDER": drinkorder_constructors.add(constructor)
                    elif token == ")":  
                        if stack:
                            stack.pop()

                if processed_count % 1000 == 0:
                    print(f"Processed {processed_count} entries for semantic constructors...")

        print(f"Finished processing {processed_count} entries.")
        print(f"Found {len(pizzaorder_constructors)} unique PIZZAORDER constructors and {len(drinkorder_constructors)} unique DRINKORDER constructors.")
        return pizzaorder_constructors, drinkorder_constructors


    def _get_dataset_file(self, dataset_type):
        """
        Helper function to return the appropriate preprocessed dataset file.
        """
        if dataset_type == "train":
            return self.preprocessed_train_file
        elif dataset_type == "dev":
            return self.preprocessed_dev_file
        elif dataset_type == "test":
            return self.preprocessed_test_file
        else:
            raise ValueError("Invalid dataset type. Must be 'train', 'dev', or 'test'.")
