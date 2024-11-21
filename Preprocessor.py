import os
import heapq
import json

class Preprocessor:
    def __init__(self, train_file, dev_file, test_file, preprocessed_train_file, preprocessed_dev_file, preprocessed_test_file):
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.preprocessed_train_file = preprocessed_train_file
        self.preprocessed_dev_file = preprocessed_dev_file
        self.preprocessed_test_file = preprocessed_test_file

    def _get_dataset_files(self, dataset_type):
        """
        Helper function to return input and output file paths based on dataset type.
        """
        if dataset_type == "train":
            return self.train_file, self.preprocessed_train_file
        elif dataset_type == "dev":
            return self.dev_file, self.preprocessed_dev_file
        elif dataset_type == "test":
            return self.test_file, self.preprocessed_test_file
        else:
            raise ValueError("Invalid dataset type. Must be 'train', 'dev', or 'test'.")
    
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

    def filter_and_clean_dataset(self, dataset_type="train", start=0, end=None):
        """
        Cleans the dataset file, retaining only 'SRC' and 'TOP' attributes of each entry.
        """
        input_file, output_file = self._get_dataset_files(dataset_type)

        if not os.path.exists(input_file):
            raise FileNotFoundError(f"The input file '{input_file}' does not exist.")

        if start < 0:
            start = 0

        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            batch = []
            line_count = 0
            processed_count = 0

            for line in infile:
                line_count += 1
                if line_count < start + 1:
                    continue
                if end is not None and line_count > end:
                    break

                instance = json.loads(line)
                cleaned_instance = {f"{dataset_type}.SRC": instance.get(f"{dataset_type}.SRC"),
                                    f"{dataset_type}.TOP": instance.get(f"{dataset_type}.TOP")}
                batch.append(cleaned_instance)

                if len(batch) >= 1000:
                    for item in batch:
                        outfile.write(json.dumps(item) + '\n')
                    processed_count += len(batch)
                    print(f"Processed {processed_count} entries so far.")
                    batch = []

            for item in batch:
                outfile.write(json.dumps(item) + '\n')
            processed_count += len(batch)
            if batch:
                print(f"Processed {processed_count} entries so far.")

        print(f"Completed processing {processed_count} entries from {input_file} to {output_file}.")

    def get_top_trees(self, metric="depth", dataset_type="train", top_n=10):
        """
        Returns the top_n instances with the deepest or largest parse trees.
        
        Parameters:
        - metric (str): The metric to use ("depth" or "size").
        - dataset_type (str): The type of dataset ('train', 'dev', 'test').
        - top_n (int): The number of top entries to return.
        """
        _, preprocessed_file = self._get_dataset_files(dataset_type)

        if not os.path.exists(preprocessed_file):
            print(f"Preprocessed file '{preprocessed_file}' not found. Generating it from the input file...")
            self.filter_and_clean_dataset(dataset_type)

        if not os.path.exists(preprocessed_file):
            raise FileNotFoundError(f"Neither preprocessed file '{preprocessed_file}' nor the input file exists.")

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
                    print(f"Processed {processed_count} entries...")

        print(f"Finished processing {processed_count} entries.")

        top_trees = sorted(min_heap, key=lambda x: -x[0])
        return [item[2] for item in top_trees]
    
    def get_samples_with_special_characters(self, dataset_type="train", n=10, special_characters=None):
        """
        Returns the first n samples containing special characters in SRC from preprocessed files.
        
        Parameters:
        - dataset_type (str): The type of dataset ('train', 'dev', 'test').
        - n (int): The number of samples to return.
        - special_characters (str): A string of special characters to check. If None, returns an empty list.
        """
        if not special_characters:
            print("No special characters provided. Returning an empty list.")
            return []

        _, preprocessed_file = self._get_dataset_files(dataset_type)

        if not os.path.exists(preprocessed_file):
            print(f"Preprocessed file '{preprocessed_file}' not found. Generating it from the input file...")
            self.filter_and_clean_dataset(dataset_type)

        if not os.path.exists(preprocessed_file):
            raise FileNotFoundError(f"Neither preprocessed file '{preprocessed_file}' nor the input file exists.")

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
                    print(f"Processed {processed_count} entries...")

        print(f"Finished processing {processed_count} entries.")
        return results





