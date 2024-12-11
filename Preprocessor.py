import os
import json
import re
import heapq
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

    def _remove_exr_and_top_decoupled(self, entry, dataset_type):
        """
        Removes 'EXR' and 'TOP-DECOUPLED' fields from a dataset entry.
        """
        entry.pop(f"{dataset_type}.EXR", None)
        entry.pop(f"{dataset_type}.TOP-DECOUPLED", None)
        return entry

    def _remove_order_wrapper(self, entry, dataset_type):
        """
        Removes the 'ORDER' wrapper from the 'TOP' field in the dataset entry.
        """
        top_field = entry.get(f"{dataset_type}.TOP", "")
        if top_field.startswith("(ORDER") and top_field.endswith(")"):
            entry[f"{dataset_type}.TOP"] = top_field[7:-1].strip()
        return entry

    def _process_entry(self, entry, dataset_type):
        """
        Applies cleaning functions to a single dataset entry.
        """
        entry = self._remove_exr_and_top_decoupled(entry, dataset_type)
        entry = self._remove_order_wrapper(entry, dataset_type)
        return entry

    def filter_and_clean_dataset(self, dataset_type="train", start=0, end=None):
        """
        Cleans the dataset file by processing each entry and retaining only 'SRC' and modified 'TOP' attributes.
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
                processed_instance = self._process_entry(instance, dataset_type)
                cleaned_instance = {
                    f"{dataset_type}.SRC": processed_instance.get(f"{dataset_type}.SRC"),
                    f"{dataset_type}.TOP": processed_instance.get(f"{dataset_type}.TOP")
                }
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

    def extract_orders_from_entry(self, entry, dataset_type="train"):
        """
        Extracts and formats pizza and drink orders from the 'TOP' field while preserving the order of elements
        in the original text.
        """
        top_field = entry.get(f"{dataset_type}.TOP")

        def process_elements(top_field):
            n = len(top_field)
            i=0
            while i < n:
                if top_field[i] == '(':
                    # Start a nested group
                    i += 1  # Move past '('
                    # Extract the first word after '('
                    start = i
                    while i < n and top_field[i] != ' ' and top_field[i] != ')':
                        i += 1
                    keyword = top_field[start:i]  # Extract the full word
                if keyword == 'PIZZAORDER' or keyword == 'DRINKORDER':
                    i=start-1
                    break
                i+=1
            def recursive_process(i, IsPizza=[]):
                # IsPizza=[]
                orders = []
                current_order = []
                n = len(top_field)
                buffer = []  # To collect content within the current level
                while i < n:
                    if top_field[i] == '(':
                        # Start a nested group
                        i += 1  # Move past '('
                        # Extract the first word after '('
                        start = i
                        while i < n and top_field[i] != ' ' and top_field[i] != ')':
                            i += 1
                        keyword = top_field[start:i]  # Extract the full word

                        if keyword == 'PIZZAORDER':
                            IsPizza.append(True);
                        if keyword == 'DRINKORDER':
                            IsPizza.append(False);
                        if keyword == 'PIZZAORDER' or keyword == 'DRINKORDER':
                            if buffer:
                                buffer = []
                            if current_order:
                                orders.append(' '.join(current_order).strip())
                                current_order = []
                        # Skip the space after the keyword if there's more content
                        if i < n and top_field[i] == ' ':
                            i += 1

                        # Recursively process the rest of the group
                        nested_order, i,IsPizza = recursive_process(i,IsPizza)
                        if buffer:
                            # buffer.append(',')
                            current_order.append(''.join(buffer).strip())
                            buffer = []
                        current_order.append(' '.join(nested_order).strip())
                    elif top_field[i] == ')':
                        # End of the current group
                        if buffer:  # Append any remaining content
                            current_order.append(''.join(buffer).strip())
                        if current_order:
                            orders.append(' '.join(current_order).strip())
                        return orders, i + 1,IsPizza
                    else:
                        # Collect characters into the buffer
                        buffer.append(top_field[i])
                        i += 1

                # Append any remaining buffer at the top level
                if buffer:
                    current_order.append(''.join(buffer).strip())
                if current_order:
                    orders.append(' '.join(current_order).strip())
                return orders, i,IsPizza

            # Start processing from the top level
            parsed_result, _ ,IsPizza= recursive_process(i,[])
            paired_orders = [[order.strip(), IsPizza[idx]] for idx, order in enumerate(parsed_result)]
            return paired_orders

        return process_elements(top_field)


    def _extract_orders(self, dataset_type="train",start=0, end=None):
        input_file, _ = self._get_dataset_files(dataset_type)
        output_file="../dataset/formatted_train.json"
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
                orders = self.extract_orders_from_entry(instance, dataset_type)
                if orders:
                    batch.append(orders)
                    if len(batch) >= 1000:
                        for item in batch:
                            for order in item:
                                outfile.write(json.dumps(order) + '\n')
                            # outfile.write(json.dumps(item) + '\n')
                        processed_count += len(batch)
                        print(f"Processed {processed_count} entries so far.")
                        batch = []
            for item in batch:
                for order in item:
                    outfile.write(json.dumps(order) + '\n')
                # outfile.write(json.dumps(item) + '\n')
            processed_count += len(batch)
            if batch:
                print(f"Processed {processed_count} entries so far.")
        
    def create_segmented_dataset(self, dataset_type="train", start=0, end=None):
        """
        Creates a segmented dataset that contains the SRC sentence and a corresponding array
        of token labels based on hierarchical PIZZAORDER and DRINKORDER annotations.
        Processes entries in batches for efficiency.
        """
        input_file, output_file = self._get_dataset_files(dataset_type)

        if not os.path.exists(input_file):
            raise FileNotFoundError(f"The input file '{input_file}' does not exist.")
        if start < 0:
            start = 0

        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:

            line_count = 0
            processed_count = 0
            batch = []

            for line in infile:
                line_count += 1
                if line_count < start + 1:
                    continue
                if end is not None and line_count > end:
                    break

                instance = json.loads(line)
                src_sentence = instance.get(f"{dataset_type}.SRC", "")
                top_field = instance.get(f"{dataset_type}.TOP", "")

                if not src_sentence or not top_field:
                    continue

                tokens = src_sentence.split()
                labels = []

                stack = []  # To track all special tokens
                counter = 0  # To track the position within the current order

                i = 0
                while i < len(top_field):
                    char = top_field[i]
                    if char == '(':
                        # Found the start of a new group
                        i += 1
                        group_start = i
                        while i < len(top_field) and top_field[i] != ' ':
                            i += 1
                        entity = top_field[group_start:i]
                        stack.append(entity)
                        if entity in {"PIZZAORDER", "DRINKORDER"}:
                            counter = 0  # Reset counter for new orders
                    elif char == ')':
                        # End of the current entity
                        if stack:
                            stack.pop()
                        i += 1
                    elif char == ' ':
                        # Skip spaces
                        i += 1
                    else:
                        # Process a token
                        group_start = i
                        while i < len(top_field) and top_field[i] != ' ':
                            i += 1
                        word = top_field[group_start:i]

                        if stack:
                            counter += 1
                            if counter == 1:
                                labels.append(f"{stack[0].split('ORDER')[0]}_BEGIN")
                            else:
                                labels.append(f"{stack[0].split('ORDER')[0]}_INTERMEDIATE")
                        else:
                            labels.append("OTHER")

                # Prepare output entry
                segmented_entry = {
                    f"{dataset_type}.SRC": src_sentence,
                    f"{dataset_type}.LABELS": labels
                }
                batch.append(segmented_entry)
                if len(batch) >= 1000:
                    for item in batch:
                        outfile.write(json.dumps(item) + '\n')
                    processed_count += len(batch)
                    print(f"Processed {processed_count} entries so far.")
                    batch = []


            # Process any remaining entries in the batch
            for item in batch:
                outfile.write(json.dumps(item) + '\n')
            processed_count += len(batch)

            print(f"Processed {processed_count} entries for segmented dataset '{output_file}'.")
    def Label_dataset(self, dataset_type="train",start=0, end=None):
        input_file, _ = self._get_dataset_files(dataset_type)
        output_file="../dataset/labels.json"
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
                orders=self.label_entry(instance, dataset_type)
                if orders:
                    batch.append(orders)
                    if len(batch) >= 1000:
                        for item in batch:
                            for order in item:
                                outfile.write(json.dumps(order) + '\n')
                            # outfile.write(json.dumps(item) + '\n')
                        processed_count += len(batch)
                        print(f"Processed {processed_count} entries so far.")
                        batch = []
            for item in batch:
                for order in item:
                    outfile.write(json.dumps(order) + '\n')
            processed_count += len(batch)
            if batch:
                print(f"Processed {processed_count} entries so far.")
    def label_entry(self, entry, dataset_type="train"):
        """
        Labels each word in top-decoupled field as (Topping,Quantity,....)
        """
        top_decoupled = entry.get(f"{dataset_type}.TOP-DECOUPLED")
        # Stack to keep track of hierarchy
        stack = []
        labels = []
        # Regular expression to split by parentheses and words
        tokens = re.findall(r'\(|\)|[^\s()]+', top_decoupled)
        current_labels = []  # This will hold the current stack of labels
        buffer=[]
        for token in tokens:
            if token in ['PIZZAORDER', 'DRINKORDER', 'ORDER']:
                continue
            if token == '(':
                # Starting a new level in the hierarchy, push current labels onto the stack
                stack.append(current_labels.copy())
                buffer=[]
            elif token == ')':
                # Ending the current level, pop the stack and restore previous context
                current_labels = stack.pop()
                if buffer:
                    phrase=' '.join(buffer).strip()
                    combined_label = " ".join(current_labels)
                    labels.append((phrase, combined_label))
                    buffer=[]
            elif token.isupper() and token not in ['PIZZAORDER', 'DRINKORDER', 'ORDER']:
                # If the token is an uppercase word, it is a label, so add it to the current context
                current_labels.append(token)
            else:
                # Otherwise, it's a word (could be number or item), assign the combined labels
                combined_label = " ".join(current_labels)
                labels.append((token, combined_label))  
            # Combine multi-word phrases
        combined_labels = []
        i = 0
        while i < len(labels):
            word, label = labels[i]
            if i + 1 < len(labels) and labels[i + 1][1] == label:
                # Combine consecutive words with the same label
                combined_word = word
                while i + 1 < len(labels) and labels[i + 1][1] == label:
                    combined_word += ' ' + labels[i + 1][0]
                    i += 1
                combined_labels.append((combined_word, label))
            else:
                combined_labels.append((word, label))
            i += 1

        return combined_labels

    def extract_top_by_depth_and_size(self, sample_size):
        """
        Extract the top `sample_size` entries by tree depth and then by tree size from the training dataset.
        """
        input_file, output_file = self._get_dataset_files("train")
        temp_depth_file = f"{output_file}_top_depth.json"
        temp_size_file = f"{output_file}_top_size.json"

        if not os.path.exists(input_file):
            raise FileNotFoundError(f"The input file '{input_file}' does not exist.")

        # Step 1: Find the top `sample_size` entries by tree depth
        depth_heap = []  # Min-heap to track top entries by depth
        excluded_indices = set()
        processed_count = 0
        
        print("Finding top entries by tree depth...")

        with open(input_file, 'r') as infile:
            for index, line in enumerate(infile):
                instance = json.loads(line)
                top_field = instance.get("train.TOP")

                if not top_field:
                    continue

                depth = self._calculate_depth(top_field)
                if len(depth_heap) < sample_size:
                    heapq.heappush(depth_heap, (depth, index, instance))
                elif depth > depth_heap[0][0]:
                    heapq.heapreplace(depth_heap, (depth, index, instance))

                processed_count += 1
                if processed_count % 1000 == 0:
                    print(f"Processed {processed_count} entries for tree depth...")

        depth_top_entries = [heapq.heappop(depth_heap) for _ in range(len(depth_heap))]
        depth_top_entries.sort(reverse=True, key=lambda x: x[0])

        with open(temp_depth_file, 'w') as depth_outfile:
            for _, index, instance in depth_top_entries:
                excluded_indices.add(index)
                depth_outfile.write(json.dumps(instance) + '\n')

        print(f"Saved top {sample_size} entries by tree depth to {temp_depth_file}")

        # Step 2: Find the top `sample_size` entries by tree size excluding depth entries
        size_heap = []  # Min-heap to track top entries by size
        processed_count = 0

        print("Finding top entries by tree size, excluding those already extracted by depth...")

        with open(input_file, 'r') as infile:
            for index, line in enumerate(infile):
                if index in excluded_indices:
                    continue

                instance = json.loads(line)
                top_field = instance.get("train.TOP")

                if not top_field:
                    continue

                size = self._calculate_size(top_field)
                if len(size_heap) < sample_size:
                    heapq.heappush(size_heap, (size, index, instance))
                elif size > size_heap[0][0]:
                    heapq.heapreplace(size_heap, (size, index, instance))

                processed_count += 1
                if processed_count % 1000 == 0:
                    print(f"Processed {processed_count} entries for tree size...")

        size_top_entries = [heapq.heappop(size_heap) for _ in range(len(size_heap))]
        size_top_entries.sort(reverse=True, key=lambda x: x[0])

        with open(temp_size_file, 'w') as size_outfile:
            for _, _, instance in size_top_entries:
                size_outfile.write(json.dumps(instance) + '\n')

        print(f"Saved top {sample_size} entries by tree size to {temp_size_file}")

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





                        