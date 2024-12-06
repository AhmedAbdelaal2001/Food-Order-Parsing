import os
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
        Extracts and formats pizza and drink orders from the 'TOP_DECOUPLED' field while preserving the order of elements
        in the original text.
        """
        top_field = entry.get(f"{dataset_type}.TOP-DECOUPLED")

        def process_elements(top_field):
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

                        if keyword=='NOT':
                            # buffer.append(keyword)
                            current_order.append("no")
                        if keyword == 'PIZZAORDER':
                            IsPizza.append(True);
                        if keyword == 'DRINKORDER':
                            IsPizza.append(False);
                        if keyword == 'PIZZAORDER' or keyword == 'DRINKORDER':
                            if buffer:
                                buffer.append(',')
                                current_order.append(''.join(buffer).strip())
                                buffer = []

                            if current_order:
                                orders.append(' '.join(current_order).strip())
                                current_order = []
                        # Skip the space after the keyword if there's more content
                        if i < n and top_field[i] == ' ':
                            i += 1

                        # Recursively process the rest of the group
                        nested_order, i = recursive_process(i)
                        current_order.append(' '.join(nested_order).strip())
                    elif top_field[i] == ')':
                        # End of the current group
                        if buffer:  # Append any remaining content
                            current_order.append(''.join(buffer).strip())
                        if current_order:
                            orders.append(' '.join(current_order).strip())
                        return orders, i + 1
                    else:
                        # Collect characters into the buffer
                        buffer.append(top_field[i])
                        i += 1

                # Append any remaining buffer at the top level
                if buffer:
                    current_order.append(''.join(buffer).strip())
                if current_order:
                    orders.append(' '.join(current_order).strip())
                    orders_list = []
                i = 0
                # print(len(IsPizza))
                for order in orders[0].split(','):
                    orders_list.append([order.strip(), IsPizza[i]])  # Create a pair of order and IsPizza boolean
                    i += 1
                return orders_list, i

            # Start processing from the top level
            parsed_result, _ = recursive_process(0)
            return parsed_result

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
                outfile.write(json.dumps(item) + '\n')
            processed_count += len(batch)
            if batch:
                print(f"Processed {processed_count} entries so far.")
                