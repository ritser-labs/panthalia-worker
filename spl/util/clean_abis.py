import os
import json
import sys

def process_json_files(directory):
    # Loop through every subfolder in the specified directory
    for subdir, _, files in os.walk(directory):
        for file in files:
            # Check if the file is a .json file
            if file.endswith('.json'):
                file_path = os.path.join(subdir, file)
                
                # Open the JSON file and load its contents
                with open(file_path, 'r') as json_file:
                    try:
                        data = json.load(json_file)
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON file: {file_path}")
                        continue

                # Retain only the "abi" key if it exists
                if "abi" in data:
                    new_data = {"abi": data["abi"]}
                else:
                    new_data = {}

                # Overwrite the JSON file with the new data
                with open(file_path, 'w') as json_file:
                    json.dump(new_data, json_file, indent=4)

                print(f"Processed and updated: {file_path}")

if __name__ == "__main__":
    # Check if the user has provided a directory as an argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory>")
        sys.exit(1)

    # Get the directory from the arguments
    directory = sys.argv[1]

    # Check if the provided path is a valid directory
    if not os.path.isdir(directory):
        print("Invalid directory provided. Please check the path and try again.")
        sys.exit(1)

    # Process the JSON files in the specified directory
    process_json_files(directory)
