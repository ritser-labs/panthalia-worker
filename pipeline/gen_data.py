import torch
from util import safe_serialize, upload_file
from constants import INPUT_DATA_URL, OUTPUT_DATA_URL

def generate_and_upload_training_data(n_samples=100000, input_size=10, output_size=10):
    # Generate random inputs
    x = torch.randn(n_samples, input_size)
    
    # Linear transformation to generate outputs
    y = x * 4 + 3
    
    # Serialize and upload
    serialized_x = safe_serialize(x)
    serialized_y = safe_serialize(y)
    input_data_url = upload_file(serialized_x, INPUT_DATA_URL)
    output_data_url = upload_file(serialized_y, OUTPUT_DATA_URL)

    return input_data_url, output_data_url

# Use the function to generate data
input_data_url, output_data_url = generate_and_upload_training_data()

# Assuming the URLs are now saved or logged for use in the training process
print("Input Data URL:", input_data_url)
print("Output Data URL:", output_data_url)
