import base64

def generate_tiktoken_file(output_path):
    # Create a list of characters to include in the tokenization
    characters = [chr(i) for i in range(32, 127)]  # printable ASCII characters
    additional_chars = ['\n', '\t', ' ']  # add newline, tab, and space explicitly
    characters += additional_chars

    # Generate the .tiktoken file content
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, char in enumerate(characters):
            # Encode the character in Base64
            base64_token = base64.b64encode(char.encode('utf-8')).decode('utf-8')
            # Write the Base64 encoded token and its ID to the file
            f.write(f"{base64_token} {idx}\n")

# Example usage
generate_tiktoken_file('char.tiktoken')
