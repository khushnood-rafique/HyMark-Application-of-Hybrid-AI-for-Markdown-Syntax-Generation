#!/usr/bin/env python
# coding: utf-8

# ## Summary
# This code module is designed to convert a given text into a structured table. It uses Byte-Pair Encoding for preprocessing, then employs the Fairseq library for table generation. The results can be presented in markdown format for better visualization. To use this module effectively, ensure that all necessary data, models, and dependencies are correctly set up.

# ## Dependencies and Constants
# Before diving into the logic, the code begins by importing necessary modules and defining some constants.

# In[7]:


import os
import subprocess
import logging
from fairseq.data.encoders.gpt2_bpe import get_encoder

# Constants for preprocess_text
DEFAULT_WORKERS = 60
SPLIT = "test"
LANG = "text"
DATA_DIR = os.getcwd()+"/text_to_table/data/testdata"
CKPT = os.getcwd()+"/text_to_table/checkpoints/checkpoint_average_best-3.pt"

# Constants for table_generation
DEFAULT_BEAM = 5
DEFAULT_BUFFER_SIZE = 1024
DEFAULT_MAX_TOKENS = 4096
DEFAULT_USER_DIR = "text_to_table/src/"
DEFAULT_TASK = "text_to_table_task"
DEFAULT_TABLE_MAX_COLUMNS = 10


# ## BPE Encoding Utilities
# Byte-Pair Encoding (BPE) is a type of subword tokenization method. This section contains utility functions and a class to handle BPE encoding.
# 
# 
# 

# In[2]:


class MultiprocessingEncoder(object):
    def __init__(self, encoder_json, vocab_bpe):
        self.encoder = get_encoder(encoder_json, vocab_bpe)

    def encode(self, line):
        ids = self.encoder.encode(line)
        return list(map(str, ids))

    def decode(self, tokens):
        return self.encoder.decode(tokens)

    def encode_line(self, line):
        """
        Encode a single line.
        """
        line = line.strip().replace(" <NEWLINE> ", "\n")
        tokens = self.encode(line)
        return " ".join(tokens)

    def decode_line(self, line):
        tokens = map(int, line.strip().split())
        return self.decode(tokens)

def encode_text_with_bpe(text, encoder_json, vocab_bpe):
    """
    Encode a single text string using BPE.

    Parameters:
    - text: The string text to encode.
    - encoder_json: Path to the encoder.json file.
    - vocab_bpe: Path to the vocab.bpe file.

    Returns:
    - Encoded text as a string.
    """
    encoder = MultiprocessingEncoder(encoder_json, vocab_bpe)
    return encoder.encode_line(text)

    


# ## Text Preprocessing Functions
# This section handles the preprocessing of the input text to make it suitable for table generation.

# In[3]:


def preprocess_text(text, data_dir):
    """
    Preprocesses the given text using BPE encoding and fairseq-preprocess.

    Parameters:
    - text: The text to preprocess.
    - data_dir: Directory for input and output data.
    """
    try:
        # Encode the text using BPE encoding
        encoded_text = encode_text_with_bpe(text, os.path.join(data_dir, "encoder.json"), os.path.join(data_dir, "vocab.bpe"))
        
        # Write the encoded text to a file
        output_file_path = os.path.join(data_dir, f"{SPLIT}.bpe.{LANG}")
        with open(output_file_path, "w") as file:
            file.write(encoded_text)
        
        return output_file_path  # Return the path of the preprocessed file

    except Exception as e:
        # Handle any exceptions that may occur during preprocessing
        print("An error occurred during preprocessing:", str(e))
        return str(e)  # Return None to indicate an error   


# ## Table Generation and Formatting
# This section is dedicated to converting the preprocessed text into a structured table using the Fairseq library, post-processing the output, and then formatting the result into markdown for better visualization.
# 
# The function table_generation uses the Fairseq library to generate a table from the preprocessed text. It runs the Fairseq interactive command and then post-processes the output to extract the final table content.
# 
# Once the table is generated, the next step is to make the output readable. The function convert_fairseq_output_to_text takes the raw output from Fairseq, extracts the hypothesis, and then decodes it using the GPT-2 decoding method to provide a clearer representation of the table.
# 
# For better presentation, especially on platforms that support markdown rendering, the function convert_to_markdown is used. It takes the generated table string and converts it into a markdown table format, making it easier to read and understand.
# 
# 

# In[13]:


def table_generation(data_path, ckpt):
    """
    Generate table from the text using Fairseq and return the result.

    Parameters:
    - data_path: Path to the data directory.
    - ckpt: Checkpoint path for the Fairseq model.

    Returns:
    - content: Generated table content as a string.
    """
    os.environ["PYTHONPATH"] = "."

    # fairseq-interactive command
    cmd_fairseq = [
        "fairseq-interactive", os.path.join(data_path, "bins"),
        "--path", ckpt,
        "--beam", str(DEFAULT_BEAM),
        "--remove-bpe",
        "--buffer-size", str(DEFAULT_BUFFER_SIZE),
        "--max-tokens", str(DEFAULT_MAX_TOKENS),
        "--user-dir", DEFAULT_USER_DIR,
        "--task", DEFAULT_TASK,
        "--table-max-columns", str(DEFAULT_TABLE_MAX_COLUMNS),
        "--unconstrained-decoding"
    ]
    
    # Run fairseq-interactive and redirect input/output
    with open(os.path.join(data_path, "test.bpe.text"), "r") as input_file, \
         open(f"{ckpt}.test_vanilla.out", "w") as output_file:
        try:
            subprocess_result = subprocess.run(cmd_fairseq, stdin=input_file, stdout=output_file, check=True, text=True)
            logging.info(subprocess_result.stdout)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error during Fairseq interactive: {e}")
            return str(e)

    script_dir = os.getcwd()+"/text_to_table/scripts/eval"
    # Convert fairseq output to text
    convert_fairseq_output_to_text(script_dir, f"{ckpt}.test_vanilla.out")

    # Read and return the content
    result_file_path = os.path.join(os.getcwd()+"/text_to_table/checkpoints", "checkpoint_average_best-3.pt.test_vanilla.out.text")
    try:
        with open(result_file_path, 'r') as file:
            content = file.read()
    except FileNotFoundError:
        logging.error(f"Result file {result_file_path} not found!")
        return "Result file not found!"

    return content

def convert_fairseq_output_to_text(script_dir, filename):
    """
    Convert Fairseq output to plain text.

    Parameters:
    - script_dir: Directory containing the necessary Python scripts.
    - filename: File to be converted.
    """
    # Run get_hypothesis.py
    try:
        subprocess_result = subprocess.run(["python", os.path.join(script_dir, "get_hypothesis.py"), filename, f"{filename}.hyp"], check=True, text=True)
        logging.info(subprocess_result.stdout)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during hypothesis extraction: {e}")
        return str(e)

    # Run gpt2_decode.py
    try:
        subprocess_result = subprocess.run(["python", os.path.join(script_dir, "gpt2_decode.py"), f"{filename}.hyp", f"{filename}.text"], check=True, text=True)
        logging.info(subprocess_result.stdout)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during GPT-2 decoding: {e}")
        return str(e)


def convert_to_markdown(table_str):
    """Convert the given table string into markdown format."""
    rows = table_str.split("| <NEWLINE> |")
    
    # Extract headers and values
    headers = []
    values = []
    for row in rows:
        parts = row.strip('|').split('|')
        headers.append(parts[0].strip())
        values.append(parts[1].strip())
    
    # Convert to markdown format
    header_str = "| " + " | ".join(headers) + " |"
    separator_str = "| " + " | ".join(["-"*len(header) for header in headers]) + " |"
    value_str = "| " + " | ".join(values) + " |"
    
    return "\n".join([header_str, separator_str, value_str])



# ## Endpoint Function
# Finally, the generate_table_endpoint function combines all the steps to provide an endpoint for generating a table from a given text.

# In[22]:


def generate_table_endpoint(text):
    try:
        # Preprocess the text
        preprocess_path = preprocess_text(text, DATA_DIR)
       
        # Generate the table
        table_content = table_generation(DATA_DIR, CKPT)

        # Convert the table content to markdown
        table_markdown = convert_to_markdown(table_content)

        return table_markdown
    
    except Exception as e:
        # Handle any exceptions that may occur during the entire process
        logging.error("An error occurred during table generation:", exc_info=True)
        return str(e)
    

