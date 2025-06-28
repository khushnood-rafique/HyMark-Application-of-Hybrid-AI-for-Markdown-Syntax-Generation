#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import math
import pandas as pd
import numpy as np
import requests
import nltk
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import argrelextrema
from fastapi import FastAPI, HTTPException, Request
from typing import Optional
import joblib
from helper import preprocess
from nltk.tokenize.texttiling import TextTilingTokenizer
from fastapi.responses import Response
from table_generation import generate_table_endpoint
from transformers import AutoTokenizer, T5ForConditionalGeneration
from pydantic import BaseModel




app = FastAPI()

    
os.environ["TOKENIZERS_PARALLELISM"] = "false"

    
    # 1. Load the model
model = joblib.load('code_model.joblib')

sentence_transformer_model = SentenceTransformer('all-mpnet-base-v2')

    # 2. Load the model and tokenizer for Headings
tokenizer = AutoTokenizer.from_pretrained("czearing/article-title-generator")
heading_model = T5ForConditionalGeneration.from_pretrained("czearing/article-title-generator")


    # 3. Check NLTK resources and download if necessary
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    

# Utility function to check if the line starts with a bullet point
def is_bullet_point(line):
    # Regular expression pattern to match bullet points
    bullet_point_regex = r'^\s*(?:[iIvVxX]{1,4})*(?:[a-zA-Z0-9])*(?:[.\)\-\*])\s?([^.\n]*)'
    return bool(re.match(bullet_point_regex, line))

# Utility function to detect hyperlinks in the text
def detect_links(text):
    # Regular expression pattern to match URLs
    pattern = r"(?:(?i)https?://|www\.)\S+"
    match = re.search(pattern, text)
    return bool(match)

# Utility function to find image links in the text
def find_image_links(text):
    # Regular expression pattern to match image URLs
    url_pattern = re.compile(r"(?:(?i)https?://)?(?:www\.)?\S+(?:/\S+)*\.(?:jpg|jpeg|png|gif|bmp|svg)")
    matches = re.findall(url_pattern, text)
    return bool(matches)

# Reversed sigmoid function to transform values
def rev_sigmoid(x: float) -> float:
    return (1 / (1 + math.exp(0.5*x)))

# Utility function to activate the similarities array
def activate_similarities(similarities: np.array, p_size=10) -> np.array:
    x = np.linspace(-10, 10, p_size)
    y = np.vectorize(rev_sigmoid) 
    activation_weights = np.pad(y(x), (0, similarities.shape[0]-p_size))
    diagonals = [similarities.diagonal(each) for each in range(0, similarities.shape[0])]
    diagonals = [np.pad(each, (0, similarities.shape[0]-len(each))) for each in diagonals]
    diagonals = np.stack(diagonals)
    diagonals = diagonals * activation_weights.reshape(-1, 1)
    activated_similarities = np.sum(diagonals, axis=0)
    return activated_similarities

# Utility function to check if a line is nested (starts with a tab)
def is_Nested(line):
    return line.startswith('\t')

# Function to generate a title for the given text using a pretrained model
def generate_title(text):
    input_text = "summarize: " + text
    input_tensor = tokenizer.encode(input_text, return_tensors="pt")
    generated = heading_model.generate(input_tensor, max_length=20, num_return_sequences=1, top_k=50, top_p=0.95, do_sample=True, early_stopping=False, num_beams=2)
    generated_title = tokenizer.decode(generated[0], skip_special_tokens=True)
    return generated_title

# Utility function to check if the text is a code snippet
def is_code(text):
    # Check for empty or whitespace-only strings
    if text == '' or text == ' ':
        return False
    # Predict if the text is a code snippet or not
    prediction = model.predict(text)
    if(prediction[0] == "Code"):
        return True
    else:
        return False

# Function to generate a main heading for the text
def get_main_heading(text):
    return "# " + generate_title(text)

# Function to generate a sub-heading for the text
def get_sub_heading(text):
    return "## " + generate_title(text)

# Utility function to check if the text has multiple paragraphs
def has_multiple_paragraphs(text):
    paragraphs = text.split('\n\n')
    return len(paragraphs) > 1
    
# Utility function to format bullet points in the text
def format_bullet_points(line):
    # Extract the bullet point marker from the line
    bullet_point_marker = re.search(r'^\s*(?:[iIvVxX]{1,4})*(?:[a-zA-Z0-9])*(?:[.\)\-\*])\s?', line).group()
    # Format nested and non-nested bullet points differently
    if is_Nested(line):
        return line.replace(bullet_point_marker, '\t+ ')
    return line.replace(bullet_point_marker, '1. ')

# Utility function to format image links in the text
def format_image_links(line):
    # Regular expression pattern to match image URLs
    url_pattern = re.compile(r"(?:(?i)https?://)?(?:www\.)?\S+(?:/\S+)*\.(?:jpg|jpeg|png|gif|bmp|svg)")
    # Replace image URLs with markdown image syntax
    return re.sub(url_pattern, r"![Image](\g<0>)", line)

# Utility function to format hyperlinks in the text
def format_links(line):
    return f"[Link]({line})"
    
# Utility function to format code blocks in the text
def format_code_block(line, is_in_code_block):
    if not is_in_code_block:
        # Start of a new code block
        return "```\n" + line, True
    # Inside a code block
    return line, True

# Utility function to end a code block
def format_end_code_block(line):
    return "```\n" + line, False

# Utility function to create paragraphs in the text
def create_paragraphs(text):
    sentences = text.split('. ')
    embeddings = sentence_transformer_model.encode(sentences)
    similarities = cosine_similarity(embeddings)
    activated_similarities = activate_similarities(similarities, p_size=5)
    minima = argrelextrema(activated_similarities, np.less, order=2)
    split_points = [each for each in minima[0]]
    text = ''
    for num, each in enumerate(sentences):
        if num in split_points:
            text += f'\n\n {each}. '
        else:
            text += f'{each}. '
    return text


# Custom string builder class for efficient string concatenation
class StringBuilder:
    def __init__(self):
        self._strings = []

    def append(self, value):
        self._strings.append(value)

    def __str__(self):
        return ''.join(self._strings)

# Function to process individual text tiles
def process_tile(tile):
    sb = StringBuilder()
    try:
        # Generate sub-heading for the tile
        sub_heading = get_sub_heading(tile)
        sb.append(sub_heading + '\n')
        # Process each line in the tile
        lines = tile.split('\n')
        is_in_code_block = False
        formatted_lines = []
        for line in lines:
            # Format the line based on its content
            if is_bullet_point(line):
                formatted_lines.append(format_bullet_points(line))
            elif find_image_links(line):
                formatted_lines.append(format_image_links(line))
            elif detect_links(line):
                formatted_lines.append(format_links(line))
            elif is_code(line) or line.strip().startswith('#'):
                line, is_in_code_block = format_code_block(line, is_in_code_block)
                formatted_lines.append(line)
            else:
                if is_in_code_block:
                    line, is_in_code_block = format_end_code_block(line)
                formatted_lines.append(line)
        # If we're inside a code block at the end of the tile, end it
        if is_in_code_block:
            formatted_lines.append("```")
        processed_tile = '\n'.join(formatted_lines)
        sb.append(processed_tile)
    except Exception as e:
        sb.append(f"An error occurred while processing the tile: {e}")
    return str(sb)

# Main function to process and format the text
def process_text(main_text, sub_text=""):
    sb = StringBuilder()
    # Check for empty main text or if sub_text is not part of main_text
    if not main_text:
        sb.append("Main text is empty.")
        return str(sb)
    if sub_text and sub_text not in main_text:
        sb.append("Sub_text is not part of the main_text.")
        return str(sb)
    main_text = main_text.replace(sub_text, "")
    try:
        if not has_multiple_paragraphs(main_text):
            main_text = create_paragraphs(main_text)
    except Exception as e:
        sb.append(f"Error while creating paragraphs: {e}")
        return str(sb)
    try:
        # Tokenize the main text into tiles
        tt = TextTilingTokenizer()
        tiles = tt.tokenize(main_text)
    except Exception as e:
        sb.append(f"Error during text tiling tokenization: {e}")
        return str(sb)
    try:
        # Generate a main heading for the text
        main_heading = get_main_heading(main_text)
        sb.append(main_heading + '\n\n')
    except Exception as e:
        sb.append(f"Error generating main heading: {e}")
        return str(sb)
    # Process each tile
    for tile in tiles:
        tile_result = process_tile(tile)
        sb.append(tile_result)
        sb.append('\n\n')
    # If there's sub_text, convert it to a table
    if sub_text:
        try:
            table = generate_table_endpoint(sub_text)
            sb.append(table)
        except Exception as e:
            sb.append(f"Error converting sub_text to table: {e}")
    return str(sb)

class TextPayload(BaseModel):
    main_text: str
    sub_text: Optional[str] = ""

@app.post("/process_texts/", response_class=Response)
def process_texts(payload: TextPayload):
    try:
        result = process_text(payload.main_text, payload.sub_text)
        return Response(content=result, media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# In[ ]:




