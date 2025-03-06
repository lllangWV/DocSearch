import json
import os
from glob import glob

import matplotlib.pyplot as plt
import tiktoken



def plot_character_distribution(pdfs_dirs, bins=20, filename=None):
    
    character_lengths=[]
    for pdf_dir in pdfs_dirs:
        with open(os.path.join(pdf_dir,f'combined_pdf_info.json'), 'r') as f:
            data = json.load(f)
            for key, value in data.items():
                if key.startswith('page_'):
                    character_lengths.append(len(value))


    plt.hist(character_lengths, bins=bins)
    plt.xlabel('Character Length')
    plt.ylabel('Count')
    plt.title('Distribution of Character Lengths in Minutes')

    # Calculate and display the total number of characters
    total_characters = sum(character_lengths)
    max_length = max(character_lengths)
    min_length = min(character_lengths)

    plt.text(0.95, 0.95, f'Total Characters: {total_characters}\nMax Length: {max_length}\nMin Length: {min_length}',
             verticalalignment='top', horizontalalignment='right',
             transform=plt.gca().transAxes, color='blue', fontsize=10)

    if filename:
        plt.savefig(f'data/minutes/plots/character_lengths.png')
    else:
        plt.show()
    

def count_tokens(text, model="gpt-4o"):
    enc = tiktoken.encoding_for_model(model)
    num_tokens = len(enc.encode(text))
    return num_tokens


if __name__ == "__main__":
    # Define the directory path where the text files are located
    raw_minutes_dir = os.path.join('data/minutes/raw')
    interim_dir=os.path.join('data/minutes/interim')
    pdfs_dirs=glob(os.path.join(interim_dir,'*'))
    plot_character_distribution(pdfs_dirs=pdfs_dirs,bins=50, filename=f'data/minutes/plots/character_lengths.png')



    # print("Prompt:")
    # with open(prompt_file, 'r', encoding='utf-8') as f:
    #     text=f.read()
    #     num_tokens=count_tokens(text)
    #     print(f"Number of tokens: {num_tokens}")


    # print("Response:")
    # with open(response_file, 'r', encoding='utf-8') as f:
    #     text=f.read()
    #     num_tokens=count_tokens(text)
    #     print(f"Number of tokens: {num_tokens}")