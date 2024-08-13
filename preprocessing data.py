from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import pipeline

def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

gandhi_text = load_dataset('gandhi.txt')
einstein_text = load_dataset('einstein.txt')

combined_text = gandhi_text + "\n\n" + einstein_text

with open('combined_dataset.txt', 'w', encoding='utf-8') as f:
    f.write(combined_text)