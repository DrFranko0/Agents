import json
from transformers import AutoTokenizer

with open("training_data.json","r") as file:
    data=json.load(file)

texts=[ { "input":item["prompt"] , "output":item["response"] } for item in data ]

combined=combined_texts = [f"{item['input']} {tokenizer.eos_token} {item['output']}" for item in texts]

tokenizer = AutoTokenizer.from_pretrained("")
tokenized_data = tokenizer(
    combined,
    truncation=True,
    padding=True,
    return_tensors="pt"
)

tokenizer.save_pretrained("./fine_tuned_model")