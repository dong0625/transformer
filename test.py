import numpy as np
import torch
import evaluate
from transformers import AutoTokenizer

bleu = evaluate.load("bleu")
print(AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased').all_special_ids)
""" 
original_texts = ["hello world this is a test", "qwer asdf zxcv"]
encoded_list = tokenizer(original_texts)["input_ids"]
decoded_list = list(map(lambda x: tokenizer.decode(x, skip_special_tokens=True), encoded_list))

print(bleu.compute(predictions=decoded_list, references=original_texts)) """