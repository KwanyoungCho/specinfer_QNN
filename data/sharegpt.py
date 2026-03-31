from datasets import load_dataset

dataset = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered")

print(dataset)
print(dataset['train'][0])