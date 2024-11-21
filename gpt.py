import torch
import numpy

batch_size = 32  
block_size = 8  

# somreetgjing
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
 
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
 
data = torch.tensor(encode(text), dtype=torch.long)

print("\n")
print(f"Size of the dataset: {len(text)} Characters")
print("\n")
print(f"Vocabulary Size: {vocab_size}")
print("\n")
print("Unique Characters in the dataset - Tokens")
print(''.join(chars))
print("\n")
print("First 100 encoded characterd in the dataset\n")
print(str(data[:100].numpy().tolist()))

with open("milestone1.txt", "w") as f:
    f.write(f"Size of the dataset: {len(text)} Characters")
    f.write(f"Vocabulary Size: {vocab_size}")
    f.write("Unique Characters in the dataset - Tokens")
    f.write(''.join(chars))
    f.write("\nFirst 100 encoded characterd in the dataset\n")
    f.write(str(data[:100].numpy().tolist()))