"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import numpy as np

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
# init from a model saved in a specific directory
ckpt_path = os.path.join('out-binary-add', 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
# with torch.no_grad():
#     with ctx:
#         for k in range(num_samples):
#             y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
#             print(decode(y[0].tolist()))
#             print('---------------')


# test on the validation data
num_samples = 1
max_new_tokens = 10
data_dir = 'data/binary_add/'
data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
test_data = decode(data).split('\n')[:-1]

num_correct = 0
total = 0
for t in test_data:
    start, answer = t.split('=')
    start += '='
    print(f"Testing {start}")
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=0.1, top_k=top_k)
                gen = decode(y[0].tolist())[len(start_ids):]
                response = gen.split('\n')[0]
                # updat acc
                total += 1
                print(f"Answer: {response}")
                if response == answer:
                    num_correct +=1 


data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
test_data = decode(data).split('\n')[:-1]

train_num_correct = 0
train_total = 0
for t in test_data:
    start, answer = t.split('=')
    start += '='
    print(f"Testing {start}")
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=0.1, top_k=top_k)
                gen = decode(y[0].tolist())[len(start_ids):]
                response = gen.split('\n')[0]
                # updat acc
                train_total += 1
                print(f"Answer: {response}")
                if response == answer:
                    train_num_correct +=1 


## test on the 6 digit numbers
input_file_path = os.path.join(os.path.dirname(__file__), './binary_sums_6.txt')

with open(input_file_path, 'r') as f:
    data = f.read()
test_data = data.split('\n')[:-1]
print(f"length of dataset in characters: {len(data):,}")

six_digit_correct = 0
six_digit_total = 0
for t in test_data:
    start, answer = t.split('=')
    start += '='
    print(f"Testing {start}")
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=0.1, top_k=top_k)
                gen = decode(y[0].tolist())[len(start_ids):]
                response = gen.split('\n')[0]
                # updat acc
                six_digit_total += 1
                print(f"Answer: {response}")
                if response == answer:
                    six_digit_correct +=1 

print(f"Train accuracy: {train_num_correct / train_total}")
print(f"Validation accuracy: {num_correct / total}")
print(f"Six Digit Accuracy: {six_digit_correct / six_digit_total}")

