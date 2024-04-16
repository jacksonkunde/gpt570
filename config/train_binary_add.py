# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-binary-add'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'jackson-binary-add'
wandb_run_name = 'nano-gpt'

dataset = 'binary_add'
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 512 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 32

# architecture config
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0

learning_rate = 1e-5
max_iters = 60000
lr_decay_iters = 600000 # make equal to max_iters usually
min_lr = 1e-6 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
device = 'cuda'

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
