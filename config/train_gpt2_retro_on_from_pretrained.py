# Configuration file for training nanoRETRO, an extension of GPT-2 with retrieval capabilities.

out_dir = 'out_gpt2_124M_retro_on_from_pretrained'
wandb_run_name = 'gpt2-124M-retro-on-from-pretrained'
use_retrieval = True
init_from = 'gpt2'
learning_rate = 6e-5
min_lr = 6e-6
warmup_iters = 0
lr_decay_iters = 60000
