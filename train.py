"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

from contextlib import nullcontext
from dataclasses import dataclass, replace
from inspect import signature
import math
import os
import requests
import time
from typing import List, Optional
import unicodedata

from datasets import load_dataset, DownloadConfig
import tiktoken
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig

@dataclass
class DatasetConfig:
    dataset: str
    subset: Optional[str] = None
    split: str = "train"
    filter_fn: Optional[callable] = None
    text_field: str = "text"
    doc_id_field: Optional[str] = None
    sample_weight: float = 1.0
    initially_skip: Optional[int] = None # can be used for resuming
    shuffle_buffer_size: int = 1000
    shuffle_seed: int = 1337
    num_shards: Optional[int] = None
    shard_index: Optional[int] = None

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 100
log_interval = 10
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# logging
wandb_log = True
wandb_project = 'nanoRETRO'
wandb_run_name = 'gpt2-124M' # 'run' + str(time.time())
# benchmarks
multiple_choice_benchmarks = ['HellaSwag', 'MMLU', 'Winogrande'] # DeepEval benchmark classes to run
# data
# DOLMino 50B mix, ex-math:
# https://huggingface.co/datasets/allenai/dolmino-mix-1124#mix-compositions
dolmino_mix = [
    # dataset, weight, n_shards, randomize
    ('dclm', 47.2, 247, False),
    ('flan', 16.6, 209, False),
    ('pes2o', 5.85, 26, False),
    ('wiki', 7.11, 2, True), # two big shards need shuffling
    ('stackexchange', 2.45, 16, False),
]
train_datasets = [
    DatasetConfig(
        dataset='allenai/dolmino-mix-1124',
        subset=d[0],
        doc_id_field="id",
        sample_weight=d[1]/d[2],
        shuffle_buffer_size=100000 if d[3] else 1,
        num_shards=d[2],
        shard_index=shard_index,
    )
    for d in dolmino_mix
    for shard_index in range(d[2])
]
download_config = DownloadConfig(
    max_retries=100, # push through HF outages
    num_proc=10, # parallelize downloads
)
# we use benchmarks as val; splitting datasets doesn't add anything
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
tokenizer = 'gpt2'
vocab_size = 50304 # (50257 rounded up for efficiency)
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# retrieval settings
use_retrieval = False  # Whether to enable RETRO mechanism
retrieval_layers = (5, 8, 11) # Layers at which to apply retrieval (0-indexed)
retrieval_chunk_size = 64  # Splits of input tokens for which neighbors will be retrieved
retrieval_embedding_model = 'nomic-ai/nomic-embed-text-v1.5' # Embedding model for neighbor search
retrieval_milvus_host = "localhost" # Milvus host
retrieval_milvus_port = "19530" # Milvus port
retrieval_milvus_collection_name = 'omnikb_8192_gpt2'  # Milvus collection name
retrieval_k_neighbors = 2  # Number of neighbors to retrieve
retrieval_neighbor_size = 2048 # Maximum sequence length for neighbor encoder
retrieval_compressed_size = 16 # Final compressed length for neighbor encodings
retrieval_transformer_layers = 2 # Number of transformer layers to use in neighbor compression
retrieval_neighbor_continuations = False # Whether to extend retrieved neighbors with next records
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
class StreamingDatasetsManager:
    def __init__(self, dataset_configs: List[DatasetConfig]):
        self.tokenizer = tiktoken.get_encoding(tokenizer)
        self.encode_kwargs = {"allowed_special": {"<|endoftext|>"}} # eots are in the dataset
        self.datasets = []
        for dc in dataset_configs:
            self.datasets.append(
                self._init_dataset(dc)
            )
    
    def _init_dataset(self, config: DatasetConfig):
        epoch = 0
        stream = self._get_stream(config, epoch)
        return {
            "config": config,
            "stream": stream,
            "iterator": iter(stream),
            "epoch": epoch,
        }

    def _get_stream(self, config: DatasetConfig, epoch: int):
        stream = load_dataset(
            config.dataset,
            config.subset,
            split=config.split,
            streaming=True,
            download_config=download_config,
        ).shuffle(buffer_size=config.shuffle_buffer_size, seed=config.shuffle_seed)
        if config.num_shards is not None:
            assert config.shard_index is not None
            stream=stream.shard(num_shards=config.num_shards, index=config.shard_index)
        stream.set_epoch(epoch) # in place
        if config.filter_fn:
            stream = stream.filter(config.filter_fn, with_indices=True)
        if epoch == 0 and config.initially_skip:
            stream = stream.skip(config.initially_skip)
        return stream

    def _refresh_stream(self, idx: int):
        config = self.datasets[idx]["config"]
        current_epoch = self.datasets[idx]["epoch"]
        new_epoch = current_epoch + 1
        stream = self._get_stream(config, new_epoch)
        self.datasets[idx]["stream"] = stream
        self.datasets[idx]["iterator"] = iter(stream)
        self.datasets[idx]["epoch"] = new_epoch

    def _sanitize(self, doc_id: str):
        # avoid querying with invalid UTF-8 characters
        if not isinstance(doc_id, str):
            doc_id = str(doc_id)
        try:
            doc_id = doc_id.encode('unicode-escape').decode('ascii')
        except Exception as e:
            print(f"Error sanitizing doc_id: {e}")
            doc_id = "invalid_doc_id"
        return doc_id

    def get_batch(self, batch_size: int, block_size: int, device: str = "cuda"):
        # Initialize tensors for the batch
        x = torch.zeros((batch_size, block_size), dtype=torch.long, device=device)
        y = torch.zeros((batch_size, block_size), dtype=torch.long, device=device)
        doc_ids = []
        
        for b in range(batch_size):
            # Sample a dataset based on weights
            sample_weights = torch.tensor([dataset["config"].sample_weight for dataset in self.datasets], device=device)
            dataset_idx = torch.multinomial(sample_weights, 1).item()
            dataset = self.datasets[dataset_idx]
            
            # Get documents until we fill this sequence
            sequence_tokens = []
            sequence_doc_ids = []
            while len(sequence_tokens) < block_size + 1:  # +1 because we need both x and y
                success = False
                retries = 0
                max_retries = 5
                # Try retrieving the next document with retries upon HTTP errors
                while not success:
                    try:
                        iterator = dataset["iterator"]
                        doc = next(iterator)
                        success = True
                    except StopIteration:
                        self._refresh_stream(dataset_idx)
                    except requests.exceptions.HTTPError as e:
                        retries += 1
                        if retries > max_retries:
                            raise e
                        wait_time = 2 ** retries  # exponential backoff
                        print(f"HTTP Error encountered: {e}, retrying in {wait_time} seconds (retry {retries}/{max_retries})")
                        time.sleep(wait_time)
                
                config = dataset["config"]
                # Tokenize the text using the specified field
                text_field = config.text_field
                new_tokens = self.tokenizer.encode(doc[text_field], **self.encode_kwargs)
                if len(sequence_tokens) == 0:
                    # For first doc, start at a random position
                    start_idx = torch.randint(0, len(new_tokens), (1,)).item()
                    new_tokens = new_tokens[start_idx:]
                new_tokens.append(self.tokenizer.eot_token)
                sequence_tokens.extend(new_tokens)
                # add doc id if specified
                if config.doc_id_field:
                    doc_id = doc[config.doc_id_field]
                    doc_id = self._sanitize(doc_id)
                    sequence_doc_ids.append(doc_id)
            
            # Trim to exact size needed
            sequence_tokens = sequence_tokens[:block_size + 1]
            
            # Populate x and y for this sequence
            x[b] = torch.tensor(sequence_tokens[:-1], dtype=torch.long, device=device)
            y[b] = torch.tensor(sequence_tokens[1:], dtype=torch.long, device=device)
            doc_ids.append(sequence_doc_ids)
        
        batch_id = hash(x)
        
        return x, y, doc_ids, batch_id

datasets = {
    "train": StreamingDatasetsManager(train_datasets),
}

def get_batch(split):
    return datasets[split].get_batch(batch_size, block_size, device)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0

# model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    tokenizer=tokenizer,
    vocab_size=vocab_size,
    dropout=dropout,
    use_retrieval=use_retrieval,
    retrieval_layers=retrieval_layers,
    retrieval_chunk_size=retrieval_chunk_size,
    retrieval_embedding_model=retrieval_embedding_model,
    retrieval_milvus_host=retrieval_milvus_host,
    retrieval_milvus_port=retrieval_milvus_port,
    retrieval_milvus_collection_name=retrieval_milvus_collection_name,
    retrieval_k_neighbors=retrieval_k_neighbors,
    retrieval_neighbor_size=retrieval_neighbor_size,
    retrieval_compressed_size=retrieval_compressed_size,
    retrieval_transformer_layers=retrieval_transformer_layers,
    retrieval_neighbor_continuations=retrieval_neighbor_continuations,
) # start with model_args from command line
gptconf = GPTConfig(**model_args)
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # Update model_args with checkpoint's model_args
    model_args.update(checkpoint_model_args)
    # create the model
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(
        dropout=dropout,
        use_retrieval=use_retrieval,
        retrieval_layers=retrieval_layers,
        retrieval_chunk_size=retrieval_chunk_size,
        retrieval_embedding_model=retrieval_embedding_model,
        retrieval_milvus_host=retrieval_milvus_host,
        retrieval_milvus_port=retrieval_milvus_port,
        retrieval_milvus_collection_name=retrieval_milvus_collection_name,
        retrieval_k_neighbors=retrieval_k_neighbors,
        retrieval_neighbor_size=retrieval_neighbor_size,
        retrieval_compressed_size=retrieval_compressed_size,
        retrieval_transformer_layers=retrieval_transformer_layers,
        retrieval_neighbor_continuations=retrieval_neighbor_continuations,
    )
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, doc_ids, _ = get_batch(split) # no prefetching during evaluation
            with ctx:
                logits, loss = model(X, Y, doc_ids=doc_ids)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# benchmarks
if multiple_choice_benchmarks:
    import deepeval.benchmarks
    from deepeval.models.base_model import DeepEvalBaseLLM

    class DeepEvalMCBenchmarkWrapper(DeepEvalBaseLLM):
        def __init__(self, model):
            self.model = model
            self.tokenizer = tiktoken.get_encoding(tokenizer)

        # Required by DeepEval
        def load_model(self):
            return self.model

        def _setup_model(self):
            model = self.load_model()
            model.eval()
            return model

        def _cleanup_model(self, model):
            model.train()

        def _clean_and_tokenize(self, prompt):
            # Remove confusing confinement instructions
            prompt = prompt.replace("\n\nOutput 'A', 'B', 'C', or 'D'. Full answer not needed.", "")
            
            # Tokenize and clip to block size
            prompt_ids = self.tokenizer.encode(prompt)
            prompt_ids = prompt_ids[-block_size:]
            return prompt_ids

        def _generate_single(self, prompt_ids):
            model = self._setup_model()
            output_ids = model.generate(prompt_ids, max_new_tokens=1, temperature=0.1)
            self._cleanup_model(model)
            return output_ids

        # Required by DeepEval
        def generate(self, prompt):
            prompt_ids = self._clean_and_tokenize(prompt)
            prompt_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            
            with ctx, torch.no_grad():
                output_ids = self._generate_single(prompt_ids)
            new_output_ids = output_ids.tolist()[0][len(prompt_ids.tolist()[0]):]
            output = self.tokenizer.decode(new_output_ids)
            
            return output

        # Required by DeepEval
        async def a_generate(self, prompt):
            return self.generate(prompt)

        def batch_generate(self, prompts):
            prompt_ids = [self._clean_and_tokenize(p) for p in prompts]
            max_len = max(max(len(ids) for ids in prompt_ids), 1) # deepeval checks with empty prompt
            SPACE_TOKEN = 202 # pad with prepended spaces
            prompt_ids = [
                [SPACE_TOKEN] * (max_len - len(ids)) + ids 
                for ids in prompt_ids
            ]
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
            
            with ctx, torch.no_grad():
                output_ids = self._generate_single(prompt_ids)
            new_output_ids = [
                ids[len(input_ids):] for ids, input_ids in zip(output_ids.tolist(), prompt_ids.tolist())
            ]
            outputs = [self.tokenizer.decode(ids) for ids in new_output_ids]

            return outputs

        def get_model_name(self):
            return "nanoGPT"

    def get_benchmark_class(benchmark_name):
        try:
            return getattr(deepeval.benchmarks, benchmark_name)
        except AttributeError:
            raise ImportError(f"Benchmark class '{benchmark_name}' not found in `deepeval.benchmarks`.")

    def run_deep_eval_benchmark(model, benchmark_name):
        benchmark_class = get_benchmark_class(benchmark_name)
        benchmark_model = DeepEvalMCBenchmarkWrapper(model)
        benchmark_obj = benchmark_class()
        
        eval_params = signature(benchmark_obj.evaluate).parameters
        if 'batch_size' in eval_params:
            results = benchmark_obj.evaluate(model=benchmark_model, batch_size=batch_size)
        else:
            results = benchmark_obj.evaluate(model=benchmark_model)

        result_dict = {
            'overall_score': benchmark_obj.overall_score,
        }
        if hasattr(benchmark_obj, 'task_scores'):
            result_dict['task_scores'] = [(row['Task'], float(row['Score'])) 
                                        for _, row in benchmark_obj.task_scores.iterrows()]

        return result_dict

# training loop
X, Y, doc_ids, batch_id = get_batch('train')  # Fetch the very first batch including doc_ids
if model.retrieval_enabled:  # Start prefetching for first batch
    model.retriever_manager.prefetch_neighbors(X, doc_ids, batch_id)
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}")
        benchmark_scores = {}
        for benchmark_name in multiple_choice_benchmarks:
            benchmark_results = run_deep_eval_benchmark(raw_model, benchmark_name)
            print(f"{benchmark_name} results: {benchmark_results}")
            benchmark_scores[benchmark_name] = benchmark_results['overall_score']
        if wandb_log:
            wandb_dict = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            }
            for benchmark_name, score in benchmark_scores.items():
                wandb_dict[f"benchmarks/{benchmark_name}"] = score
            wandb.log(wandb_dict)
        if iter_num > 0:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'config': config,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        # async prefetch next batch while model is doing the forward pass on the GPU
        X_next, Y_next, doc_ids_next, batch_id_next = get_batch('train')
        if model.retrieval_enabled:
            model.retriever_manager.prefetch_neighbors(X_next, doc_ids_next, batch_id_next)
        with ctx:
            logits, loss = model(X, Y, doc_ids=doc_ids, prefetched_neighbors_key=batch_id)
            loss = loss / gradient_accumulation_steps # scale loss for gradient accumulation
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
        X, Y, doc_ids, batch_id = X_next, Y_next, doc_ids_next, batch_id_next
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
