# nanoRETRO

Extension of [nanoGPT](https://github.com/karpathy/nanoGPT) which incorporates RETRO [(Borgeaud et al., 2022)](https://arxiv.org/abs/2112.04426) for fun and science. This matches the paper's final implementation except for the following differences:

- Using Milvus instead of Faiss for the vector database
- Using all-mpnet-base-v2 instead of bert for chunk and neighbor embeddings
- Retrieving neighbors on-the-fly instead of pre-computing
- Using absolute instead of relative position embeddings in the neighbor encoder, and none in cross attention

## environment setup

Just create a venv and install dependencies...

```sh
python -m venv venv
source venv/bin/activate
pip install datasets deepeval pymilvus sentence-transformers tiktoken torch tqdm transformers wandb
```

...then start + populate Milvus. Contact [@rfdearborn](https://github.com/rfdearborn) for canned knowledge base images!

Dependencies:

- `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
- `deepeval` for benchmarks <3
- `milvus` for knowledge externalization via their fast+robust vector db <3
- `pytorch` <3
- `sentence-transformers` for efficient text embeddings <3
- `tiktoken` for OpenAI's fast BPE code <3
- `tqdm` for progress bars <3
- `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
- `wandb` for optional logging <3

## configuration

The main model is defined in `model.py`. Parameters and their defaults are defined in `train.py`, and are mostly the same as in the original repo but with these notable additions:

- `multiple_choice_benchmarks`: a list of DeepEval benchmark classes to run on each eval interval
- `train_datasets`, `val_datasets`: lists of `DatasetConfig` objects defining training and eval dataset streams
- various RETRO-specific parameters: read the paper for details :)

Parameters can be overriden by files in `config/`.

## training

To begin training on a CPU or single-GPU machine, run:

```sh
python train.py [config/<optional_override_file>.py]
```

For multi-GPU training, run this or similar:

```sh
torchrun --standalone --nproc_per_node=8 train.py [config/<optional_override_file>.py]
```


## inference

Use the script `sample.py` to sample either from pre-trained GPT-2 models released by OpenAI, or from a model you trained yourself. For example, here is a way to sample from the largest available `gpt2-xl` model:

```sh
python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```

If you'd like to sample from a model you trained, use the `--out_dir` to point the code appropriately. You can also prompt the model with some text from a file, e.g. ```python sample.py --start=FILE:prompt.txt```.
