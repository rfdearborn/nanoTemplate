# nanoTemplate

Reduced [nanoGPT](https://github.com/karpathy/nanoGPT) for fast experimentation with mechanisms and architectures.

## environment setup

Just create a venv and install dependencies...

```sh
python -m venv venv
source venv/bin/activate
pip install datasets numpy tiktoken torch tqdm transformers wandb 
```

...then download and tokenize [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/):

```
python data/openwebtext/prepare.py
```

You're ready to tinker with GPT2-class models!

Dependencies:

- `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
- `numpy` <3
- `pytorch` <3
- `tiktoken` for OpenAI's fast BPE code <3
- `tqdm` for progress bars <3
- `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
- `wandb` for optional logging <3

## configuration

The main model is defined in `model.py`. Parameters and their defaults are defined in `train.py`, and are mostly the same as in the original repo. They can be overriden by files in `config/`.

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
