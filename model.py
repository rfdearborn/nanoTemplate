"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

from collections import OrderedDict
import concurrent.futures
from contextlib import nullcontext
from dataclasses import dataclass
import inspect
import math
import multiprocessing
from threading import Lock

from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.dropout_p = config.dropout

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(C, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout_p if self.training else 0, is_causal=True)
        x = x.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        x = self.dropout(self.c_proj(x))
        return x

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, cross_attention_module=None, neighbor_kv=None):
        x = x + self.attn(self.ln_1(x))
        if cross_attention_module is not None:
            assert neighbor_kv is not None, "neighbor_kv must be provided when using cross attention"
            x = x + cross_attention_module(x, neighbor_kv)
        x = x + self.mlp(self.ln_2(x))
        return x

# Hardcore pytorch avoidance
# (There's prob a better way to do this but this works)
RETRIEVER_CHUNK_ENCODERS = {}
def get_retriever_chunk_encoder(model_name):
    global RETRIEVER_CHUNK_ENCODERS
    if model_name not in RETRIEVER_CHUNK_ENCODERS:
        RETRIEVER_CHUNK_ENCODERS[model_name] = SentenceTransformer(model_name)
    return RETRIEVER_CHUNK_ENCODERS[model_name]

PAD_TOKEN_ID = 220 # tokenizer has no pad token; 220 is " "

@torch._dynamo.disable
class NeighborRetriever(nn.Module):

    def _connect_to_milvus(self, config):
        connections.connect(
            alias="default",
            host=config.retrieval_milvus_host,
            port=config.retrieval_milvus_port
        )
        print(f"Connected to Milvus server at {config.retrieval_milvus_host}:{config.retrieval_milvus_port}")

    def __init__(self, config):
        super().__init__()
        self.k_neighbors = config.retrieval_k_neighbors
        self.neighbor_size = config.retrieval_neighbor_size
        self.use_continuations = config.retrieval_neighbor_continuations
        self.chunk_size = config.retrieval_chunk_size

        # Initialize Milvus collection
        self._connect_to_milvus(config)
        self.collection = Collection(config.retrieval_milvus_collection_name)
        self.collection.load()

        # Load the embedding models
        self.tokenizer = tiktoken.get_encoding(config.tokenizer)
        self.chunk_encoder = get_retriever_chunk_encoder(config.retrieval_embedding_model)
        self.embedding_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        self._neighbor_token_cache = OrderedDict()
        self._neighbor_token_cache_max_size = 10000

    def __del__(self):
        connections.disconnect("default")
        print("Disconnected from Milvus server")

    def forward(self, idx, doc_ids):
        """
        idx: Tensor of shape (B, T)
        doc_ids: List of lists, length B, each containing unique source doc_ids for that sequence
        """
        B, T = idx.size()
        device = idx.device
        n_chunks = T // self.chunk_size
        assert T % self.chunk_size == 0, "Sequence length must be a multiple of chunk size"

        # Reshape idx into chunks
        idx_chunks = idx.view(B, n_chunks, self.chunk_size)

        # Flatten batch and chunk dimensions for parallel processing
        B_n_chunks = B * n_chunks
        idx_chunks_flat = idx_chunks.view(B_n_chunks, self.chunk_size)

        # Decode chunks to text and clip to max seq length
        idx_chunks_flat_decoded = [
            self.tokenizer.decode(idx_chunk.tolist())
            for idx_chunk in idx_chunks_flat
        ]
        # Shape: (B_n_chunks)

        # Encode chunks for retrieval using dedicated stream for parallelization
        with torch.cuda.stream(self.embedding_stream) if self.embedding_stream else nullcontext():
            query_embeddings_flat = self.chunk_encoder.encode(
                idx_chunks_flat_decoded,
                convert_to_tensor=True,
                batch_size=512 # diminishing returns beyond
            ).cpu().numpy()
            # Shape: (B_n_chunks, retrieval_embedding_model_dim)

        if self.embedding_stream:
            self.embedding_stream.synchronize()

        # Retrieve neighbors
        excl_doc_ids = set().union(*doc_ids) # We trade a slight chance of false positives for native filtering
        expr = f"doc_id not in {list(excl_doc_ids)}" if excl_doc_ids else None
        output_fields = ["id"] if self.use_continuations else ["doc_id", "text"]
        _results = self.collection.search(
            data=query_embeddings_flat,
            anns_field="vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=self.k_neighbors,
            expr=expr,
            output_fields=output_fields
        )

        results = []
        if self.use_continuations:
            # Get IDs for direct neighbors and their continuations
            expanded_ids = set()
            for result in _results:
                for hit in result:
                    expanded_ids.add(hit.id)
                    expanded_ids.add(hit.id + 1)
            
            # Batch fetch all 
            _expanded_results = self.collection.query(
                expr=f"id in {list(expanded_ids)}",
                output_fields=["id", "doc_id", "text"]
            )
            results_by_id = {str(c["id"]): c for c in _expanded_results}
            
            # Reconstruct results with continuations
            for result in _results:
                group_results = []
                for hit in result:
                    id = hit.id
                    neighbor = results_by_id.get(str(id))
                    continuation = results_by_id.get(str(id + 1))
                    valid_continuation = continuation and continuation["doc_id"] == neighbor["doc_id"]
                    text = neighbor["text"]
                    if valid_continuation:
                        text += " " + continuation["text"]
                    group_results.append({
                        "text": text,
                        "doc_id": neighbor["doc_id"]
                    })
                results.append(group_results)
        else:
            for result in _results:
                group_results = []
                for hit in result:
                    group_results.append({
                        "text": hit.entity.get("text"),
                        "doc_id": hit.entity.get("doc_id")
                    })
                results.append(group_results)
            
        # Process neighbors
        def _process(result, chunk_text):
            processed_tokens = []
            for i in range(self.k_neighbors):
                if i < len(result):
                    hit = result[i]
                    text = hit["text"]
                    # Trim to avoid unnecessary encoding, assuming 4 chars per token plus 10% buffer
                    # Then encode and truncate to exact token length needed
                    trim_pos = int(4 * self.neighbor_size * 1.1)
                    trimmed_text = text[:trim_pos]
                    # Check cache and update using LRU strategy
                    use_cache = self._neighbor_token_cache_max_size > 0
                    if use_cache and trimmed_text in self._neighbor_token_cache:
                        tokens = self._neighbor_token_cache[trimmed_text]
                        self._neighbor_token_cache.move_to_end(trimmed_text)
                    else:
                        tokens = self.tokenizer.encode(trimmed_text)[:self.neighbor_size]
                        if use_cache:
                            self._neighbor_token_cache[trimmed_text] = tokens
                            if len(self._neighbor_token_cache) > self._neighbor_token_cache_max_size:
                                self._neighbor_token_cache.popitem(last=False)
                else:
                    # Pad with empty result if we don't have enough neighbors
                    tokens = []
                # Pad tokens to neighbor_size if necessary
                padding_len = self.neighbor_size - len(tokens)
                if padding_len > 0:
                    tokens.extend([PAD_TOKEN_ID] * padding_len)
                processed_tokens.append(tokens)
            return processed_tokens
        
        # Note: I tried ThreadPoolExecutor but this is faster, esp with prefetching
        neighbor_tokens = [
            _process(result, idx_chunks_flat_decoded[i])
            for i, result in enumerate(results)
        ]
        # Shape: (B_n_chunks, k_neighbors, neighbor_size)
        
        return torch.tensor(neighbor_tokens, device=device).view(
            B, n_chunks, self.k_neighbors, self.neighbor_size
        )

class AsyncRetrieverManager:
    def __init__(self, config):
        self.lock = Lock()
        self.retriever = NeighborRetriever(config)
        self.prefetch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() // 2)
        self.prefetch_futures = {}

    def prefetch_neighbors(self, idx, doc_ids, key):
        with self.lock:
            self.prefetch_futures[key] = self.prefetch_executor.submit(self.retriever, idx, doc_ids)

    @torch._dynamo.disable
    def get_neighbors(self, idx, doc_ids, prefetch_key=None):
        if prefetch_key is not None:
            with self.lock:
                future = self.prefetch_futures.pop(prefetch_key, None)
                if future is not None:
                    return future.result()
        return self.retriever(idx, doc_ids)

    def __del__(self):
        self.prefetch_executor.shutdown()

class NeighborEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.chunk_size = config.retrieval_chunk_size
        self.n_embd = config.n_embd
        self.tokenizer = tiktoken.get_encoding(config.tokenizer)

        self.wte = nn.Embedding(
            config.vocab_size,
            config.n_embd
        )

        self.wpe = nn.Embedding(
            config.retrieval_neighbor_size,
            config.n_embd
        )

        self.wne = nn.Embedding(
            config.retrieval_k_neighbors,
            config.n_embd
        )

    def forward(self, neighbor_tokens):
        """
        neighbor_tokens: Tensor of shape (B, n_chunks, k_neighbors, neighbor_size)
        """
        B, n_chunks, k_neighbors, neighbor_size = neighbor_tokens.size()
        bnk = B * n_chunks * k_neighbors
        device = neighbor_tokens.device

        # Flatten neighbor_tokens for efficient processing
        neighbor_tokens = neighbor_tokens.view(bnk, neighbor_size)
        
        # Embed tokens
        te = self.wte(neighbor_tokens)
        
        pos = torch.arange(neighbor_size, device=device) # abs pos embeddings for simplicity
        pe = self.wpe(pos)
        
        npos = torch.arange(k_neighbors, device=device)
        ne = self.wne(npos)
        ne = ne.unsqueeze(1).expand(-1, neighbor_size, -1)
        ne = ne.repeat(B * n_chunks, 1, 1)
        
        x = te + pe + ne

        x = x.view(B, n_chunks, k_neighbors, neighbor_size, self.n_embd)

        return x

class NeighborKVProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_ne = LayerNorm(config.n_embd, bias=config.bias)
        self.cross_attn_kv = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)

    def forward(self, neighbor_embeddings):
        neighbor_embeddings_norm = self.ln_ne(neighbor_embeddings)
        neighbor_kv = self.cross_attn_kv(neighbor_embeddings_norm)
        return neighbor_kv

class CrossAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.chunk_size = config.retrieval_chunk_size

        self.ln_hs = LayerNorm(config.n_embd, bias=config.bias)

        self.cross_attn_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.cross_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.dropout_p = config.dropout

    def forward(self, hidden_states, neighbor_kv):
        """
        hidden_states: Tensor of shape (B, T, C)
        neighbor_kv: Tensor of shape (B, n_chunks, k_neighbors, neighbor_size, 2 * C)
        """
        B, n_chunks, k_neighbors, neighbor_size, two_C = neighbor_kv.size()
        C = two_C // 2
        _, T, _ = hidden_states.size()
        device = hidden_states.device

        # Apply LayerNorm
        hidden_states = self.ln_hs(hidden_states)  # (B, T, C)

        # Shift hidden_states to maintain causality (as in paper)
        shift_amount = self.chunk_size - 1
        hidden_states_shifted = hidden_states[:, shift_amount:, :]  # (B, T-shift_amount, C)
        hidden_states_shifted = F.pad(hidden_states_shifted, (0, 0, 0, shift_amount))  # (B, T, C)

        # Reshape shifted hidden states into chunks
        hidden_states_shifted_chunks = hidden_states_shifted.view(B, n_chunks, self.chunk_size, C)  # (B, n_chunks, chunk_size, C)

        # Compute query projection
        q = self.cross_attn_q(hidden_states_shifted_chunks)  # (B, n_chunks, chunk_size, C)

        # Unpack precomputed neighbor_kv
        k, v = neighbor_kv.split(C, dim=4)  # each: (B, n_chunks, k_neighbors, neighbor_size, C)

        # Reshape for multi-head attention
        head_dim = C // self.n_head
        q = q.view(B * n_chunks, self.chunk_size, self.n_head, head_dim).transpose(1, 2)  # (B*n_chunks, n_head, chunk_size, head_dim)
        k = k.view(B * n_chunks, k_neighbors * neighbor_size, self.n_head, head_dim).transpose(1, 2)  # (B*n_chunks, n_head, k_neighbors*neighbor_size, head_dim)
        v = v.view(B * n_chunks, k_neighbors * neighbor_size, self.n_head, head_dim).transpose(1, 2)  # (B*n_chunks, n_head, k_neighbors*neighbor_size, head_dim)
        
        # Compute scaled dot-product attention
        x = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=False,
            dropout_p=self.dropout_p if self.training else 0,
        )  # (B*n_chunks, n_head, chunk_size, head_dim)
        
        # Merge heads and apply output projection
        x = x.transpose(1, 2).contiguous().view(B * n_chunks, self.chunk_size, C)  # (B * n_chunks, chunk_size, C)
        x = self.cross_proj(x)  # (B * n_chunks, chunk_size, C)
        
        # De-chunk and un-shift
        x = x.view(B, T, C)  # (B, T, C)
        x = torch.cat([
            torch.zeros(B, shift_amount, C, device=device),  # identity for dropped tokens, as in paper
            x[:, :-shift_amount, :]
        ], dim=1)  # (B, T, C)
        
        # Apply dropout
        x = self.dropout(x)  # (B, T, C)

        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    tokenizer: str = "gpt2"
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    use_retrieval: bool = False
    retrieval_layers: list = (5, 8, 11)
    retrieval_chunk_size: int = 64
    retrieval_embedding_model: str = 'sentence-transformers/all-mpnet-base-v2'
    retrieval_milvus_host: str = "localhost"
    retrieval_milvus_port: str = "19530"
    retrieval_milvus_collection_name: str = "omnikb_64_gpt2"
    retrieval_k_neighbors: int = 2
    retrieval_neighbor_size: int = 128
    retrieval_neighbor_continuations: bool = True

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.tokenizer = tiktoken.get_encoding(config.tokenizer)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # instantiate retrieval modules
        self.retrieval_enabled = config.use_retrieval
        if self.retrieval_enabled:
            self.retriever_manager = AsyncRetrieverManager(config)
            self.neighbor_encoder = NeighborEncoder(config)
            self.neighbor_kv_projection = NeighborKVProjection(config)
            self.cross_attention_modules = nn.ModuleList([
                CrossAttention(config) for _ in config.retrieval_layers
            ])
            
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        self.print_parameter_summary()

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def print_parameter_summary(self):
        """Prints a summary of parameters in the model."""
        print("\nParameter Summary:")
        total_params = 0
        for name, param in self.named_parameters():
            param_count = param.numel()
            total_params += param_count
            print(f"{name:70} {param.shape} \t params: {param_count}")
        print(f"\nTotal Parameters: {total_params}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, doc_ids=None, prefetched_neighbors_key=None, debug=False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # process retrieval once for all retrieval modules
        if self.retrieval_enabled:
            assert doc_ids is not None, "doc_ids must be provided when using retrieval."
            retrieval_layers = set(self.config.retrieval_layers)
            first_retrieval_layer = min(retrieval_layers)
            neighbor_tokens = self.retriever_manager.get_neighbors(idx, doc_ids, prefetched_neighbors_key)

        # transformer blocks with interleaved retrieval
        retrieval_layer_index = 0
        neighbor_kv = None
        for layer_index, block in enumerate(self.transformer['h']):
            cross_attention_module = None
            if self.retrieval_enabled:
                # encode neighbors at first retrieval layer
                if layer_index == first_retrieval_layer:
                    neighbor_embeddings = self.neighbor_encoder(neighbor_tokens)
                    neighbor_kv = self.neighbor_kv_projection(neighbor_embeddings)
                # apply cross attention at retrieval layers
                if layer_index in retrieval_layers:
                    cross_attention_module = self.cross_attention_modules[retrieval_layer_index]
                    retrieval_layer_index += 1
            x = block(x, cross_attention_module, neighbor_kv)

        # layer norm
        x = self.transformer.ln_f(x)

        # output
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        if debug:
            # print each input, neighbors, and output
            print("\n===== Debug Output =====")

            # Ensure idx is 2D for debug output
            debug_idx = idx.unsqueeze(0) if idx.dim() == 1 else idx
            
            # Print input text
            for b in range(len(debug_idx)):
                print(f"\n==== Batch {b} ====")
                print(f"\n=== Input text ===\n{self.tokenizer.decode(debug_idx[b].tolist())}")
                
                # Print retrieved neighbors if retrieval is enabled
                if self.retrieval_enabled:
                    print("\n=== Retrieved neighbors ===")
                    n_chunks = len(debug_idx[b]) // self.config.retrieval_chunk_size
                    for chunk in range(n_chunks):
                        print(f"\n== Chunk {chunk} ==")
                        for k in range(self.config.retrieval_k_neighbors):
                            neighbor_tokens_for_chunk = neighbor_tokens[b][chunk][k]
                            # Filter out padding tokens
                            valid_tokens = [t for t in neighbor_tokens_for_chunk.tolist() if t != PAD_TOKEN_ID]
                            neighbor_text = self.tokenizer.decode(valid_tokens)
                            print(f"  Neighbor {k}: {neighbor_text}")
                
                # Print output logits/predictions
                if targets is None: # Generation mode
                    top_k = 5
                    last_logits = logits[b, -1]
                    top_probs, top_indices = torch.topk(F.softmax(last_logits, dim=-1), top_k)
                    print("\n=== Top predictions for next token ===")
                    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
                        token_text = self.tokenizer.decode([idx])
                        print(f"  {token_text!r}: {prob:.3f}")
                else: # Training mode
                    print("\n=== Training loss ===\n", loss.item())
            
            print("\n===== End Debug Output =====")

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout and retrieval params can be overridden
        allowed_override_args = {
            'dropout',
            'use_retrieval',
            'retrieval_layers',
            'retrieval_chunk_size',
            'retrieval_embedding_model',
            'retrieval_milvus_host',
            'retrieval_milvus_port',
            'retrieval_milvus_collection_name',
            'retrieval_k_neighbors',
            'retrieval_neighbor_size',
            'retrieval_neighbor_continuations',
        }
        assert all(k in allowed_override_args for k in override_args), f"Only {allowed_override_args} can be overridden"
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # apply any overrides
        for k, v in override_args.items():
            config_args[k] = v
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k 
            for k in sd_keys
            if not k.endswith('.attn.bias') # discard this mask / buffer, not a param
            and not any(
                k.startswith(x)
                for x in ['retriever_manager', 'neighbor_encoder', 'cross_attention_modules'] # ignore retrieval layers
            )
        ]

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def pad_to_chunk_size(self, idx):
        """Pad sequence to be multiple of chunk_size if retrieval is enabled"""
        if not self.retrieval_enabled:
            return idx
        
        chunk_size = self.config.retrieval_chunk_size
        if idx.size(1) % chunk_size == 0:
            return idx
        
        pad_length = chunk_size - (idx.size(1) % chunk_size)
        padding = torch.full((idx.size(0), pad_length), PAD_TOKEN_ID, dtype=idx.dtype, device=idx.device)
        return torch.cat([padding, idx], dim=1) # left pad to preserve the original sequence

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # Pad the input sequence to be a multiple of chunk_size if retrieval is enabled
            idx_cond = self.pad_to_chunk_size(idx_cond)
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond, doc_ids=[]) # input idx not associated with any docs
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # clamp in case vocab_size is padded up
            idx_next = idx_next.clamp(max=self.tokenizer.max_token_value)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
