# ToMe: Core Code Reference

The entire algorithm lives in `tome/merge.py`. Everything else (patch modules, vis) wraps it. Below is the annotated source.

---

## `bipartite_soft_matching` — the matching kernel

```python
def bipartite_soft_matching(
    metric: torch.Tensor,       # [B, N, C]  — the similarity feature per token
    r: int,                     # number of tokens to remove this layer
    class_token: bool = False,
    distill_token: bool = False,
    tome_info: dict = None,     # extended config (feature_choice, partition_style, etc.)
) -> Tuple[Callable, Callable]:
    """Returns (merge_fn, unmerge_fn). Call merge_fn(x) to reduce tokens."""
```

### What it does step by step

```python
# 1. Guard: never remove more than 50% of non-special tokens
protected = int(class_token) + int(distill_token)
t = metric.shape[1]
r = min(r, (t - protected) // 2)
if r <= 0:
    return do_nothing, do_nothing   # no-op functions

# 2. Normalize for cosine similarity
with torch.no_grad():
    metric = metric / metric.norm(dim=-1, keepdim=True)

    # 3. Alternating partition: A = even indices, B = odd indices
    a, b = metric[..., ::2, :], metric[..., 1::2, :]

    # 4. Pairwise cosine similarity matrix
    scores = a @ b.transpose(-1, -2)   # [B, Na, Nb]

    # 5. Pin special tokens so they are never sources
    if class_token:
        scores[..., 0, :] = -math.inf
    if distill_token:
        scores[..., :, 0] = -math.inf

    # 6. Greedy: pick top-r most similar pairs
    node_max, node_idx = scores.max(dim=-1)           # best B for each A: [B, Na]
    edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]  # rank A tokens

    unm_idx = edge_idx[..., r:, :]    # A tokens that stay
    src_idx = edge_idx[..., :r, :]    # A tokens that merge into B
    dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)  # their B partners
```

### The returned merge closure

```python
def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
    # x: [B, N, C]
    src, dst = x[..., ::2, :], x[..., 1::2, :]   # split same as metric
    n, t1, c = src.shape

    unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
    src = src.gather(dim=-2, index=src_idx.expand(n, r, c))

    # Accumulate src into the matched dst positions
    dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

    return torch.cat([unm, dst], dim=1)   # [B, N-r, C]
```

`mode` is `"mean"` for the merge step itself, but `merge_wavg` calls it with `"sum"` then divides by accumulated sizes (see below).

### The returned unmerge closure

```python
def unmerge(x: torch.Tensor) -> torch.Tensor:
    # Inverse: scatter merged representation back to original N positions
    unm_len = unm_idx.shape[1]
    unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
    n, _, c = unm.shape

    src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

    out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)
    out[..., 1::2, :] = dst
    out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
    out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)
    return out
```

---

## `merge_wavg` — weighted average merge

```python
def merge_wavg(
    merge: Callable,
    x: torch.Tensor,
    size: torch.Tensor = None,   # [B, N, 1], None → initialized to ones
    tome_info: dict = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if size is None:
        size = torch.ones_like(x[..., 0, None])   # each token = 1 original patch

    x    = merge(x * size, mode="sum")   # sum of weighted features
    size = merge(size,     mode="sum")   # sum of weights
    x    = x / size                      # normalize → weighted average

    return x, size    # both [B, N-r, C] and [B, N-r, 1]
```

Always use `merge_wavg` instead of calling `merge` directly during inference — it correctly handles tokens that have been merged in previous layers.

---

## `merge_source` — source tracking

```python
def merge_source(
    merge: Callable,
    x: torch.Tensor,
    source: torch.Tensor = None,  # [B, N, N_orig]
) -> torch.Tensor:

    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")   # union of covered patches
    return source
```

Call this **before** `merge_wavg` each layer (order matters: source tracks token identity, not values).

---

## `parse_r` — schedule helper

```python
def parse_r(num_layers: int, r) -> List[int]:
    # r: int  →  [r] * num_layers
    # r: (int, float)  →  linear schedule parameterized by inflection ∈ [-1, 1]
    # r: list  →  padded/truncated to num_layers
    ...
    return list_of_ints  # length == num_layers
```

The returned list is consumed via `.pop(0)` in each block's forward, so it must be freshly generated at the start of each model forward pass.

---

## `benchmark` — throughput measurement

```python
def benchmark(
    model, device=0, input_size=(3,224,224), batch_size=64,
    runs=40, throw_out=0.25, use_fp16=False, verbose=False
) -> float:  # images/second
```

Handles CUDA sync correctly. `throw_out` discards the first fraction of runs as GPU warmup. Always call `model.eval()` before benchmarking.
