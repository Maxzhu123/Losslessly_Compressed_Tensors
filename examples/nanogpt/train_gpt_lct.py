"""
train_gpt_lct.py: independent lossless LCT storage options for nanoGPT.

This file descends from the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt).
It was prepared as a simplified version of the speedrun for use in neural net optimization research.
"""
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import time
from datetime import datetime

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from examples.nanogpt.dataloader import load_data_shard
from examples.nanogpt.dist_configs import weight_dist, momentum_dist, act_dist, act_relu_dist
from LCT.components.layers import Linear, Embedding, RMSNormFunction, RMSLinear, Relu2Linear, FlashAttention, dense_weight, compress_weight, named_trainable_tensors, rms_norm_qkv
from LCT.components.sparse_utils import SparseAdamW, SparseMuon
from LCT.tensor_buffer import TensorBuffer, _free_regions_snapshot

DATA_ROOT = Path(__file__).resolve().parent
LOG_ROOT = DATA_ROOT / "logs"
COMPRESS_WEIGHTS = True  # All BF16 matrices, including embeing and LM head.
COMPRESS_ACTIVATIONS = True  # BF16 saves, including native FlashAttention Q/K/V/output.
COMPRESS_OPTIMISER = True  # Muon momentum only; AdamW moments stay dense.
BUFFER = True  # Shared fallback arena; never reset while weights/state are live.
COMPILE = True  # Compile dense model regions; LCT operations run eagerly.
CHECKPOINT_HEAD = True
CHUNK_TOKENS = 4096  # Tokens per checkpointed output projection and loss.
BUFFER_SIZE_MIB = 64
MIN_COMPRESS_ELEMENTS = 65536  # Avoid padding small activations to a full codec block.
TRAIN_STEPS = 3450
SAVE_EVERY = 300
VOCAB_SIZE = 50304
NUM_LAYERS = 12
MODEL_DIM = 768
SEED = 0
# Matrices use BF16 in every mode; biases and RMSNorm gains use FP32.
# Compile dense regions around eager LCT operations and native FlashAttention.
# Keep the dense loss compiled separately, including during validation.
SEQ_LEN = 1024
TRAIN_BATCH_TOKENS = 8 * 64 * 1024  # Tokens per optimizer step.
VAL_TOKENS = 20 * 524288
TRAIN_MICROBATCH_SEQUENCES = 64  # Tune without changing tokens per optimizer step.
VAL_MICROBATCH_SEQUENCES = 4


def data_generator(pattern, batch_size, seq_len=SEQ_LEN, device=None):
    """Yield token batches on one GPU, cycling through training shards."""
    if batch_size % seq_len:
        raise ValueError("Token batch must divide evenly into sequences")
    files = sorted(DATA_ROOT.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No data shards matched {DATA_ROOT / pattern}")
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    shard_index, position = 0, 0
    tokens = load_data_shard(files[shard_index])
    while True:
        if tokens.numel() < batch_size + 1:
            raise ValueError(f"Shard is too small for the global batch: {files[shard_index]}")
        if position + batch_size + 1 > tokens.numel():
            shard_index = (shard_index + 1) % len(files)
            del tokens
            tokens = load_data_shard(files[shard_index])
            position = 0
            continue
        window = tokens[position:position + batch_size + 1]
        inputs = window[:-1].to(device=device, dtype=torch.int32, non_blocking=True)
        targets = window[1:].to(device=device, dtype=torch.int64, non_blocking=True)
        position += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


########################################
#             Architecture             #
########################################

class SoftcapCrossEntropy(torch.autograd.Function):
    """Summed loss for valid vocabulary targets; save logits and row statistics."""

    @staticmethod
    def head_loss(x, targets, weight, gains, bias, buffer, compressed, min_elements):
        W_dense = dense_weight(weight)

        def project_and_loss(chunk, chunk_targets):
            # Checkpointed chunks are consumed immediately; keep their input dense.
            # Keep the original weight as the gradient target; reuse its dense cache.
            logits = RMSLinear.apply(
                chunk, weight, gains, bias, buffer,
                compressed and torch.is_grad_enabled() and not CHECKPOINT_HEAD,
                act_dist, min_elements, W_dense,
            )
            return SoftcapCrossEntropy.apply(logits, chunk_targets)

        if not CHECKPOINT_HEAD:
            return project_and_loss(x, targets)
        x = x.reshape(-1, x.shape[-1])
        targets = targets.reshape(-1)
        loss = x.new_zeros((), dtype=torch.float32)
        for start in range(0, targets.numel(), CHUNK_TOKENS):
            chunk = x[start:start + CHUNK_TOKENS]
            chunk_targets = targets[start:start + CHUNK_TOKENS]
            if torch.is_grad_enabled():
                # Non-reentrant checkpoint hooks conflict with the compiled loss backward.
                loss = loss + checkpoint(project_and_loss, chunk, chunk_targets,
                                         use_reentrant=True, preserve_rng_state=False)
            else:
                loss = loss + project_and_loss(chunk, chunk_targets)
        return loss

    @staticmethod
    @torch.compile
    def forward(ctx, logits: Tensor, targets: Tensor):
        x = logits.float().reshape(targets.numel(), -1)
        capped = 15 * x * (x.square() + 15**2).rsqrt()
        logsumexp = capped.logsumexp(dim=-1, keepdim=True)
        selected = capped.gather(1, targets.reshape(-1, 1))
        ctx.save_for_backward(logits, targets, logsumexp)
        return (logsumexp - selected).sum()

    @staticmethod
    @torch.compile
    def backward(ctx, grad_output):
        logits, targets, logsumexp = ctx.saved_tensors
        x = logits.float().reshape(targets.numel(), -1)
        inv = (x.square() + 15**2).rsqrt()
        probabilities = (15 * x * inv - logsumexp).exp()
        # Compilation fuses the comparison into backward without allocating a mask.
        is_target = torch.arange(x.shape[-1], device=x.device) == targets.reshape(-1, 1)
        # d[15*x/sqrt(x*x + 225)]/dx = 3375/(x*x + 225)**1.5.
        grad_logits = (probabilities - is_target.to(x.dtype)) * inv.pow(3) * 15**3 * grad_output
        return grad_logits.reshape(logits.shape).to(logits.dtype), None


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # half-truncate RoPE (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim//4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim=128, compress_activations=False,
                 min_compress_elements=MIN_COMPRESS_ELEMENTS):
        super().__init__()
        self.compress_activations = compress_activations
        self.min_compress_elements = min_compress_elements
        self.tensor_buffer = None
        if dim % head_dim:
            raise ValueError("model_dim must be divisible by head_dim")
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor, norm_weight: Tensor):
        B, T = x.size(0), x.size(1)
        q, k, v = rms_norm_qkv(
            x, norm_weight, self.q, self.k, self.v, buffer=self.tensor_buffer,
            compressed=self.compress_activations and torch.is_grad_enabled(),
            distribution=act_dist, min_elements=self.min_compress_elements,
        )
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)
        norm_args = (None, self.tensor_buffer,
                     self.compress_activations and torch.is_grad_enabled(),
                     act_dist, self.min_compress_elements)
        q = RMSNormFunction.apply(q, *norm_args)
        k = RMSNormFunction.apply(k, *norm_args)
        q, k = self.rotary(q), self.rotary(k)
        # Explicit Q/K/V/output cache, shared with projection backward.
        return FlashAttention.apply(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            self.proj.weight, self.proj.bias,
            self.tensor_buffer, self.compress_activations and torch.is_grad_enabled(),
            act_dist, self.min_compress_elements,
        )

class MLP(nn.Module):
    def __init__(self, dim: int, compress_activations=False, min_compress_elements=MIN_COMPRESS_ELEMENTS):
        super().__init__()
        self.compress_activations = compress_activations
        self.min_compress_elements = min_compress_elements
        self.tensor_buffer = None
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor, norm_weight: Tensor):
        z = RMSLinear.apply(
            x, self.fc.weight, norm_weight, self.fc.bias, self.tensor_buffer,
            self.compress_activations and torch.is_grad_enabled(),
            act_dist, self.min_compress_elements,
        )
        return Relu2Linear.apply(
            z, self.proj.weight, self.proj.bias, self.tensor_buffer,
            self.compress_activations and torch.is_grad_enabled(),
            act_relu_dist, self.min_compress_elements,
        )

class Block(nn.Module):
    def __init__(self, dim: int, compress_activations=False, min_compress_elements=MIN_COMPRESS_ELEMENTS):
        super().__init__()
        self.attn = CausalSelfAttention(dim, compress_activations=compress_activations,
                                        min_compress_elements=min_compress_elements)
        self.mlp = MLP(dim, compress_activations, min_compress_elements)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(x, self.norm1.gains)
        x = x + self.mlp(x, self.norm2.gains)
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int,
                 compress_activations=False, min_compress_elements=MIN_COMPRESS_ELEMENTS):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim, dtype=torch.bfloat16)
        self.blocks = nn.ModuleList([Block(model_dim, compress_activations, min_compress_elements)
                                     for _ in range(num_layers)])
        self.compress_activations = compress_activations
        self.min_compress_elements = min_compress_elements
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
        self.tensor_buffer = None

    def set_tensor_buffer(self, buffer):
        self.tensor_buffer = buffer
        for block in self.blocks:
            block.mlp.tensor_buffer = buffer
            block.attn.tensor_buffer = buffer

    def compress_weights(self):
        for module in self.modules():
            if isinstance(module, (Linear, nn.Embedding)):
                compress_weight(module, self.tensor_buffer, distribution=weight_dist)

    def named_trainable_tensors(self):
        return named_trainable_tensors(self)

    def zero_grad(self, set_to_none=True):
        for _, parameter in self.named_trainable_tensors():
            if set_to_none or parameter.grad is None:
                parameter.grad = None
            else:
                parameter.grad.zero_()

    def checkpoint(self):
        # Stream matrices to CPU one at a time rather than decoding every
        # compressed matrix onto the GPU simultaneously through state_dict().
        from LCT.components.layers import dense_weight
        state = {name: dense_weight(p).detach().cpu() for name, p in self.named_trainable_tensors()}
        state.update({name: b.detach().cpu() for name, b in self.named_buffers()})
        return state

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self._embed(inputs)
        for block in self.blocks:
            x = block(x)
        return SoftcapCrossEntropy.head_loss(
            x, targets, self.proj.weight, self.norm2.gains, self.proj.bias,
            self.tensor_buffer, self.compress_activations, self.min_compress_elements,
        )

    def _embed(self, inputs: Tensor):
        return RMSNormFunction.apply(
            Embedding.apply(inputs, self.embed.weight), self.norm1.gains, self.tensor_buffer,
            self.compress_activations and torch.is_grad_enabled(),
            act_dist, self.min_compress_elements,
        )


@torch.no_grad()
def save_checkpoint(model, muon, step, path):
    names = {id(p): name for name, p in model.named_trainable_tensors()}
    momentum = {
        names[id(p)]: dense_weight(m).detach().cpu() if m is not None else None
        for p, m in zip(muon.params, muon.momentums)
    }
    torch.save({"step": step, "model": model.checkpoint(), "muon_momentum": momentum}, path)


def initialize_model(model):
    """Apply the original GPT initialization before compressing any weights."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("weight"):
                if "proj" in name:
                    p.zero_()
                elif "embed" in name:
                    p.normal_()
                else:
                    p.normal_(std=0.33**0.5 / p.size(-1)**0.5)
            elif name.endswith("bias"):
                p.zero_()
            elif name.endswith("gains"):
                p.fill_(1)
            else:
                raise ValueError(f"Uninitialized parameter: {name}")


def make_optimizers(model, compressed=False):
    named = dict(model.named_trainable_tensors())
    adam = SparseAdamW([
        dict(params=[named["embed.weight"]], lr=0.3),
        dict(params=[named["proj.weight"]], lr=1/320),
        dict(params=[p for p in named.values() if p.ndim < 2], lr=0.01),
    ], compressed=False, buffer=model.tensor_buffer)
    muon = SparseMuon([p for name, p in named.items() if name.startswith("blocks.") and p.ndim >= 2],
                      lr=0.035, weight_decay=0.025, compressed=compressed, buffer=model.tensor_buffer,
                      distribution=momentum_dist, zero_first_step=True, ns_iters=12)
    parameters = [p for group in adam.param_groups for p in group['params']] + muon.params
    assert len(parameters) == len(named) and {id(p) for p in parameters} == {id(p) for p in named.values()}
    for group in adam.param_groups:
        group['initial_lr'] = group['lr']
    return adam, muon


########################################
#                Setup                 #
########################################


def train(device):
    save_every = SAVE_EVERY
    torch.manual_seed(SEED)
    if TRAIN_STEPS < 1 or save_every < 1 or VAL_TOKENS < 1 or TRAIN_BATCH_TOKENS < 1:
        raise ValueError("Training steps, save frequency, and token counts must be positive")

    def print_log(s, console=True, log=True):
        if console:
            print(s, flush=True)
        if log:
            with open(logfile, "a") as f:
                print(s, file=f)

    def memory_stats():
        stats = (f"allocated:{torch.cuda.memory_allocated(device) / 2**20:.1f}MiB"
                 f" peak_allocated:{torch.cuda.max_memory_allocated(device) / 2**20:.1f}MiB")
        buffer = model.tensor_buffer
        if buffer is not None:
            used = buffer.capacity_bytes - sum(size for _, size in _free_regions_snapshot(buffer))
            stats += f" buffer_allocated:{used / 2**20:.1f}/{buffer.capacity_bytes / 2**20:.1f}MiB"
        return stats

    val_tokens = VAL_TOKENS
    batch_size = TRAIN_BATCH_TOKENS
    if TRAIN_MICROBATCH_SEQUENCES < 1 or VAL_MICROBATCH_SEQUENCES < 1:
        raise ValueError("Microbatch sizes must be positive")
    if batch_size % SEQ_LEN:
        raise ValueError("Token batch must divide evenly into sequences")
    sequences = batch_size // SEQ_LEN
    accumulation_steps = (sequences + TRAIN_MICROBATCH_SEQUENCES - 1) // TRAIN_MICROBATCH_SEQUENCES

    val_inputs, val_targets = next(data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens, SEQ_LEN, device="cpu"))

    model = GPT(vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, model_dim=MODEL_DIM,
                compress_activations=COMPRESS_ACTIVATIONS).to(device)
    initialize_model(model)
    if BUFFER and (COMPRESS_WEIGHTS or COMPRESS_ACTIVATIONS or COMPRESS_OPTIMISER):
        model.set_tensor_buffer(TensorBuffer(BUFFER_SIZE_MIB * 2**20, device=device))
    if COMPRESS_WEIGHTS:
        model.compress_weights()
    if COMPILE:
        model.compile()



    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
    model_dir = LOG_ROOT / (timestamp + "_lct")
    model_dir.mkdir(parents=True)
    logfile = model_dir / f"{timestamp}.txt"
    print(logfile)
    # we begin by logging this file itself
    print_log(Path(__file__).read_text(), console=False)
    print_log(f"COMPRESS_WEIGHTS={COMPRESS_WEIGHTS}, COMPRESS_ACTIVATIONS={COMPRESS_ACTIVATIONS}, "
              f"COMPRESS_OPTIMISER={COMPRESS_OPTIMISER}, BUFFER={BUFFER}")
    print_log(f"BF16 matrices; FP32 biases/gains; native FlashAttention; compile={COMPILE}")
    print_log(f"checkpoint head={CHECKPOINT_HEAD}, chunk tokens={CHUNK_TOKENS}")
    print_log(f"sequence length={SEQ_LEN}, batch={batch_size} tokens, "
           f"train microbatch={TRAIN_MICROBATCH_SEQUENCES} sequences, "
           f"accumulation={accumulation_steps}, validation microbatch={VAL_MICROBATCH_SEQUENCES} sequences")
    print_log("="*100)
    print_log(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
           + f" on {torch.cuda.get_device_name(device)}")
    print_log("="*100)


    ########################################
    #       Init & Optim Hyperparams       #
    ########################################

    # we want to minimize this while still reaching 3.28 val loss
    train_steps = TRAIN_STEPS
    optimizer1, optimizer2 = make_optimizers(model, compressed=COMPRESS_OPTIMISER)
    optimizers = [optimizer1, optimizer2]

    # learning rate schedule: stable then decay
    def set_hparams(step, cooldown_frac=0.7):
        progress = step / train_steps
        assert 0 <= progress < 1
        if progress < 1 - cooldown_frac:
            eta = 1.0
        else:
            eta = (1 - progress) / cooldown_frac
        for group in optimizer1.param_groups:
            group["lr"] = group["initial_lr"] * eta
        optimizer2.lr = 0.035 * eta
        optimizer2.neg_lr.fill_(-optimizer2.lr)
        optimizer2.decay.fill_(1 - optimizer2.lr * optimizer2.weight_decay)

    ########################################
    #        Training and Validation       #
    ########################################

    train_loader = data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size, SEQ_LEN)
    if model.tensor_buffer is not None:
        print_log(f"Tensor buffer: {BUFFER_SIZE_MIB} MiB")

    # save model at step 0 before any training
    with torch.no_grad():
        save_checkpoint(model, optimizer2, 0, model_dir / "0.pt")
    torch.cuda.synchronize(device)

    # start the clock
    training_time = 0
    last_val_step = 0
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    step_start = t0
    print("Starting training")
    for step in range(train_steps + 1):

        # --------------- VALIDATION SECTION -----------------
        val_step_freq = 125 if step / train_steps < 0.9 else 25
        if step == train_steps or step % val_step_freq == 0:
            # stop the clock
            torch.cuda.synchronize(device)
            time_since_last_val = time.perf_counter() - t0
            step_avg = time_since_last_val / (step - last_val_step) if step > 0 else float("nan")
            last_val_step = step
            training_time += time_since_last_val
            model.eval()
            torch.cuda.reset_peak_memory_stats(device)
            val_loss = 0
            with torch.no_grad():
                val_mbs = VAL_MICROBATCH_SEQUENCES
                for i in range(0, len(val_inputs), val_mbs):
                    v_in = val_inputs[i:i+val_mbs].to(device=device, non_blocking=True)
                    v_tgt = val_targets[i:i+val_mbs].to(device=device, non_blocking=True)
                    val_loss += model(v_in, v_tgt)
            val_loss /= val_tokens
            print_log(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
                   + f" time_since_last_val:{time_since_last_val:.3f}s {memory_stats()}", console=True)
            model.train()
            # start the clock again
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            step_start = t0

            del v_in, v_tgt
            torch.cuda.empty_cache()

        if step == train_steps:
            break

        # --------------- TRAINING SECTION -----------------
        torch.cuda.reset_peak_memory_stats(device)
        inputs, targets = next(train_loader)
        train_mbs = TRAIN_MICROBATCH_SEQUENCES
        train_loss = torch.zeros((), device=device)
        for i in range(0, len(inputs), train_mbs):
            loss = model(inputs[i:i+train_mbs], targets[i:i+train_mbs])
            train_loss += loss.detach()
            loss.backward()
        for name, p in model.named_trainable_tensors():
            assert p.grad is not None, name
        # set optimization hyperparameters and take a step
        set_hparams(step)
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
        if (step + 1) % save_every == 0 or step + 1 == train_steps:
            with torch.no_grad():
                save_checkpoint(model, optimizer2, step + 1, model_dir / f"{step + 1}.pt")
            torch.cuda.synchronize(device)
        torch.cuda.synchronize(device)
        approx_training_time = training_time + (time.perf_counter() - t0)
        step_time = time.perf_counter() - step_start
        step_start = time.perf_counter()
        print_log(f"step:{step+1}/{train_steps} train_time:{approx_training_time:.3f}s"
               + f" step_time:{step_time:.3f}s train_loss:{train_loss / batch_size:.5f} {memory_stats()}", console=True)



def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    train(device)


if __name__ == "__main__":
    main()
