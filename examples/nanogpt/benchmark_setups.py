"""Quick CUDA time/memory check for train_gpt_lct; edit the options below.

Uses the training model, initialization, data, optimizers and batch settings.
No validation, checkpoints or log files. First-time compilation can take longer.
"""
import gc
import time
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.nanogpt import train_gpt_lct as training

COMPRESS_WEIGHTS = True
COMPRESS_ACTIVATIONS = True
COMPRESS_OPTIMISER = True
BUFFER = True
COMPILE = True
LOG_GRAPH_BREAKS = True
BUFFER_SIZE_MIB = 64
SEQ_LEN = training.SEQ_LEN
TRAIN_BATCH_TOKENS = training.TRAIN_BATCH_TOKENS
SEQUENCES_PER_MICROBATCH = training.TRAIN_MICROBATCH_SEQUENCES
WARMUP_STEPS = 3
MEASURE_STEPS = 5


def main():
    import torch
    from LCT.tensor_buffer import TensorBuffer, _free_regions_snapshot

    # Emit graph-break reasons and source locations while Dynamo traces the model.
    torch._logging.set_logs(graph_breaks=LOG_GRAPH_BREAKS)

    sequences = TRAIN_BATCH_TOKENS // SEQ_LEN
    accumulation_steps = (sequences + SEQUENCES_PER_MICROBATCH - 1) // SEQUENCES_PER_MICROBATCH
    compressed = COMPRESS_WEIGHTS or COMPRESS_ACTIVATIONS or COMPRESS_OPTIMISER
    mode = "lct" if compressed else "dense"
    use_buffer = BUFFER and compressed
    print(f"Running {mode}: weights={COMPRESS_WEIGHTS}, activations={COMPRESS_ACTIVATIONS}, "
          f"Muon momentum={COMPRESS_OPTIMISER}, buffer={use_buffer}, "
          f"{SEQUENCES_PER_MICROBATCH} sequences/microbatch, "
          f"{accumulation_steps} microbatches/step", flush=True)


    torch.cuda.set_device(0)
    torch.manual_seed(training.SEED)
    model = training.GPT(training.VOCAB_SIZE, training.NUM_LAYERS, training.MODEL_DIM,
                         compress_activations=COMPRESS_ACTIVATIONS,
                         min_compress_elements=training.MIN_COMPRESS_ELEMENTS).cuda().train()
    training.initialize_model(model)

    if use_buffer:
        model.set_tensor_buffer(TensorBuffer(BUFFER_SIZE_MIB * 2**20, device="cuda"))
    if COMPRESS_WEIGHTS:
        model.compress_weights()
    if COMPILE:
        model.compile()

    adam, muon = training.make_optimizers(model, compressed=COMPRESS_OPTIMISER)
    tokens = TRAIN_BATCH_TOKENS
    loader = training.data_generator("data/fineweb10B/fineweb_train_*.bin", tokens, SEQ_LEN)
    peaks = []
    step_times = []
    for step in range(WARMUP_STEPS + MEASURE_STEPS):
        if step == WARMUP_STEPS:
            # Dynamo's graph-break tracing can leave tensor-owning cycles behind.
            # Release warmup garbage before measuring steady-state memory/time.
            gc.collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        inputs, targets = next(loader)
        total_loss = torch.zeros((), device="cuda")
        for i in range(0, len(inputs), SEQUENCES_PER_MICROBATCH):
            loss = model(inputs[i:i+SEQUENCES_PER_MICROBATCH], targets[i:i+SEQUENCES_PER_MICROBATCH])
            total_loss += loss.detach()
            loss.backward()
            del loss
        adam.step()
        muon.step()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 2**20
        warmup = step < WARMUP_STEPS
        if not warmup:
            peaks.append(peak)
            step_times.append(elapsed)
        buffer_stats = ""
        buffer = model.tensor_buffer
        if buffer is not None:
            used = buffer.capacity_bytes - sum(size for _, size in _free_regions_snapshot(buffer))
            buffer_stats = f"buffer_allocated:{used / 2**20:.1f}/{buffer.capacity_bytes / 2**20:.1f}MiB, "
        print(f"{mode} step {step+1} ({'warmup' if warmup else 'measured'}): "
              f"peak={peak:.1f} MiB, "
              f"after step={torch.cuda.memory_allocated()/2**20:.1f} MiB, "
              f"{buffer_stats}"
              f"loss={total_loss.item()/tokens:.4f}, time={elapsed:.2f}s", flush=True)
    print(f"Peak allocated after warmup: {max(peaks):.1f} MiB", flush=True)
    print(f"Mean step time after warmup: {sum(step_times)/len(step_times):.3f}s", flush=True)
    return {"peak_mib": max(peaks), "mean_step_seconds": sum(step_times)/len(step_times),
            "final_train_loss": total_loss.item()/tokens, "tokens_per_step": tokens}



if __name__ == "__main__":
    main()
