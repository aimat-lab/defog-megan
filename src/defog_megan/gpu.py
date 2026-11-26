"""
GPU sizing helpers shared by the training experiments.

Imports nothing MEGAN-derived, so it does not need the explainer stack.
"""

from __future__ import annotations

import typing as t


#: Fraction of device memory a worst-case training step may occupy by default.
#: Not a round number chosen for taste: the colour-graph base run accepted batch
#: 128 at a 21.7 GiB peak on a 24 GiB card (90%), trained for 140 epochs, and
#: then died at the optimizer step with 2.9 GiB stranded in allocator fragments.
DEFAULT_HEADROOM: float = 0.80


def largest_fitting_batch(model, graphs, upper: int, lower: int,
                          log: t.Callable[[str], None],
                          headroom: float = DEFAULT_HEADROOM) -> int:
    """Halve the batch size until a real forward+backward fits *with headroom*.

    Memory for a dense graph transformer scales with the square of the graph
    size -- the edge stream is ``(bs, n, n, de)`` -- so the safe batch is not
    something to guess. This probes it in seconds using the *largest* graphs
    available, so the estimate is worst-case rather than average.

    **"Fits once" is not "fits for 100k steps."** A batch is only accepted if its
    worst-case peak stays below ``headroom`` of total device memory; anything
    above that is treated as not fitting, because a long run's allocator
    fragments and a 10% margin does not survive it.

    :param model: a LightningModule exposing ``training_step``; must be on CUDA.
    :param graphs: the pool to draw worst-case batches from.
    :param upper: largest batch to try (halved on each rejection).
    :param lower: floor -- raise rather than train below this.
    :param log: where to report each probe decision.
    :returns: the largest batch whose worst-case peak left the required headroom.
    """
    import torch
    from torch_geometric.loader import DataLoader

    if not torch.cuda.is_available():
        log("no CUDA -- skipping the batch probe")
        return min(upper, 32)

    biggest = sorted(graphs, key=lambda g: -int(g.x.shape[0]))[:upper]
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    batch_size = upper

    while batch_size >= lower:
        try:
            batch = next(iter(DataLoader(biggest[:batch_size], batch_size=batch_size)))
            out = model.training_step(batch.to("cuda"), 0)
            loss = out["loss"] if isinstance(out, dict) else out
            loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 2 ** 30
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            if peak > headroom * total:
                log(f"batch probe: {batch_size} fits but peaks at {peak:.1f}/{total:.1f} "
                    f"GiB ({peak / total:.0%} > {headroom:.0%} limit) -- too tight "
                    f"for a long run, halving")
                batch_size //= 2
                continue

            worst_n = max(int(g.x.shape[0]) for g in biggest[:batch_size])
            log(f"batch probe: {batch_size} accepted (peak {peak:.1f}/{total:.1f} GiB, "
                f"{peak / total:.0%}, on the {worst_n}-node worst case)")
            return batch_size

        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            log(f"batch probe: {batch_size} does not fit, halving")
            batch_size //= 2

    raise RuntimeError(
        f"even batch size {lower} does not fit on this GPU -- the model or the "
        f"graph sizes need to come down before this run is worth starting"
    )
