"""Load MEGAN checkpoints under a torch new enough to default ``weights_only=True``.

PyTorch 2.6 flipped ``torch.load``'s ``weights_only`` default to ``True``. MEGAN's
checkpoints predate that and store their hyperparameters in
``lightning_fabric.utilities.data.AttributeDict``, so on the cluster's torch 2.13 the load
raises ``UnpicklingError`` before any MEGAN code runs. The workstation's torch 2.5 still
defaults to ``False``, which is why this appears only on the cluster.

**Why the obvious fix does not work.** ``torch.serialization.add_safe_globals`` is the
sanctioned mechanism and it is not enough here: the restricted unpickler refuses
``SETITEMS`` on any dict subclass outside its own hard-coded set
(``dict``/``OrderedDict``/``Counter``) whether or not the class is allowlisted, and
``AttributeDict`` is a dict subclass. Allowlisting it changes nothing.

**Why not pass the flag through.** ``ConceptSpace.from_paths`` loads via upstream's
``Megan.load``, which exposes no ``weights_only`` argument, and the three MEGAN repos are
read-only for this project.

So this restores the old default *only for the duration of the MEGAN load* and puts it
back afterwards. The scope is the point: a module-level monkeypatch would silently disable
the check for every later ``torch.load`` in the process, including DeFoG's own base and
adapter checkpoints, which have no reason to give it up.

    with megan_checkpoint_load():
        space = ConceptSpace.from_paths(ckpt, clustering, processing)

On a torch that predates the change this is a no-op.
"""

import contextlib
import typing as t


@contextlib.contextmanager
def megan_checkpoint_load() -> t.Iterator[bool]:
    """Force ``weights_only=False`` for ``torch.load`` inside this block only.

    Yields ``True`` when the patch was actually needed (torch >= 2.6), ``False`` otherwise,
    so a caller can log which path it took.

    The MEGAN checkpoint is a local artifact produced by this project's own training runs,
    which is the condition under which lifting the check is reasonable; do not widen this
    to checkpoints arriving from anywhere else.
    """
    import torch

    if tuple(int(x) for x in torch.__version__.split(".")[:2]) < (2, 6):
        yield False
        return

    original = torch.load

    def patched(*args, **kwargs):
        kwargs["weights_only"] = False
        return original(*args, **kwargs)

    torch.load = patched
    try:
        yield True
    finally:
        # Restore unconditionally: an exception inside the block must not leave the
        # process with the safety check disabled.
        torch.load = original


def load_concept_space(model_path: str, clustering_path: str, processing_path: str,
                       **kwargs):
    """``ConceptSpace.from_paths`` with the checkpoint-compatibility scope applied."""
    from defog_megan.concepts import ConceptSpace

    with megan_checkpoint_load() as patched:
        space = ConceptSpace.from_paths(model_path, clustering_path, processing_path,
                                        **kwargs)
    space._loaded_with_weights_only_patch = patched
    return space
