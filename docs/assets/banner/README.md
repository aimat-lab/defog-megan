# The README banner

`banner.gif` (21.7 s, loops) is the animation at the top of the [README](../../../README.md). It has two parts in one picture.

**Above, what a concept is.** MEGAN, the self-explaining graph network, explains molecules, and the glow marks the part of a
molecule that drives the property it predicts. Three small molecules go through MEGAN, one per lane, and this repeats in
three waves with different molecules. Explanations that look alike gather in one cluster of the card on the right. A
cluster is a **concept**, and it has a colour and a letter (A orange, B pink, C purple, the colours of the paper). The nine
molecules are drawn for the banner. They are an illustration and **not data**.

**Below, a real recording of the sampler**, twice from the **same noise**:

1. the frozen DeFoG base with a logP adapter docked on it turns the noise into a molecule at logP 2.7 (requested: 2.5). MEGAN
   reads the molecule back, and its explanation lands in the concept the molecule happens to belong to, B (pink), which
   lights up;
2. a concept adapter docks next to the logP adapter and takes the request φ\*, the membership vector of the concept that is
   asked for (A, one filled cell, with a line from its cluster). The same noise ends in a different molecule at logP 2.5, and
   its explanation lands in A (orange).

The lower part follows the two-run format of the [defog-adapt](https://github.com/aimat-lab/defog-adapt) banner. The upper
part is new: it shows what the concept adapter's request refers to. The banner illustrates the process. It is **not a
measurement**: the pair was chosen from many for legibility.

## What is recorded, what is computed, what is drawn, what was chosen

| | |
|---|---|
| **Recorded** | The discrete state at each of the 200 steps: the element of every one of the 18 nodes and the bond order of every pair, for both runs, from identical initial noise (`trajectory.json`). Every atom and bond flip in the lower animation is a flip the model made. |
| **Computed** | The glow and the letter of each finished molecule, by the explainer. The glow is the set of atoms that MEGAN's explanation marks in its second channel (node importance of at least 0.5), the channel in which the requested concept lives. The letter is the concept of that channel the molecule belongs to (the lowest membership value, below 0.5). Run 2 belongs to the requested concept `ch1_cl34` (membership 0.21), drawn as A. Run 1 belongs to `ch1_cl30`, drawn as B, but only just (membership 0.496, the threshold is 0.5), and its membership in the requested concept is 0.85. The read-back arrow draws this result: the concept that the explainer found for the finished molecule. |
| **Drawn** | Node *positions*. A graph has no coordinates, so the noise is a scatter and the finished molecule an RDKit 2D layout, and the drawing glides from one to the other (each node to the scatter slot that minimises the total move to both finished layouts) while the categories flip. The pulses, the rings at the adapters and the time axis are decoration: a pulse stands for one application of the model. **The whole upper part is drawn.** The nine molecules are designed (their SMILES are in `banner.py`, every bond is drawn single), the glow on each marks its ring system by construction and is not MEGAN's output, and the clusters, their dots and the dots' flight are placed by hand. C exists only to make a third cluster. The path of the read-back arrow and the dot on it are decoration. |
| **Chosen** | The pair. 384 same-noise pairs were recorded for the request (12 seeds of 32), 88 of them had two molecules that draw cleanly (connected, no charges, ring sizes 5 to 7, a layout with no crossing bonds) and are near the requested logP, and 4 of those had run 1 explained by another concept and run 2 by the requested one. These were ranked by how close both land to the requested logP, how large and connected the explanations are, and how clean the denoising is to watch, and the best was taken (seed 4, batch index 19). |

## Settings of the recorded run

The ZINC base and the logP and concept adapters of the `zinc-logp` group of the model zoo. Run 1 uses the logP adapter at
weight 2 (the weight of the paper's property arm); run 2 uses the concept adapter `ch1_cl34` and the logP adapter at weights
1 and 1 (the paper's joint arm), composed in probability space. Both ask for logP 2.5, and both draw 18 heavy atoms with 200
steps, polydec time distortion, eta 0 and omega 0. These are chosen to be readable (eta 0 keeps what is right right, and a
fixed size keeps the layouts comparable), not to score: the paper's runs use 500 steps, eta 500 for molecules and the sizes of
the base's own distribution. The two molecules are `CCSc1ccc(NCC2CCNC2)cc1C#N` (logP 2.69) and
`CN(CC1CCOC1)C(=O)Cc1ccc(Br)cc1` (logP 2.49, Crippen).

## Files

| file | what it is |
|---|---|
| `banner.gif` | the animation, which is the whole banner: 1600 x 723 px, 326 frames at 15 fps, about 3.7 MB |
| `trajectory.json` | the recorded run: both trajectories, the shared noise, the explanation of each finished molecule and every setting above |
| `record_trajectories.py` | `record` same-noise pairs from the zoo and rank them, `rank` again, `export` one pair |
| `candidates.py` | the contact sheet of the ranked candidate pairs, drawn the way the banner draws molecules |
| `banner.py`, `molecules.py` | one frame of the banner (the timeline, the designed upper part and the recorded lower part), and the ball-and-stick molecule drawing and layouts |
| `render_banner.py` | renders every frame and assembles the GIF with one shared palette |

## Rebuilding

```bash
python record_trajectories.py --zoo $DEFOG_MEGAN_ZOO record --work WORK --concept ch1_cl34 --level 2.5
python record_trajectories.py export --work WORK --rank 0 --out trajectory.json
python render_banner.py frames --work FRAMES && python render_banner.py gif --work FRAMES --out banner.gif
```

Sampling is deterministic given `(seed, batch, index)` and the same library, checkpoints and hardware class, and those numbers
are in `trajectory.json`, but the ranking depends on the whole recording, so the pair is the one in the file. The recorder
needs the zoo (`python -m defog_megan.zoo fetch zinc-logp`), the `megan` and `chem` extras and, for speed, a GPU. The drawing
uses the author's figure toolchain (figlib and VectorCraft, found through `$FIGURE_SKILL`), which is not published, together
with Pillow, SciPy, RDKit and ffmpeg. So the GIF and `trajectory.json` are the artefacts to reuse; the scripts document exactly how
they were made.
