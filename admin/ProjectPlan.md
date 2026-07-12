# Project Plan — Vision-Guided Grasping with a Domain-Randomization Study

A weekend+ project on Mac (Apple Silicon) using `gym_hil` (MuJoCo) + PyTorch.
Goal: build a perception-to-action grasping pipeline, then measure how domain
randomization affects its robustness. This doubles as a CV/portfolio piece and
a draft README.

## How to use this plan
- Each row is one ~15-minute chunk. Do it, hit **Done when**, then stop or continue.
- Every stage ends at a **git commit** — a clean stopping point.
- **Ref** numbers map to the Reference table at the bottom.
- When a chunk turns into a "find the right field" hunt (marked ⚠), box it: if
  you pass the time, hardcode a placeholder, leave a `# TODO verify`, and move on.

## Five habits (matter more than any single task)
1. Build in tiny pieces — write ~5 lines, run, check, continue.
2. Make things into functions — named block, clear inputs, one output.
3. Print to see — when unsure what something is, print its value and shape.
4. Ugly-but-working beats elegant-but-broken. Clean up later.
5. Commit every time something works — each commit is a save point.

---

## Stage 2 — Randomize the scene (a wrapper)
**You'll have learnt:** classes and inheritance; overriding a method to build on
top of existing code; reading and writing MuJoCo model fields.

| # | Task (~15 min) | Done when | Ref |
|---|----------------|-----------|-----|
| 2.1 | In `randomize.py`, write `RandomizeWrapper(gymnasium.Wrapper)` that overrides `reset()` but just calls `super().reset()` and returns it | Harness runs *through* the wrapper with identical behaviour (transparent pass-through) | 4 |
| 2.2 ⚠ | After reset, get `env.unwrapped.model` and print the cube's colour (look for `geom_rgba`) — don't change it yet | You can print the cube's current RGBA | 1, 5 |
| 2.3 | In `reset()`, set the cube's RGBA to random values each episode; save a few frames across resets | Frames show different-coloured cubes | 5, 6 |
| 2.4 | (Stretch) Same pattern for cube start position + a light value | Resets visibly vary in location/brightness | 5, 6 |
| 2.5 | `git add -A` and commit "domain randomization wrapper" | `git log` shows the commit | 3 |

---

## Stage 3 — Build a labeled dataset
**You'll have learnt:** file input/output; loops that produce data; keeping
images and labels aligned.

| # | Task (~15 min) | Done when | Ref |
|---|----------------|-----------|-----|
| 3.1 ⚠ | In `collect_data.py`, reset through the wrapper; pull `obs["pixels"]["front"]` and the cube's true position from the model/data state; print both shapes | One `(image, position)` pair in memory | 1, 5 |
| 3.2 | Save the image as a PNG (`imageio.imwrite`) and append the position to a CSV | A file exists you can reopen and read back correctly | 2 |
| 3.3 | Loop 3.1–3.2 over ~300 randomized resets; name files predictably (`img_0001.png`) so images/labels stay aligned | Folder of images + a labels file exist | 1 |
| 3.4 | Load a random saved pair back, show the image, print its label | 2–3 pairs eyeballed and aligned (catches the #1 dataset bug) | 2 |
| 3.5 | Commit `collect_data.py` (gitignore the data folder if large; keep the labels file) | Committed | 3 |

---

## Stage 4 — Build the perception model (a small CNN)
**You'll have learnt:** defining a model as a class (`nn.Module`); the forward
pass; thinking in tensor shapes; placing tensors on a device.

| # | Task (~15 min) | Done when | Ref |
|---|----------------|-----------|-----|
| 4.1 | In `model.py`, write `class CNN(nn.Module)` with `__init__` and a `forward` stub that returns zeros of shape `(batch, 3)` | Class imports and instantiates with no error | 7 |
| 4.2 | Add 2–3 `Conv2d` + `ReLU` + pooling layers in `__init__` | Still instantiates cleanly | 7 |
| 4.3 | Flatten and add a `Linear` head that outputs 3 numbers (the position) | Model defined end to end | 7 |
| 4.4 | Shape test: push a fake random tensor `(1, 3, 128, 128)` through; check output is `(1, 3)` | Output shape is correct (plumbing before water) | 7 |
| 4.5 | Move model + input to `mps`; run the fake forward on device | Forward runs on `mps`, output shape correct | 11 |
| 4.6 | Commit "perception model" | Committed | 3 |

---

## Stage 5 — Write the training loop
**You'll have learnt:** the standard ML training loop (forward → loss → backward
→ step); `Dataset`/`DataLoader`; the overfit-a-tiny-set debugging trick;
saving/loading weights. *This is the most reusable pattern in all of ML coding.*

| # | Task (~15 min) | Done when | Ref |
|---|----------------|-----------|-----|
| 5.1 | Write a `Dataset` class (`__init__`, `__len__`, `__getitem__`) that loads one image+label from disk as tensors | `dataset[0]` returns an `(image, position)` tensor pair | 8 |
| 5.2 | Wrap it in a `DataLoader`; pull one batch; print the batch shapes | A batch prints with correct image/label shapes | 8 |
| 5.3 | Create the loss (`MSELoss`) and optimizer (`Adam`) | Both objects created, no error | 9 |
| 5.4 | Write one training step: forward → loss → `backward()` → `step()` → `zero_grad()`; print the loss | One step runs and a loss number prints | 9 |
| 5.5 | Overfit test: train on just 10 images for many epochs — loss should fall near zero | Loss drops close to 0 on the tiny set (proves the loop works) | 9 |
| 5.6 | Train on the full set; watch loss fall; `torch.save` the weights | Weights file saved and loss decreased | 9, 10 |
| 5.7 | Commit "training loop + weights" | Committed | 3 |

---

## Stage 6 — Close the loop
**You'll have learnt:** composing separate pieces into one system; loading a
saved model; testing each half separately before wiring them; a simple
proportional controller.

| # | Task (~15 min) | Done when | Ref |
|---|----------------|-----------|-----|
| 6.1 | Load the weights (`torch.load`); write `predict_position(obs)` returning 3 numbers | A prediction prints for one observation | 10 |
| 6.2 | Compare predicted vs true position over a few resets; print the error | You can see the error magnitude | 1 |
| 6.3 | Write the controller: `action = k * (target - current_ee_pos)`, clipped to the action space | Function returns a valid action | 1 |
| 6.4 | Test the controller on the **true** position first (not the model) | Arm visibly moves toward the cube in the video | 1, 2 |
| 6.5 | Add "close gripper when close enough" | Arm reaches and closes | 1 |
| 6.6 | Swap the true position for the model's prediction → full perception policy | Policy runs end to end | — |
| 6.7 | Run it through `evaluate()`; record the success rate | Success rate prints | 1 |
| 6.8 | Commit "closed loop" | Committed | 3 |

---

## Stage 7 — Run the experiment
**You'll have learnt:** running a clean controlled experiment (change one
variable); saving results to a file; basic plotting.

| # | Task (~15 min) | Done when | Ref |
|---|----------------|-----------|-----|
| 7.1 | Set a random seed; add a domain-randomization on/off flag to data collection/training | The flag switches randomization cleanly | 5 |
| 7.2 | Train perception **with** DR → save `model_dr.pt` (mostly waiting) | File saved | 9, 10 |
| 7.3 | Train perception **without** DR → save `model_nodr.pt` | File saved | 9, 10 |
| 7.4 | Evaluate both under randomized conditions; write the numbers to a file | A results file with both scores | 1 |
| 7.5 | One matplotlib bar/line plot comparing them; save to `results/` | Plot image saved | 12 |
| 7.6 | Commit results + plot | Committed | 3 |

---

## Stage 8 — Write it up
**You'll have learnt:** documentation and repo hygiene; framing work for a reader
or hiring manager. *This is the part most people skip — and real professional signal.*

| # | Task (~15 min) | Done when | Ref |
|---|----------------|-----------|-----|
| 8.1 | README skeleton: Problem / Method / Results / How to run | Headings exist | 3 |
| 8.2 | Put the GIF + plot at the top; write a 2–3 sentence result | Visuals + result render on GitHub | 2 |
| 8.3 | Write "how to run"; test it yourself from a clean clone | You can follow your own steps start to finish | 3 |
| 8.4 | Add a short "why this matters for sim-to-real" paragraph + list the stack (MuJoCo, Gymnasium, PyTorch, domain randomization, hand-written training loop) | Paragraph + tech list present | — |
| 8.5 | Final commit; make the repo public | Public repo with all artifacts | 3 |

---

## References

| # | Source | Use it for | Link |
|---|--------|-----------|------|
| 1 | gym-hil repo | Obs/action API, example loop, where cube/model state lives | https://github.com/huggingface/gym-hil |
| 2 | imageio docs | `mimsave` (GIF/video), `imwrite`/`imread` (PNGs) | https://imageio.readthedocs.io |
| 3 | Git basics | add/commit/log, `.gitignore` | https://git-scm.com/book/en/v2/Git-Basics-Recording-Changes-to-the-Repository |
| 4 | Gymnasium wrappers | Subclassing `Wrapper`, overriding `reset` | https://gymnasium.farama.org/api/wrappers/ |
| 5 | MuJoCo Python bindings | Reading/writing `geom_rgba`, lights, body/`qpos` positions | https://mujoco.readthedocs.io/en/stable/python.html |
| 6 | MuJoCo (DeepMind repo) | Domain-randomization concepts to copy | https://github.com/google-deepmind/mujoco |
| 7 | PyTorch — Build Model | Defining a model with `nn.Module` + `forward` | https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html |
| 8 | PyTorch — Datasets & DataLoaders | Custom `Dataset`, batching with `DataLoader` | https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html |
| 9 | PyTorch — Optimization | The training loop (loss, backward, optimizer step) | https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html |
| 10 | PyTorch — Save & Load Model | `torch.save` / `torch.load`, `state_dict` | https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html |
| 11 | PyTorch — MPS (Apple Silicon) | Running on the Mac GPU (`device="mps"`) | https://docs.pytorch.org/docs/stable/notes/mps.html |
| 12 | matplotlib — pyplot | Making and saving the comparison plot | https://matplotlib.org/stable/tutorials/pyplot.html |

**One caveat:** the gym-hil / MuJoCo field names (cube colour, position) depend on
how gym-hil defines its model, and this stack shifts across versions. Use the
gym-hil source (Ref 1) to find the real geom/body names, then the MuJoCo docs
(Ref 5) for how to read/set them — that two-source cross-check *is* the work in
chunks 2.2 and 3.1, not a detour. 