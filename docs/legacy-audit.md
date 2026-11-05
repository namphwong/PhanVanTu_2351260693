# Phase 0 legacy audit

## Scope and environment

Reviewed the existing SLARDA repository at the start of Phase 0. The original
README specifies Python 3.x and PyTorch 1.7, but does not pin Python, package
versions, or provide an installable dependency manifest. The active execution
environment reports Python 3.12.10, PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu,
and CUDA unavailable. The new project baseline targets Python 3.10-3.12 and
PyTorch 2.5+.

## Findings from source review

- `Autorgressive_Adaptation/dataloader/dataloader.py` uses `torch.load` without
  an explicit `map_location`; loading checkpoints created on another device
  may fail. Its dataset transform also assigns normalized values back into
  `self.x_data` during item access, so repeated reads mutate the stored sample.
- `Autorgressive_Adaptation/models/models.py::CNN_Opp_HAR_SL.forward` calls
  unrestricted `squeeze()`. A batch of one can lose its batch dimension.
- `Autorgressive_Adaptation/models/models.py::Discriminator_AR.forward` uses
  `view` to reinterpret feature memory as sequence tokens instead of an
  explicit transpose; token order and axes may be wrong.
- `Autorgressive_Adaptation/models/models.py::CPC.forward` samples a timestep
  without constraining it against the available sequence length and allocates
  intermediate tensors on CPU. Short sequences or GPU use can fail.
- `Autorgressive_Adaptation/args.py` parses process command-line arguments at
  import time, which makes importing the legacy model from tests or other
  applications fragile.
- The legacy README documents PyTorch 1.7 and the project has no lockfile,
  package metadata, or automated smoke tests.

## Verification status

Installed the declared dependencies in a local `.venv` and ran
`python -m pytest -q --basetemp=.pytest-tmp`: both synthetic smoke tests passed
(dataloader batch contract and `CNN_Opp_HAR_SL` forward contract). Ruff reported
no issues, and `git diff --check` passed. The default pytest temp directory
outside the workspace was not writable, so the smoke command explicitly puts
its temp directory inside the repository; `.pytest-tmp/` is ignored by Git. No
training data or pretrained checkpoint is present. Phase 0 does not modify
legacy model behavior.

## Commit and remaining errors

Commit SHA: `2e30c9461dbd06a0f3dd5f1e81ce5d2dc5edb8c9` (Phase 0 commit).

Known legacy issues are listed above. These findings remain open for their
relevant implementation phases; the Phase 0 smoke gate passes on synthetic
inputs.
