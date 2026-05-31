# Data And Code Availability Statement

The FPDE source code is available in this repository and is packaged as the
`fpde` Python distribution.

## Code Availability

The installable package source is under:

```text
src/fpde/
```

Install the released package from PyPI:

```bash
python -m pip install fpde
```

Install a local checkout in editable mode:

```bash
python -m pip install -e .
```

Run the test suite:

```bash
python -m pytest
```

## Example Code

The repository includes a runnable scikit-learn example:

```bash
python examples/minimal_fpde_example.py
```

Additional example material is available under `examples/`.

## Data Availability

This repository does not redistribute external datasets.

The minimal example uses the breast cancer dataset bundled with scikit-learn.
Any experiments that use additional datasets should cite the original dataset
source and document the download, preprocessing, and split procedure.

## Reproducibility Materials

Use the following files when preparing a reproducibility package:

- `README.md` for installation and quick-start instructions.
- `docs/reproducibility_checklist.md` for reporting requirements.
- `docs/api_reference.md` for public API details.
- `CITATION.cff` for citation metadata.
- `RELEASE_NOTES.md` for version history.
