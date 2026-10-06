# Publishing VanillaSort

Build package releases from `feat/inference-package`. The `main` branch contains the research implementation.

The PyPI Trusted Publisher for `vanillasort` uses:

| Field | Value |
| --- | --- |
| GitHub owner | `IgarashiAkatuki` |
| Repository | `VanillaSort` |
| Workflow | `publish.yml` |
| Environment | `pypi` |

Before creating a release tag, update the version in `pyproject.toml` and `src/vanillasort/__init__.py`, then validate the distributions:

```bash
python -m pip install -e ".[dev]" build twine
pytest -q
python -m build
python -m twine check --strict dist/*
```

Commit the release changes on the package branch and push a matching version tag, such as `v0.1.0`. The [publishing workflow](../.github/workflows/publish.yml) checks the tag, builds the wheel and source distribution, tests the installed wheel, and publishes through PyPI Trusted Publishing.

After the workflow succeeds, verify the new version on [PyPI](https://pypi.org/project/vanillasort/) and install it in a fresh environment. Pretrained weights are downloaded from the immutable Hugging Face revision in `src/vanillasort/configs/default_model.json` on first inference.
