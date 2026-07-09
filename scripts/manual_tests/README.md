# Manual validation programs

These programs are long-running, diagnostic validations rather than automated
pytest tests. They print detailed numerical output and may write plots or data
under `outputs/`.

Run them explicitly from the repository root, for example:

```bash
python scripts/manual_tests/solver_basic.py
python scripts/manual_tests/solver_intermediate.py
python scripts/manual_tests/solver_uranium_lda.py
python scripts/manual_tests/mesh_builder.py
```

The automated regression suite lives in `tests/` and follows pytest's
`test_*.py` naming convention. Failures in that suite raise normally and are
therefore visible to CI.
