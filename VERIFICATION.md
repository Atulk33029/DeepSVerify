# Package verification

- Project test suite: 207 passed.
- Streamlit AppTest: startup passed; all four headline metrics rendered.
- Exporter synthetic check: model tensors preserved exactly, threshold unchanged, optimizer state removed.
- Trained checkpoint comparison and cloud deployment remain to be run after adding your actual model. The real trained checkpoint is not included in this ZIP.

Run the suite with `python -m pytest tests code/preprocessing/test_*.py -q` after installing requirements-dev.txt and setting PYTHONPATH to code and the project root.
