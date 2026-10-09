"""Phase 2 preprocessing package: EDA and duplicate-audit logic.

This package intentionally does NOT yet contain the actual image
transformation pipeline (foreground extraction, resize/pad, etc.) --
those decisions are pending real-data EDA results (see project chat
history / phase2_preprocessing_report.json once produced). Only the
data-driven analysis steps that don't depend on those pending decisions
are implemented so far: dimension/format statistics (eda.py) and
duplicate-group classification (duplicate_audit.py).
"""
