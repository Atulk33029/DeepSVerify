"""Phase 3A: writer-disjoint dataset splitting and split-audit logic.

Does NOT generate genuine/forgery pairs -- that is Phase 3B, not started
here. This package only decides which writers go into train/val/test and
audits that decision against duplicate groups and class-availability
requirements.
"""
