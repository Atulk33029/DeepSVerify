"""Phase 3B: leakage-safe, deterministic pair generation for the Siamese
verification task. Consumes the Phase 3A split manifest as the sole
source of truth for writer->split assignment and eligibility -- never
rediscovers either.
"""
