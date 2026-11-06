"""Training entry points.

Keep this package initializer lightweight.  Importing ``tinyllava.train.trag``
for the standalone loss tests must not eagerly import DeepSpeed and the full
Transformers training stack.
"""
