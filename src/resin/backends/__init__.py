"""Optional model execution packages.

Each model package keeps the same small layout: ``config`` validates checkpoint
metadata, ``model`` defines modules and request-owned state, ``reference`` loads
weights and provides the readable runner, and ``cuda`` adapts native resource
plans to pinned kernels. Hardware-specific modules remain explicit and optional.

This top-level module is intentionally empty so importing it cannot initialize
PyTorch, CUDA, or a kernel compiler.
"""
