"""Needle backend exporters.

Each backend lives in its own module (`coreai`, `coreml`, `onnx`) and is
imported explicitly so that pulling in one backend never requires the optional
runtime dependencies of the others.
"""
