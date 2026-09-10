"""Optional compiled execution of retained v2 CPU actors on Apple silicon."""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import subprocess
from pathlib import Path

import numpy as np

from .model import NumpyActor

_LIBRARY = None
_FLOAT = ctypes.POINTER(ctypes.c_float)
_INT = ctypes.POINTER(ctypes.c_int)


def library():
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    if platform.system() != "Darwin":
        raise RuntimeError("native actor currently requires macOS Accelerate")
    source = Path(__file__).with_suffix(".cpp")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    cache = Path(__file__).resolve().parents[2] / ".cache"
    cache.mkdir(exist_ok=True)
    target = cache / f"actor-{digest}.dylib"
    if not target.exists():
        temporary = cache / f"actor-{digest}-{os.getpid()}.dylib"
        subprocess.run(
            [
                "clang++",
                "-O3",
                "-std=c++17",
                "-shared",
                "-fPIC",
                "-framework",
                "Accelerate",
                str(source),
                "-o",
                str(temporary),
            ],
            check=True,
            capture_output=True,
        )
        temporary.replace(target)
    lib = ctypes.CDLL(str(target))
    lib.astro_create.argtypes = [ctypes.c_int] * 7 + [
        ctypes.c_float,
        ctypes.POINTER(_FLOAT),
        ctypes.c_int,
    ]
    lib.astro_create.restype = ctypes.c_void_p
    lib.astro_destroy.argtypes = [ctypes.c_void_p]
    lib.astro_options.argtypes = [
        ctypes.c_void_p,
        _FLOAT,
        _FLOAT,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        _FLOAT,
    ]
    lib.astro_values.argtypes = [ctypes.c_void_p, _FLOAT, _INT, ctypes.c_int, _FLOAT]
    _LIBRARY = lib
    return lib


def _ptr(array):
    return array.ctypes.data_as(_FLOAT)


class NativeActor(NumpyActor):
    def __init__(self, spec, weights):
        super().__init__(spec, weights)
        if spec.objective_version != 2:
            raise ValueError("native runtime requires objective_version=2")
        if (
            spec.residual_blocks < 0
            or not np.isfinite(spec.layer_norm_eps)
            or spec.layer_norm_eps <= 0
        ):
            raise ValueError("invalid residual depth or normalization epsilon")
        if any(
            value <= 0
            for value in (
                spec.state_size,
                spec.action_size,
                spec.hidden_size,
                spec.action_hidden_size,
                spec.bootstrap_heads,
                spec.families,
            )
        ):
            raise ValueError("model dimensions must be positive")
        names = []
        shapes = []

        def linear(prefix, width_in, width_out=None):
            names.extend([prefix + ".weight", prefix + ".bias"])
            shapes.extend(
                [
                    (width_in,) if width_out is None else (width_out, width_in),
                    (width_in if width_out is None else width_out,),
                ]
            )

        def residual(prefix, width):
            linear(prefix + ".norm", width)
            linear(prefix + ".fc1", width, 2 * width)
            linear(prefix + ".fc2", 2 * width, width)

        def trunk(prefix, blocks, width_in, width_out):
            linear(prefix + "_in", width_in, width_out)
            linear(prefix + "_norm", width_out)
            for index in range(blocks):
                residual(f"{prefix}_blocks.{index}", width_out)

        trunk("state", spec.residual_blocks, spec.state_size, spec.hidden_size)
        trunk("action", 1, spec.action_size, spec.action_hidden_size)
        trunk(
            "fusion",
            spec.residual_blocks,
            spec.hidden_size + spec.action_hidden_size,
            spec.hidden_size,
        )
        for head in range(spec.bootstrap_heads):
            residual(f"head_blocks.{head}", spec.hidden_size)
            linear(f"head_outputs.{head}", spec.hidden_size, spec.families)
        linear("value_output", spec.hidden_size, spec.families * spec.bootstrap_heads)
        for name, shape in zip(names, shapes, strict=True):
            if name not in weights or weights[name].shape != shape:
                raise ValueError(f"invalid native actor tensor {name}: expected {shape}")
            if not np.isfinite(weights[name]).all():
                raise ValueError(f"nonfinite native actor tensor {name}")
        self._arrays = [np.ascontiguousarray(weights[name], dtype=np.float32) for name in names]
        pointers = (_FLOAT * len(names))(*[_ptr(array) for array in self._arrays])
        self._lib = library()
        self._handle = self._lib.astro_create(
            spec.state_size,
            spec.action_size,
            spec.hidden_size,
            spec.action_hidden_size,
            spec.residual_blocks,
            spec.bootstrap_heads,
            spec.families,
            spec.layer_norm_eps,
            pointers,
            len(names),
        )

    def __del__(self):
        if getattr(self, "_handle", None):
            self._lib.astro_destroy(self._handle)
            self._handle = None

    def _options(self, state, actions, family, head):
        state = np.ascontiguousarray(state, dtype=np.float32).reshape(-1)
        actions = np.ascontiguousarray(actions, dtype=np.float32).reshape(
            (-1, self.spec.action_size)
        )
        if state.size != self.spec.state_size or not 0 <= family < self.spec.families:
            raise ValueError("invalid actor inputs")
        if head is not None and not 0 <= head < self.spec.bootstrap_heads:
            raise ValueError("invalid head")
        out = np.empty(
            (len(actions), self.spec.bootstrap_heads) if head is None else (len(actions),),
            dtype=np.float32,
        )
        self._lib.astro_options(
            self._handle,
            _ptr(state),
            _ptr(actions),
            len(actions),
            family,
            -1 if head is None else head,
            _ptr(out),
        )
        return out

    def predict_options(self, state, actions, family):
        return self._options(state, actions, family, None)

    def predict_option_head(self, state, actions, family, head):
        return self._options(state, actions, family, head)

    def predict_values(self, states, families):
        states = np.ascontiguousarray(states, dtype=np.float32).reshape((-1, self.spec.state_size))
        families = np.ascontiguousarray(families, dtype=np.int32).reshape(-1)
        if (
            len(families) != len(states)
            or np.any(families < 0)
            or np.any(families >= self.spec.families)
        ):
            raise ValueError("invalid value families")
        out = np.empty((len(states), self.spec.bootstrap_heads), dtype=np.float32)
        self._lib.astro_values(
            self._handle, _ptr(states), families.ctypes.data_as(_INT), len(states), _ptr(out)
        )
        return out
