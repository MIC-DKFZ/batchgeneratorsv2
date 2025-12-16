from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import List, Optional, Dict, Any

import numpy as np

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


class ComposeTransforms(BasicTransform):
    def __init__(self, transforms: List[BasicTransform]):
        super().__init__()
        self.transforms = transforms

    def apply(self, data_dict, **params):
        for t in self.transforms:
            data_dict = t(**data_dict)
        return data_dict


@dataclass
class _TimingStats:
    total_s: float = 0.0
    n: int = 0

    def add(self, dt: float) -> None:
        self.total_s += dt
        self.n += 1

    @property
    def mean_s(self) -> float:
        return self.total_s / self.n if self.n > 0 else 0.0


class TimedComposeTransforms(BasicTransform):
    """
    ComposeTransforms variant that measures per-transform wall-clock time and prints
    average times after every `print_every` calls to `apply`.

    Notes:
    - Measures wall clock time via perf_counter.
    - For CPU-only pipelines, this is representative. For GPU, you'd need synchronization.
    """

    def __init__(self, transforms: List[BasicTransform], print_every: int = 100, name: Optional[str] = None, p_write: float = 1.0):
        super().__init__()
        if print_every <= 0:
            raise ValueError("print_every must be >= 1")
        self.transforms = transforms
        self.print_every = int(print_every)
        self.name = name or self.__class__.__name__

        self._iter = 0
        self.p_write = p_write
        self._stats: Dict[int, _TimingStats] = {i: _TimingStats() for i in range(len(transforms))}

    def reset_timings(self) -> None:
        """Reset accumulated timing statistics and iteration counter."""
        self._iter = 0
        for s in self._stats.values():
            s.total_s = 0.0
            s.n = 0

    def _transform_display_name(self, t: BasicTransform) -> str:
        # Prefer explicit "name" attribute if present, otherwise class name
        return getattr(t, "name", None) or t.__class__.__name__

    def _print_report(self) -> None:
        lines = [f"[{self.name}] Average transform times over last {self._iter} iterations:"]
        # Print in pipeline order
        for i, t in enumerate(self.transforms):
            st = self._stats[i]
            lines.append(f"  {i:02d}  {self._transform_display_name(t)}: {st.mean_s * 1e3:.3f} ms")
        print("\n".join(lines), flush=True)

    def apply(self, data_dict: Dict[str, Any], **params) -> Dict[str, Any]:
        for i, t in enumerate(self.transforms):
            t0 = perf_counter()
            data_dict = t(**data_dict)
            dt = perf_counter() - t0
            self._stats[i].add(dt)

        self._iter += 1
        if self._iter % self.print_every == 0 and np.random.uniform() < self.p_write:
            self._print_report()

        return data_dict