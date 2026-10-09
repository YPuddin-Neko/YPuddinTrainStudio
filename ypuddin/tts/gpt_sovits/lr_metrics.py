"""Observe optimizer steps without changing the upstream update or scheduler."""

from __future__ import annotations


class OptimizerStepLearningRate:
    def __init__(self):
        self._handles = []
        self._before_step = 0
        self._pending = None
        self._last_step = None
        self._last_rate = None

    def attach(self, optimizers):
        self.close()
        self._before_step = 0
        self._pending = self._last_step = self._last_rate = None
        try:
            for optimizer in optimizers:
                self._handles.append(optimizer.register_step_pre_hook(self._capture))
        except BaseException:
            self.close()
            raise

    def _capture(self, optimizer, args, kwargs):
        self._pending = float(optimizer.param_groups[0]["lr"])

    def begin_batch(self, step):
        self._before_step = int(step)
        self._pending = None

    def end_batch(self, step):
        step = int(step)
        if self._pending is not None and step > self._before_step:
            self._last_step, self._last_rate = step, self._pending
        # AMP can advance Lightning's counter without calling the raw optimizer.
        return self._last_rate if self._last_step == step else None

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
