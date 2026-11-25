"""Synchronized CUDA timing and process-aware device memory sampling."""
from collections import defaultdict
from contextlib import contextmanager
import os
import re
from threading import Event, Thread
import time


GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")


def nvml_device(nvml, torch_uuid, visibility):
    """Resolve cuda:0 without confusing CUDA order with NVML's physical order."""
    torch_uuid = str(torch_uuid)
    selected = visibility if GPU_UUID.fullmatch(visibility) else None
    reported = torch_uuid if GPU_UUID.fullmatch(torch_uuid) else None
    if selected and reported and selected.lower() != reported.lower():
        raise ValueError("CUDA device UUID differs from CUDA_VISIBLE_DEVICES")
    if selected or reported:
        return nvml.nvmlDeviceGetHandleByUUID(reported or selected)
    if nvml.nvmlDeviceGetCount() == 1:
        return nvml.nvmlDeviceGetHandleByIndex(0)
    raise ValueError("Cannot map cuda:0 to NVML; set CUDA_VISIBLE_DEVICES to the full GPU UUID")


class Meter:
    def __init__(self):
        import psutil
        import pynvml as nvml
        import torch

        self.torch, self.nvml = torch, nvml
        self.process = psutil.Process()
        nvml.nvmlInit()
        try:
            props = torch.cuda.get_device_properties(0)
            self.handle = nvml_device(nvml, getattr(props, "uuid", None),
                                      os.environ.get("CUDA_VISIBLE_DEVICES", ""))
            uuid = nvml.nvmlDeviceGetUUID(self.handle)
            self.device_uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
        except BaseException:
            nvml.nvmlShutdown()
            raise

    def close(self):
        self.nvml.nvmlShutdown()

    def usage(self):
        nvml = self.nvml
        memory = nvml.nvmlDeviceGetMemoryInfo(self.handle)
        processes = {}
        for list_processes in (nvml.nvmlDeviceGetComputeRunningProcesses,
                               nvml.nvmlDeviceGetGraphicsRunningProcesses):
            try:
                entries = list_processes(self.handle)
            except nvml.NVMLError_NotSupported:
                continue
            for entry in entries:
                amount = entry.usedGpuMemory
                if entry.pid != os.getpid() and amount is not None and 0 <= amount <= memory.total:
                    processes[entry.pid] = max(processes.get(entry.pid, 0), amount)
        return dict(device_used=int(memory.used), device_total=int(memory.total),
                    other_processes=processes, other_process_bytes=sum(processes.values()))

    def sample(self):
        torch = self.torch
        return {**self.usage(), "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(), "cpu_rss": self.process.memory_info().rss}

    @contextmanager
    def measure(self, record, *, interval=0.05):
        torch = self.torch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = self.sample()
        ongoing = dict(device_peak=before["device_used"], other_process_peak=before["other_process_bytes"])
        stop = Event()
        errors = []

        def worker():
            while not stop.wait(interval):
                try:
                    sample = self.usage()
                    ongoing["device_peak"] = max(ongoing["device_peak"], sample["device_used"])
                    ongoing["other_process_peak"] = max(ongoing["other_process_peak"], sample["other_process_bytes"])
                except Exception as exc:
                    errors.append(str(exc))
                    return

        thread = Thread(target=worker, daemon=True)
        thread.start()
        start = time.perf_counter()
        try:
            yield record
            torch.cuda.synchronize()
        finally:
            elapsed = time.perf_counter() - start
            stop.set()
            thread.join()
            after = self.sample()
            peaks = dict(allocated=torch.cuda.max_memory_allocated(),
                         reserved=torch.cuda.max_memory_reserved(),
                         device_used=max(ongoing["device_peak"], after["device_used"]))
            baseline = self.baseline
            record.update(seconds=elapsed, before=before, after=after, peak=peaks,
                          peak_increase={key: max(0, value - baseline[key]) for key, value in peaks.items()},
                          other_process_peak=max(ongoing["other_process_peak"], after["other_process_bytes"]),
                          device_sampler_errors=errors)


class QueryProbe:
    def __init__(self, torch, eos, stop_on_eos=False, *, check=True):
        self.torch, self.eos, self.stop_on_eos = torch, eos, stop_on_eos
        self.check = check
        self.times = defaultdict(float)
        self.per_cell = defaultdict(lambda: defaultdict(float))
        self.selected, self.tokens, self.eos_positions = [], [], []
        self.steps, self.cache_bytes = 0, 0
        self.stage = "query"

    @contextmanager
    def phase(self, name, cell=None):
        self.stage = name
        self.torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        self.torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        self.times[name] += seconds
        if cell is not None:
            self.per_cell[cell][name] += seconds

    def logits(self, logits, token):
        if self.check and not bool(self.torch.isfinite(logits).all()):
            raise FloatingPointError(f"Nonfinite logits at decode step {self.steps}")
        if token == self.eos:
            self.eos_positions.append(self.steps)
        self.steps += 1

    def cache(self, caches):
        size = 0
        for cache in caches:
            for layer in cache.to_legacy_cache():
                for tensor in layer["recurrent_state"]:
                    if tensor is not None:
                        if self.check and not bool(self.torch.isfinite(tensor).all()):
                            raise FloatingPointError("Nonfinite recurrent state")
                        size += tensor.numel() * tensor.element_size()
        self.cache_bytes = max(self.cache_bytes, size)

    def report(self):
        return dict(phases_seconds=dict(self.times), per_cell_seconds=self.per_cell,
                    selected_cell_ids=self.selected, output_token_ids=self.tokens,
                    eos_positions=self.eos_positions, peak_observed_recurrent_bytes=self.cache_bytes)
