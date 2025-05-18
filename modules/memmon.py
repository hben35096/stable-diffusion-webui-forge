import threading
import time
from collections import defaultdict
import torch

try:
    import torch_musa
    HAS_MUSA = torch_musa.is_available()
except ImportError:
    HAS_MUSA = False

class MemUsageMonitor(threading.Thread):
    run_flag = None
    device = None
    disabled = False
    opts = None
    data = None

    def __init__(self, name, device, opts):
        threading.Thread.__init__(self)
        self.name = name
        self.device = device
        self.opts = opts

        self.daemon = True
        self.run_flag = threading.Event()
        self.data = defaultdict(int)

        try:
            self.cuda_mem_get_info()
            if self.is_musa():
                torch_musa.memory_stats(self.device)
            else:
                torch.cuda.memory_stats(self.device)
        except Exception as e:  # AMD or other devices
            print(f"Warning: caught exception '{e}', memory monitor disabled")
            self.disabled = True

    def is_musa(self):
        return HAS_MUSA and 'musa' in str(self.device)

    def cuda_mem_get_info(self):
        if self.is_musa():
            index = self.device.index if self.device.index is not None else torch_musa.current_device()
            total = torch_musa.get_device_properties(index).total_memory
            free = total - torch_musa.memory_allocated(index)
            return free, total
        else:
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            return torch.cuda.mem_get_info(index)

    def run(self):
        if self.disabled:
            return

        while True:
            self.run_flag.wait()

            if self.is_musa():
                torch_musa.reset_peak_memory_stats()
            else:
                torch.cuda.reset_peak_memory_stats()

            self.data.clear()

            if self.opts.memmon_poll_rate <= 0:
                self.run_flag.clear()
                continue

            self.data["min_free"] = self.cuda_mem_get_info()[0]

            while self.run_flag.is_set():
                free, total = self.cuda_mem_get_info()
                self.data["min_free"] = min(self.data["min_free"], free)
                time.sleep(1 / self.opts.memmon_poll_rate)

    def dump_debug(self):
        print(self, 'recorded data:')
        for k, v in self.read().items():
            print(k, -(v // -(1024 ** 2)))

        print(self, 'raw memory stats:')

        if self.is_musa():
            tm = torch_musa.memory_stats(self.device)
        else:
            tm = torch.cuda.memory_stats(self.device)

        for k, v in tm.items():
            if 'bytes' not in k:
                continue
            print('\t' if 'peak' in k else '', k, -(v // -(1024 ** 2)))

        if not self.is_musa():
            print(torch.cuda.memory_summary())

    def monitor(self):
        self.run_flag.set()

    def read(self):
        if not self.disabled:
            free, total = self.cuda_mem_get_info()
            self.data["free"] = free
            self.data["total"] = total

            if self.is_musa():
                torch_stats = torch_musa.memory_stats(self.device)
            else:
                torch_stats = torch.cuda.memory_stats(self.device)

            self.data["active"] = torch_stats.get("active.all.current", 0)
            self.data["active_peak"] = torch_stats.get("active_bytes.all.peak", 0)
            self.data["reserved"] = torch_stats.get("reserved_bytes.all.current", 0)
            self.data["reserved_peak"] = torch_stats.get("reserved_bytes.all.peak", 0)
            self.data["system_peak"] = total - self.data["min_free"]

        return self.data

    def stop(self):
        self.run_flag.clear()
        return self.read()
