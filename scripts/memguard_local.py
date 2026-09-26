"""Import-and-forget memory watchdog. Kills THIS process if the system gets close to swapping.
macOS RSS hides compressed pages, so we watch system-level available memory and swap growth instead."""
import threading, time, os, sys, psutil
MIN_AVAIL_GB = float(os.environ.get("MG_MIN_AVAIL_GB", 3.0))
MAX_SWAP_GROWTH_GB = float(os.environ.get("MG_MAX_SWAP_GROWTH_GB", 1.0))
_base_swap = psutil.swap_memory().used / 1e9
_peak = {"proc_rss": 0.0, "min_avail": 1e9}
def _watch():
    p = psutil.Process(os.getpid())
    while True:
        av = psutil.virtual_memory().available / 1e9; sw = psutil.swap_memory().used / 1e9 - _base_swap
        _peak["proc_rss"] = max(_peak["proc_rss"], p.memory_info().rss / 1e9); _peak["min_avail"] = min(_peak["min_avail"], av)
        if av < MIN_AVAIL_GB or sw > MAX_SWAP_GROWTH_GB:
            sys.stderr.write(f"\nMEMGUARD ABORT: available={av:.2f}GB swap_growth={sw:.2f}GB proc_rss={_peak['proc_rss']:.2f}GB\n"); sys.stderr.flush()
            os._exit(99)
        time.sleep(0.25)
threading.Thread(target=_watch, daemon=True).start()
def report():
    return f"memguard: peak_proc_rss={_peak['proc_rss']:.2f}GB min_sys_available={_peak['min_avail']:.2f}GB"
