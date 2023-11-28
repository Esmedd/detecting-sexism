import time
import tracemalloc
from params import *

def simple_time_and_memory_tracker(method):

    # ### Log Level
    # 0: Nothing
    # 1: Print Time and Memory usage of functions
    LOG_LEVEL = 1

    def method_with_trackers(*args, **kw):
        ts = time.time()
        tracemalloc.start()
        result = method(*args, **kw)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        te = time.time()
        duration = te - ts
        if LOG_LEVEL > 0:
            output = f"{method.__qualname__} executed in {round(duration, 2)} seconds, using up to {round(peak / 1024**2,2)}MB of RAM"
            print(output)
        return result

    return method_with_trackers
