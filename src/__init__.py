from functools import wraps
from typing import Callable, Any
import time


def ft_time(func: Callable) -> Callable:
    @wraps(func)
    def timer(*args: Any, **kwargs) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        timed = time.gmtime(end - start)
        string = time.strftime("%M:%S", timed)
        print(f"Total time of processing: {string}")
        return result
    return timer
