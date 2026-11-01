from functools import wraps
from typing import Callable, Any
from time import time


def ft_time(func: Callable) -> Callable:
    @wraps(func)
    def timer(*args: Any, **kwargs) -> Any:
        start = time()
        result = func(*args, **kwargs)
        end = time()
        timed = time.gmtime(end - start)
        string = time.strftime("%M:%S", timed)
        print(f"Total time of processing: {string}")
        return result
    return timer
