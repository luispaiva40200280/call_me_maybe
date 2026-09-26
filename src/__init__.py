from functools import wraps
from typing import Callable, Any
from time import time


def ft_time(func: Callable) -> Callable:
    @wraps(func)
    def timer(*args: Any, **kwargs) -> Any:
        start = time()
        func(*args, **kwargs)
        end = time()
        time_structure = time.gmtime(end - start)
        string = time.strftime("%M:%S", time_structure)
        print(f"Total time of processing: {string}")
    return timer
