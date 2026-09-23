import logging
from time import perf_counter


logger = logging.getLogger(__name__)


class Timer:
    times = {}

    def __init__(self, precision=3):
        self.precision = precision

    def __call__(self, func):
        def _(*args, **kws):
            t_start = perf_counter()
            res = func(*args, **kws)
            t_end = perf_counter()
            duration = t_end - t_start
            logger.info(f"Execution of {func.__name__} took {duration:.{self.precision}f}s")
            self.times[func.__name__] = duration
            return res
        return _


timer = Timer()
