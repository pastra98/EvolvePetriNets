import time

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None
        self.time_info = {}
        self.curr_func: str = ""
        self.curr_gen: int = None


    def start(self, funcname: str, gen: int):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self.curr_func = funcname
        self.curr_gen = gen
        self._start_time = time.perf_counter()


    def stop(self, funcname: str, gen: int) -> float:
        """Stop the timer, and return the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")
        if not funcname == self.curr_func:
            raise Exception("not timing the same func")

        elapsed_time = time.perf_counter() - self._start_time
        self.time_info[gen][funcname] = elapsed_time
        self._start_time = None
        return elapsed_time


    def get_gen_times(self, gen: int) -> dict:
        return self.time_info[gen]