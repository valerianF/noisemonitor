import warnings
import functools



def validate_interval(indicator):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if self._noise_monitor.interval > 1:
                warnings.warn(f"Computing the {indicator} should be done with "
                            "an integration time equal to or below 1s. Results"
                            " might not be valid for this descriptor.\n")
            return func(self, *args, **kwargs)
        return wrapper
    return decorator