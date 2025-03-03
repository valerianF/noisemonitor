import warnings
import functools


def validate_column(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        column = kwargs.get('column', args[0] if args else None)
        if column not in self._noise_monitor.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")
        return func(self, *args, **kwargs)
    return wrapper

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