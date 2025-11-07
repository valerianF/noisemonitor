#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    # Import weathercan only if optional dependencies are available
    from . import weathercan
    __all__ = ['weathercan']
except ImportError as e:
    import warnings
    warnings.warn(
        f"Weather functionality is not available. "
        f"To enable weather features, install optional dependencies with: "
        f"pip install noisemonitor[weather]\n"
        f"Missing dependency: {e}",
        ImportWarning,
        stacklevel=2
    )
    # Create a dummy module that will raise helpful errors
    class _WeathercanUnavailable:
        def __getattr__(self, name):
            raise ImportError(
                "Weather functionality requires optional dependencies. "
                "Install them with: pip install noisemonitor[weather]"
            )
    
    weathercan = _WeathercanUnavailable()
    __all__ = ['weathercan']