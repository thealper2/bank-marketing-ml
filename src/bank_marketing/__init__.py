"""Term deposit subscription prediction on the UCI Bank Marketing dataset."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bank-marketing-ml")
except PackageNotFoundError: # running from a source tree without installation
    __version__ = "0.0.0+unknown"
    
__all__ = ["__version__"]