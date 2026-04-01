from importlib.metadata import PackageNotFoundError, version

from . import pl, pp, tl

__all__ = ["pl", "pp", "tl"]
try:
    __version__ = version("atlas-smilies")
except PackageNotFoundError:
    __version__ = "0.0.0"
