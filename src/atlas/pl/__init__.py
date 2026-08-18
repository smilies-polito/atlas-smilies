from ._new_functions import embedding, fate_probabilities, trends
from .plots import plot_embedding, plot_fate_probabilities, plot_tree, plot_trends
from .utils import MultiBranchGAM

__all__ = [
    "embedding",
    "fate_probabilities",
    "trends",
    "plot_embedding",
    "plot_fate_probabilities",
    "plot_tree",
    "plot_trends",
]
