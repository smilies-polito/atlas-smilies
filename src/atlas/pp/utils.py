import copy
from collections.abc import Sequence

from muon import MuData


def _safe_mudata(mudata: MuData, modalities: Sequence[str]) -> MuData:
    """Function that safely creates a new MuData object.

    Params:
    -------
    mudata
        MuData
    modalities
        list of strings containing modalities to keep

    Returns
    -------
    MuData

    Raises
    ------
    KeyError if any specified modality is available.

    """
    avail_mod = [mod for mod in modalities if mod in mudata.mod]
    if len(avail_mod) == 0:
        raise KeyError("No modalities available")

    new_data = MuData({k: mudata.mod[k] for k in avail_mod})
    new_data.uns = copy.deepcopy(mudata.uns)

    for storing in ["obsm", "obsp", "varm", "varp"]:
        old_s = getattr(mudata, storing)
        new_s = getattr(new_data, storing)
        for k, v in old_s.items():
            try:
                new_s[k] = v
            except (ValueError, KeyError, TypeError):
                pass

    for col in mudata.obs.columns:
        try:
            new_data.obs[col] = mudata.obs.loc[new_data.obs_names, col]
        except (ValueError, KeyError, TypeError):
            pass

    return new_data
