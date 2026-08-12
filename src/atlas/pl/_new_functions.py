from collections.abc import Sequence

import muon as mu
from anndata import AnnData
from matplotlib.axes import Axes
from muon import MuData

#: Entry point that produces the embedding this module draws by default.
_EMBEDDING_PRODUCER = "`atlas.tl.umap`"


def _resolve_basis(mudata: MuData, basis: str) -> None:
    """Fail here when no embedding answers to ``basis``.

    :func:`muon.pl.embedding` resolves the name itself, and also accepts one qualified by a
    modality, so a name carrying a qualifier is left to it rather than second-guessed. What
    is caught here is the plain name that matches nothing: muon reports it without naming
    the step that produces an embedding, which is what a caller in that position needs.
    """
    if basis in mudata.obsm or f"X_{basis}" in mudata.obsm:
        return

    modality, _, _ = basis.partition(":")
    if modality in mudata.mod:
        return

    # `.obsm` also carries one membership flag per modality, which are not embeddings.
    available = ", ".join(sorted(key for key in mudata.obsm if key not in mudata.mod))
    if not available:
        raise KeyError(
            f"'{basis}' not in mudata.obsm, which carries no embedding; run {_EMBEDDING_PRODUCER} to compute one"
        )

    raise KeyError(
        f"'{basis}' not in mudata.obsm; run {_EMBEDDING_PRODUCER} to compute an embedding, "
        f"or pass `basis` naming one of those present ({available})"
    )


def _resolve_color(mudata: MuData, color: str | Sequence[str], use_raw: bool | None) -> None:
    """Fail here when nothing answers to a colour key.

    Resolution mirrors :func:`muon.pl.embedding`'s: observations first, then the features of
    each modality, where a key may name its modality to disambiguate. Only the failing case
    is handled; a key that resolves is forwarded untouched, so muon stays the one
    implementation of resolution and this cannot drift from it.
    """
    keys = [color] if isinstance(color, str) else list(color)

    for key in keys:
        if key in mudata.obs.columns:
            continue

        carriers = [m for m in mudata.mod if _in_modality(mudata.mod[m], m, key, use_raw)]

        if len(carriers) == 1:
            continue

        # A name several modalities carry is ambiguous only when it is unqualified: the
        # qualifier is what tells them apart. Left to muon it fails while joining the
        # columns together, reporting an overlap rather than the choice to be made.
        if len(carriers) > 1 and ":" not in key:
            raise KeyError(
                f"'{key}' is a feature of more than one modality ({', '.join(sorted(carriers))}); "
                f"pass `<modality>:{key}` naming the one to color by"
            )

        if carriers:
            continue

        searched = ", ".join(sorted(mudata.mod))
        raise KeyError(
            f"'{key}' is neither a column of mudata.obs nor a feature of any modality "
            f"(searched mudata.obs and the var_names of {searched}); "
            "pass `<modality>:<feature>` to name a feature of a particular modality"
        )


def _in_modality(adata: AnnData, modality: str, key: str, use_raw: bool | None) -> bool:
    """Whether ``key`` names a feature of ``modality``, with or without naming it."""
    name = key.split(":", 1)[1] if key.startswith(f"{modality}:") else key

    if name in adata.var_names:
        return True

    return (use_raw is None or use_raw) and adata.raw is not None and name in adata.raw.var_names


def embedding(
    mudata: MuData,
    basis: str = "umap",
    color: str | Sequence[str] | None = None,
    *,
    use_raw: bool | None = None,
    layer: str | None = None,
    **kwargs,
) -> Axes | list[Axes] | None:
    """Plot cells in a stored embedding, coloured by observations or by measured features.

    The multi-modal object is drawn directly, by :func:`muon.pl.embedding`. A colour key is
    resolved against ``mudata.obs`` and against the features of every modality,

    Parameters
    ----------
    mudata
        Multimodal annotated data object carrying an embedding in ``.obsm``.
    basis
        Name of the embedding in ``mudata.obsm``, with or without the ``X_`` prefix under
        which embeddings are stored. An embedding inside a modality is named
        ``"<modality>:<basis>"``. Defaults to the embedding :func:`atlas.tl.umap` produces.
    color
        Key, or keys, to colour by. Each names a column of ``mudata.obs`` or a feature of
        one of the modalities, optionally qualified as ``"<modality>:<feature>"``. When
        ``None`` the cells are drawn uncoloured.
    use_raw
        Whether a feature is read from the ``.raw`` attribute of the modality it belongs to,
        as in :func:`muon.pl.embedding`. When ``None`` it is used if present and no valid
        ``layer`` is given.
    layer
        Layer of the modality a feature is read from, as in :func:`muon.pl.embedding`.
    **kwargs
        Additional keyword arguments passed to :func:`scanpy.pl.embedding`.

    Returns
    -------
    The axes drawn on, a list of them when several keys are given, or ``None`` when the
    figure is shown rather than returned.

    Raises
    ------
    KeyError
        If ``basis`` names no embedding on the object, or if a colour key names neither an
        observation nor a feature of any modality.

    Notes
    -----
    A colour key is resolved against ``mudata.obs`` before the modalities, so a name
    occurring in both is taken from ``.obs``. Qualifying it as ``"<modality>:<feature>"``
    selects the feature instead, and is also what distinguishes a feature name carried by
    more than one modality.

    Colouring by a categorical key records the colours chosen for it on ``mudata`` as
    ``.uns["<key>_colors"]``, following the convention the scverse ecosystem reads. The
    values therefore keep their colours across calls, and agree with figures drawn by
    :func:`scanpy.pl.embedding` or :func:`muon.pl.embedding` directly. Colours already
    recorded there are used as they are. This annotates the object that is passed.

    No colour map is imposed. When ``cmap`` is not given, the matplotlib default applies, so
    a colour map set globally through :func:`scanpy.set_figure_params` is respected.

    ``save`` writes to ``figures/<basis><save>``, as in :func:`scanpy.pl.embedding`.

    Examples
    --------
    >>> atlas.pl.embedding(mudata, color="pseudotime")
    >>> atlas.pl.embedding(mudata, color=["pseudotime", "shannon_entropy"], ncols=2)
    >>> atlas.pl.embedding(mudata, color=["GATA1", "activity:GATA1"])
    """
    _resolve_basis(mudata, basis)

    if color is not None:
        _resolve_color(mudata, color, use_raw)

    return mu.pl.embedding(mudata, basis=basis, color=color, use_raw=use_raw, layer=layer, **kwargs)
