"""Controle de qualidade por avaliador do experimento com ouvintes.

Os critérios de sinalização são objetivos e dizem respeito à **qualidade da
resposta**, nunca a concordar com o modelo:

* âncoras: sinal da valência ou do arousal correto em menos de 75% das âncoras;
* concordância com os demais: correlação item-resto (Spearman) abaixo de 0,15 em
  valência **e** em arousal;
* linha reta: mais de 70% das respostas na mesma categoria numa dimensão;
* pressa: mediana de menos de 12 s entre respostas consecutivas.

O avaliador sinalizado não é excluído da análise principal. O teste é repetido sem
ele, e também retirando um avaliador por vez (jackknife).
"""

import numpy as np
import pandas as pd
from scipy import stats

from .validation import (
    MIN_RATERS,
    DuelResult,
    analyzable,
    anchor_accuracy,
    duel,
    human_quadrant,
    repetition_agreement,
    track_labels,
)

__all__ = ["rater_quality", "flagged_raters", "main_test_without", "jackknife"]

ANCHOR_MIN_ACCURACY = 0.75
MIN_ITEM_REST_RHO = 0.15
MAX_MODAL_SHARE = 0.70
MIN_SECONDS_PER_ITEM = 12.0
MIN_PAIRED_TARGETS = 5
MIN_TARGETS_FOR_FLAGS = 10
MAX_PAUSE_SECONDS = 600


def _item_rest_agreement(own: pd.DataFrame, rest: pd.DataFrame) -> tuple[float, float, float]:
    """Correlação de Spearman de ``own`` com a média dos demais e acerto de quadrante.

    Returns:
        ``(rho_valência, rho_arousal, fração de faixas com o mesmo quadrante da mediana
        dos demais)``.
    """
    rest_stats = rest.groupby("item").agg(
        Vm=("V", "mean"), Am=("A", "mean"), Vmed=("V", "median"), Amed=("A", "median")
    )
    joined = own.set_index("item").join(rest_stats, how="inner")
    joined = joined[~joined["Vm"].isna()]
    rho_v = (
        stats.spearmanr(joined["V"], joined["Vm"]).statistic
        if len(joined) >= MIN_PAIRED_TARGETS
        else np.nan
    )
    rho_a = (
        stats.spearmanr(joined["A"], joined["Am"]).statistic
        if len(joined) >= MIN_PAIRED_TARGETS
        else np.nan
    )
    same_quadrant = (
        np.mean(
            [
                human_quadrant(v, a) == human_quadrant(vm, am)
                for v, a, vm, am in zip(joined["V"], joined["A"], joined["Vmed"], joined["Amed"])
            ]
        )
        if len(joined)
        else np.nan
    )
    return rho_v, rho_a, same_quadrant


def _seconds_between_responses(group: pd.DataFrame) -> float:
    """Mediana do intervalo entre respostas consecutivas, ignorando pausas longas."""
    gaps = group.sort_values("t")["t"].diff().dt.total_seconds().dropna()
    gaps = gaps[gaps < MAX_PAUSE_SECONDS]
    return gaps.median() if len(gaps) else np.nan


def rater_quality(responses: pd.DataFrame, item_map: pd.DataFrame) -> pd.DataFrame:
    """Métricas e sinalizações de cada avaliador, indexadas por ``rater``.

    Args:
        responses: saída de :func:`tcc_mer.validation.load_responses`.
        item_map: saída de :func:`tcc_mer.validation.load_item_map`.
    """
    responses = responses.assign(t=pd.to_datetime(responses["respondidoEm"]))
    targets = responses[responses["tipo"] == "alvo"].copy()
    targets["item"] = targets["item"].astype(str)
    anchors = anchor_accuracy(responses, item_map)
    repeats = repetition_agreement(responses)

    rows = []
    for rater, group in responses.groupby("rater"):
        own = targets[targets["rater"] == rater]
        rho_v, rho_a, same_quadrant = _item_rest_agreement(own, targets[targets["rater"] != rater])
        own_anchors = anchors[anchors["rater"] == rater]
        own_repeats = repeats[repeats["rater"] == rater]
        rows.append(
            {
                "rater": rater,
                "bloco": group["bloco"].iloc[0],
                "n": len(group),
                "alvos": len(own),
                "anc_V": own_anchors["valence_ok"].mean() if len(own_anchors) else np.nan,
                "anc_A": own_anchors["arousal_ok"].mean() if len(own_anchors) else np.nan,
                "rho_V": rho_v,
                "rho_A": rho_a,
                "quad_ok": same_quadrant,
                "neutra_V": (own["V"] == 0).mean(),
                "neutra_A": (own["A"] == 0).mean(),
                "modal_V": own["V"].value_counts(normalize=True).max(),
                "modal_A": own["A"].value_counts(normalize=True).max(),
                "sd_V": own["V"].std(),
                "sd_A": own["A"].std(),
                "rep_V_ident": (own_repeats["V_first"] == own_repeats["V_repeat"]).mean()
                if len(own_repeats)
                else np.nan,
                "listen_med_s": group["listenMs"].median() / 1000,
                "tempo_item_med_s": _seconds_between_responses(group),
                "conhece_nao": (group["conhece"] == "nao").mean(),
            }
        )

    quality = pd.DataFrame(rows).set_index("rater").round(2)
    quality["flag_anc"] = (quality["anc_V"] < ANCHOR_MIN_ACCURACY) | (
        quality["anc_A"] < ANCHOR_MIN_ACCURACY
    )
    quality["flag_conc"] = (quality["rho_V"] < MIN_ITEM_REST_RHO) & (
        quality["rho_A"] < MIN_ITEM_REST_RHO
    )
    quality["flag_reta"] = (quality["modal_V"] > MAX_MODAL_SHARE) | (
        quality["modal_A"] > MAX_MODAL_SHARE
    )
    quality["flag_pressa"] = quality["tempo_item_med_s"] < MIN_SECONDS_PER_ITEM
    quality["n_flags"] = quality[["flag_anc", "flag_conc", "flag_reta", "flag_pressa"]].sum(axis=1)
    return quality


def flagged_raters(quality: pd.DataFrame) -> list[str]:
    """Avaliadores com alguma sinalização e dados suficientes para julgá-los."""
    enough = quality["alvos"] >= MIN_TARGETS_FOR_FLAGS
    return quality[(quality["n_flags"] > 0) & enough].index.tolist()


def main_test_without(
    responses: pd.DataFrame,
    item_map: pd.DataFrame,
    excluded: list[str],
    min_raters: int = MIN_RATERS,
) -> DuelResult:
    """Teste principal (fusão simples contra áudio) sem os avaliadores em ``excluded``."""
    remaining = responses[~responses["rater"].isin(excluded)]
    return duel(analyzable(track_labels(remaining, item_map), min_raters))


def jackknife(responses: pd.DataFrame, item_map: pd.DataFrame) -> pd.DataFrame:
    """Teste principal repetido sem cada avaliador, uma linha por avaliador retirado."""
    rows = []
    for rater in sorted(responses["rater"].unique()):
        result = main_test_without(responses, item_map, [rater])
        rows.append(
            {
                "sem": rater,
                "n": result.n_tracks,
                "fusao": result.first,
                "audio": result.second,
                "prop_fusao": result.proportion,
                "p": result.p_value,
            }
        )
    return pd.DataFrame(rows).set_index("sem")
