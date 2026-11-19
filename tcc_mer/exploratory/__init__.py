"""Fusão calibrada no gold set: análise exploratória, fora do pipeline principal.

O pipeline do artigo usa rótulos brutos e fusões sem parâmetros (:mod:`tcc_mer.fusion`).
Este pacote reúne a alternativa que foi explorada e abandonada: uma calibração afim
por canal e pesos de fusão ajustados no gold set. O resultado dela é citado no
artigo como desvio do protocolo, e o código fica aqui para reproduzi-lo.
"""
