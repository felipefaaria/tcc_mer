"""Adaptador da API da Anthropic para a LF2: o único código específico de provedor.

Transforma um cliente do SDK numa função ``prompt -> resposta`` (o ``LLMClient`` de
:mod:`tcc_mer.lf2_llm`) e contabiliza tokens e custo. Este módulo não importa o SDK:
quem chama cria o cliente (``anthropic.Anthropic()``) e o injeta aqui.

O Opus 5 e o Sonnet 5 não aceitam ``temperature``; o controle de determinismo e de
custo neles é ``output_config.effort``. Só o Haiku 4.5 aceita ``temperature`` (fixada
em 0).
"""

from collections.abc import Iterable
from dataclasses import dataclass

from .lf2_llm import LLMClient

__all__ = ["PRICES", "Usage", "make_client", "estimate_cost"]

# USD por 1 milhão de tokens: (entrada, saída). Conferir na documentação da Anthropic
# antes de reutilizar, porque modelos e preços mudam.
PRICES = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
}
MODELS_WITH_TEMPERATURE = {"claude-haiku-4-5"}

# Resposta esperada: {"valence": 7.2, "arousal": 6.5}. Estimativa generosa para o pré-voo.
ESTIMATED_OUTPUT_TOKENS = 30
MAX_OUTPUT_TOKENS = 256


@dataclass
class Usage:
    """Tokens e chamadas acumulados numa sessão."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def cost(self, model: str) -> float:
        """Custo acumulado em USD."""
        price_in, price_out = PRICES[model]
        return (self.input_tokens * price_in + self.output_tokens * price_out) / 1e6


def make_client(sdk_client, model: str, effort: str = "low") -> tuple[LLMClient, Usage]:
    """Cria o cliente de LLM da LF2 e o contador de uso.

    Args:
        sdk_client: instância de ``anthropic.Anthropic``.
        model: modelo, uma chave de :data:`PRICES`.
        effort: ``output_config.effort`` para os modelos que não aceitam ``temperature``.

    Returns:
        ``(call, usage)``: ``call`` envia o prompt e devolve o texto da resposta, e
        ``usage`` acumula os tokens reais de cada chamada.
    """
    usage = Usage()

    def call(prompt: str) -> str:
        kwargs = {
            "model": model,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        }
        if model in MODELS_WITH_TEMPERATURE:
            # O SDK 1.x tirou ``temperature`` da assinatura tipada de ``create``; a API
            # ainda a aceita, então ela vai pelo corpo da requisição.
            kwargs["extra_body"] = {"temperature": 0.0}
        else:
            kwargs["output_config"] = {"effort": effort}
        response = sdk_client.messages.create(**kwargs)
        usage.input_tokens += response.usage.input_tokens
        usage.output_tokens += response.usage.output_tokens
        usage.calls += 1
        return "".join(block.text for block in response.content if block.type == "text")

    return call, usage


def estimate_cost(sdk_client, model: str, prompts: Iterable[str]) -> float:
    """Custo estimado em USD de enviar ``prompts``, sem gastar nada.

    Conta os tokens de entrada com ``count_tokens`` e supõe uma saída curta por prompt.
    """
    price_in, price_out = PRICES[model]
    total_in = n_prompts = 0
    for prompt in prompts:
        total_in += sdk_client.messages.count_tokens(
            model=model, messages=[{"role": "user", "content": prompt}]
        ).input_tokens
        n_prompts += 1
    return (total_in * price_in + n_prompts * ESTIMATED_OUTPUT_TOKENS * price_out) / 1e6
