"""Testes de tcc_mer/anthropic_adapter.py com um SDK falso."""

from types import SimpleNamespace

import pytest

from tcc_mer.anthropic_adapter import PRICES, Usage, estimate_cost, make_client


class FakeSDK:
    """Imita ``anthropic.Anthropic`` o suficiente para o adaptador."""

    def __init__(self, text='{"valence": 7, "arousal": 3}', input_tokens=100, output_tokens=20):
        self.calls = []
        self._response = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text=text),
                SimpleNamespace(type="thinking", text="ignorado"),
            ],
            usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
        )
        self.messages = SimpleNamespace(create=self._create, count_tokens=self._count_tokens)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response

    @staticmethod
    def _count_tokens(**kwargs):
        return SimpleNamespace(input_tokens=1000)


def test_cliente_devolve_so_os_blocos_de_texto():
    call, _ = make_client(FakeSDK(), "claude-sonnet-5")
    assert call("prompt") == '{"valence": 7, "arousal": 3}'


def test_modelos_com_esforco_nao_recebem_temperature():
    sdk = FakeSDK()
    make_client(sdk, "claude-sonnet-5", effort="low")[0]("prompt")
    assert sdk.calls[0]["output_config"] == {"effort": "low"}
    assert "extra_body" not in sdk.calls[0]


def test_haiku_recebe_temperature_zero_pelo_corpo_da_requisicao():
    sdk = FakeSDK()
    make_client(sdk, "claude-haiku-4-5")[0]("prompt")
    assert sdk.calls[0]["extra_body"] == {"temperature": 0.0}
    assert "output_config" not in sdk.calls[0]


def test_uso_acumula_tokens_e_chamadas():
    call, usage = make_client(FakeSDK(input_tokens=100, output_tokens=20), "claude-sonnet-5")
    call("a")
    call("b")
    assert (usage.input_tokens, usage.output_tokens, usage.calls) == (200, 40, 2)


def test_custo_usa_o_preco_do_modelo():
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000, calls=1)
    price_in, price_out = PRICES["claude-sonnet-5"]
    assert usage.cost("claude-sonnet-5") == pytest.approx(price_in + price_out)


def test_estimativa_de_custo_conta_tokens_de_entrada_e_supoe_saida_curta():
    sdk = FakeSDK()
    estimate = estimate_cost(sdk, "claude-sonnet-5", ["p1", "p2"])
    price_in, price_out = PRICES["claude-sonnet-5"]
    assert estimate == pytest.approx((2000 * price_in + 2 * 30 * price_out) / 1e6)
    assert sdk.calls == []  # count_tokens não gasta chamadas de geração
