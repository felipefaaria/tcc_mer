"""Ambiente de execução dos scripts: chaves de API e codificação do console."""

import sys

from .paths import ROOT

__all__ = ["load_env", "use_utf8_console"]


def load_env() -> None:
    """Lê ``<raiz>/.env`` (chaves de API) sem falhar se o python-dotenv não estiver instalado."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


def use_utf8_console() -> None:
    """Força UTF-8 (com substituição) na saída do console.

    O console do Windows costuma vir em cp1252, que não imprime todo Unicode (um título
    estrangeiro raro derrubaria o script no meio de uma coleta longa).
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
