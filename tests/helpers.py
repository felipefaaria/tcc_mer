"""Utilitários compartilhados pelos testes."""

import importlib.util

from tcc_mer.paths import ROOT


def load_script(relative_path: str):
    """Importa um script de ``scripts/`` ou ``docker/`` como módulo, sem executá-lo."""
    spec = importlib.util.spec_from_file_location(
        relative_path.replace("/", "_").removesuffix(".py"), ROOT / relative_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
