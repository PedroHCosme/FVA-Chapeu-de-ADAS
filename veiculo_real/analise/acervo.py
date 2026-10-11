# -*- coding: utf-8 -*-
"""Acha um log da arvore ensaios/ pelo id do carro (AAAAMMDD_HHMMSS) consultando ensaios/catalogo.csv.

    from acervo import caminho, ids, linhas
    caminho('20260930_214209')      # Path de ensaios/real/roxo/2026-10-01/20260930T214209__vref-rampa__v1.3.csv
    ids(tipo='u-degrau')            # ids que casam com o filtro (valores do catalogo sao texto: moveu='True')
    linhas(dominio='real', controlador='avanco-kn4.0')

O id e unico (a arvore tem um arquivo por conteudo), entao o codigo de analise nao depende de onde o arquivo mora.
"""
import csv
from pathlib import Path

ENSAIOS = Path(__file__).resolve().parents[2] / 'ensaios'


def linhas(**filtros):
    with open(ENSAIOS / 'catalogo.csv', encoding='utf-8', newline='') as f:
        return [r for r in csv.DictReader(f) if all(r[k] == str(v) for k, v in filtros.items())]


def ids(**filtros):
    return sorted(r['id_carro'] for r in linhas(**filtros))


def caminho(id_carro):
    r = linhas(id_carro=id_carro)
    if len(r) != 1:
        raise KeyError(f'{id_carro}: {len(r)} linhas em {ENSAIOS / "catalogo.csv"} (esperava 1)')
    return ENSAIOS / r[0]['arquivo']
