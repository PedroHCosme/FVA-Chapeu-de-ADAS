# -*- coding: utf-8 -*-
"""Monta o ZIP de entrega (relatorio PDF + fonte LaTeX + codigo + logs usados).
Uso (nesta pasta, depois de figuras_relatorio.py e pdflatex):  python monta_zip.py
"""
import os, zipfile

AQUI = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(AQUI)
OUT = os.path.join(AQUI, 'FVA_parte1_ADAS_longitudinal.zip')
PULA_DIRS = {'.venv', '__pycache__', 'saida', '.git', 'outros', 'capitulo', 'tools', 'adas', 'parada', 'sonar_dados', 'dados',
             'dados_so_proporcional', 'coppeliasim', 'assets'}
LEIAME = """FVA 2026/2 - Trabalho Parte 1: ADAS longitudinal
===============================================
relatorio.pdf          o relatorio (10 paginas max.; capa fora da contagem)
relatorio_fonte/       fonte LaTeX + logs usados + figuras_relatorio.py
codigo/simulador/      main.py (controle PI+FF), fva_car/, identifica.py, figuras.py, logs/ (todos os ensaios do CoppeliaSim)
codigo/veiculo_real/   main.py (lei avanco_rt + supervisor ADAS = FCW/AEB), fva_car/, analise/ (sintonia, parada, sonar, testa_adas)

Reproduzir os dados do relatorio (Python 3 com numpy e matplotlib):
  cd relatorio_fonte
  python figuras_relatorio.py        # le logs/sim e logs/real; gera imgs/*.pdf, tab_aeb.tex, numeros.tex
  pdflatex main.tex  (duas vezes)    # gera main.pdf = relatorio.pdf

Refazer ensaios:
  Simulador: abrir simulador_rampa.ttt (repositorio da disciplina) no CoppeliaSim; cd codigo/simulador; python main.py.
             Logs em logs/AAAAMMDD_HHMMSS/car.csv. Identificacao: python identifica.py --tmax=8
  Carro real: editar ENSAIO em codigo/veiculo_real/main.py e copiar so esse arquivo para o carro; cada corrida grava
             car.csv e sonar.csv (colunas t,d,valid,alvo,adas: adas 0 livre, 1 FCW/limitando referencia, 2 AEB).

Observacao sobre os logs reais: as datas dos logs usam o relogio do carro (atrasado e variavel); a pasta so serve para ordenar.
Documentacao do metodo: codigo/simulador/IDENTIFICACAO.md e CONTROLE_LONGITUDINAL.md.
"""


def add(z, src, arc):
    z.write(src, arc)


with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:
    z.writestr('LEIAME.txt', LEIAME)
    add(z, os.path.join(AQUI, 'main.pdf'), 'relatorio.pdf')
    # fonte do relatorio
    for f in ('main.tex', 'numeros.tex', 'tab_aeb.tex', 'figuras_relatorio.py', 'numeros.txt'):
        add(z, os.path.join(AQUI, f), f'relatorio_fonte/{f}')
    for sub in ('secoes', 'imgs', 'logs'):
        for raiz, _, arqs in os.walk(os.path.join(AQUI, sub)):
            for a in arqs:
                p = os.path.join(raiz, a)
                add(z, p, 'relatorio_fonte/' + os.path.relpath(p, AQUI).replace(os.sep, '/'))
    # codigo
    for pasta, destino in (('simulador', 'codigo/simulador'), ('veiculo_real', 'codigo/veiculo_real')):
        base = os.path.join(REPO, pasta)
        for raiz, dirs, arqs in os.walk(base):
            dirs[:] = [d for d in dirs if d not in PULA_DIRS or (pasta == 'simulador' and d == 'logs')]
            for a in arqs:
                if a.endswith(('.pyc', '.zip', '.pt', '.png', '.pdf', '.gz', '.aux', '.log', '.out', '.synctex')):
                    continue
                if a.endswith('.md') and a not in ('IDENTIFICACAO.md', 'CONTROLE_LONGITUDINAL.md'):
                    continue
                p = os.path.join(raiz, a)
                add(z, p, destino + '/' + os.path.relpath(p, base).replace(os.sep, '/'))
print(OUT, round(os.path.getsize(OUT) / 1e6, 1), 'MB')
