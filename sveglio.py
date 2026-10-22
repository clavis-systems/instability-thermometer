"""Tiene sveglio il PC (niente sospensione per inattività) solo mentre gira il comando dato, poi lo lascia.

Uso: python sveglio.py COMANDO [ARGOMENTI...]
Usa la richiesta di Windows SetThreadExecutionState: non cambia nessuna impostazione del sistema.
"""

import ctypes
import subprocess
import sys

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001

ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
try:
    codice = subprocess.call(sys.argv[1:])
finally:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
sys.exit(codice)
