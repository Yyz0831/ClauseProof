"""Isolated v3 paths; task prompt, rendering and parser remain unchanged."""
from contractnli_common import *

NAME = 'contractnli-v3'
DATA = ROOT / 'data/curated' / NAME
EVAL = ROOT / 'data/experiments' / NAME
REPORT = ROOT / 'reports' / NAME
ARM = 'expanse8b_contractnli_v3_4bit'
V1_ADAPTER_SHA = '2a1c24c8ffed30151a4cab3c923f751390a62e9803ccdc16da802af3942a920a'
V2_ADAPTER_SHA = '9ca2062d142aefbaee856056e8ead60033a5954f670e1e7d69ff33bd6b2445b0'
