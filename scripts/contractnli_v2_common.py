"""Separate v2 paths; retain the exact v1 task prompt, schema and span mapping."""
from contractnli_common import *

NAME='contractnli-v2'
DATA=ROOT/'data/curated'/NAME
EVAL=ROOT/'data/experiments'/NAME
REPORT=ROOT/'reports'/NAME
ARM='expanse8b_contractnli_v2_4bit'
OLD_ADAPTER_SHA='2a1c24c8ffed30151a4cab3c923f751390a62e9803ccdc16da802af3942a920a'
