#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

from pathlib import Path

# PROJECT_BASE_FOLDER = Path('/home/luigi/Desktop/AncorBERT/annotation-transfer')
PROJECT_BASE_FOLDER = Path('')

DATA_FOLDER = PROJECT_BASE_FOLDER / 'data'
CORPORA_FOLDER = DATA_FOLDER / 'mapped-datasets'

VECTORIZATION_FOLDER = PROJECT_BASE_FOLDER / 'vectorization'

WSD_DATASET_FOLDER = Path('data/wsd-datasets')
