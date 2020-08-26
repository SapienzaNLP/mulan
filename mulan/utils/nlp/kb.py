#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

from nltk.corpus import wordnet as wn


def wn_sense_key_to_id(sense_key):
    synset = wn.lemma_from_key(sense_key).synset()
    return 'wn:' + str(synset.offset()).zfill(8) + synset.pos()


_wn2bn = {}
_bn2wn = {}


with open('data/bn2wn.txt') as f:
    for line in f:
        line = line.strip()
        bn_id, *wn_ids = line.split('\t')
        _bn2wn[bn_id] = wn_ids[0]  # todo currently considering only last
        for wn_id in wn_ids:
            _wn2bn[wn_id] = bn_id


def wn_id2bn_id(wn_id):
    return _wn2bn[wn_id]


def bn_id2wn_id(bn_id):
    return _bn2wn[bn_id]


_to_bn_id_cache = {}


def to_bn_id(key):

    if key.startswith('bn:'):
        key_type = 'bn_id'
        transform = lambda x: x
    elif key.startswith('wn:'):
        key_type = 'wn_id'
        transform = lambda x: wn_id2bn_id(x)
    else:
        key_type = 'sense_key'
        transform = lambda x: to_bn_id(wn_sense_key_to_id(x).replace('s', 'a'))

    if key_type not in _to_bn_id_cache:
        _to_bn_id_cache[key_type] = {}

    if key not in _to_bn_id_cache[key_type]:
        _to_bn_id_cache[key_type][key] = transform(key)

    return _to_bn_id_cache[key_type][key]
