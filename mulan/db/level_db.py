#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

from pathlib import Path
from typing import Any, Callable

import plyvel


class LevelDB:

    _instances = {}

    @staticmethod
    def get_instance(path: Path, *args, **kwargs):

        if path not in LevelDB._instances:
            LevelDB._instances[path] = LevelDB(path, *args, **kwargs)

        return LevelDB._instances[path]

    def __init__(self, path: Path,
                 key_transform: Callable[[Any], bytes],
                 key_back_transform: Callable[[bytes], Any],
                 value_transform: Callable[[Any], bytes],
                 value_back_transform: Callable[[bytes], Any]):

        self.path = path

        self.key_transform = key_transform
        self.key_back_transform = key_back_transform
        self.value_transform = value_transform
        self.value_back_transform = value_back_transform

        self.db = plyvel.DB(path.__str__(), create_if_missing=True)

    def __setitem__(self, key: Any, value: Any):
        key = self.key_transform(key)
        value = self.value_transform(value)
        return self.db.put(key, value)

    def __getitem__(self, key: Any):
        key = self.key_transform(key)
        value = self.db.get(key)
        value = self.value_back_transform(value)
        return value


if __name__ == '__main__':

    db = LevelDB.get_instance(
        Path('/tmp/leveldb-example'),
        key_transform=lambda x: x.encode(),
        key_back_transform=lambda x: x.decode(),
        value_transform=lambda x: x.encode(),
        value_back_transform=lambda x: x.decode()
    )

    sentence = 'My name is Robin Hood'.split(' ')
    sentence_id = 1

    word = 'Robin'
    word_index = 3

    # save sentence
    db[str(sentence_id)] = ' '.join(sentence)

    # fetch sentence
    sentence = db[str(sentence_id)]
    print(f'# fetched sentence: {sentence}')

    # fetch word in sentence
    word = sentence.split(' ')[word_index]
    print(f'# fetched word in sentence: {word}')

