#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

import argparse
import os

from utils.nlp.kb import to_bn_id
from utils.nlp.wsd import read_from_raganato


def main(input_xml, input_key, output_folder, separator):

    assert not os.path.exists(output_folder)
    os.mkdir(output_folder)

    it = read_from_raganato(
        input_xml,
        input_key
    )

    with open(f'{output_folder}/data.txt', 'w') as f:

        for document_id, sentence_id, sentence in it:

            processed_sentence = []

            for instance in sentence:

                # discard those (they break BERT alginment)
                if instance.text == '``' or instance.text == '`':
                    continue

                if instance.labels is not None:
                    processed_sentence.append((instance.text.replace(' ', '_'), instance.lemma.replace(' ', '_'), instance.pos, ','.join([to_bn_id(label) for label in instance.labels])))
                else:
                    processed_sentence.append((instance.text.replace(' ', '_'), instance.lemma.replace(' ', '_'), instance.pos, 'X'))

            if len(processed_sentence) == 0:
                continue

            processed_sentence = [separator.join(e) for e in processed_sentence]
            processed_sentence = '\t'.join(processed_sentence)
            processed_sentence = f'{sentence_id}\t{processed_sentence}'

            f.write(f'{processed_sentence}\n')


def parse_args():
    parser = argparse.ArgumentParser(description='Load')
    parser.add_argument("xml_path", type=str, help='Path to xml raganato file')
    parser.add_argument("key_path", type=str, help='Path to key raganato file')
    parser.add_argument("-o", type=str, required=True, help='Path to output folder')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    main(
        args.xml_path,
        args.key_path,
        args.o,
        separator=' '
    )
