#
# Copyright 2020 SapienzaNLP research group (http://nlp.uniroma1.it/)
# Authors: Edoardo Barba, Luigi Procopio, Niccolò Campolungo, Tommaso Pasini, Roberto Navigli
# GitHub Repo: https://github.com/SapienzaNLP/mulan
# License: Attribution-NonCommercial-ShareAlike 4.0 International
#

import hashlib
import os
import pickle
import xml.etree.cElementTree as ET
from collections import namedtuple
from typing import List, Optional, Iterable, Callable, Tuple, Dict, Set

from tqdm import tqdm
from xml.dom import minidom

from utils.nlp.kb import wn_id2bn_id, to_bn_id


WSDInstance = namedtuple('WSDInstance', ['text', 'lemma', 'pos', 'labels'])


def read_lemma2synsets(file_path: str) -> Dict[str, Set[str]]:

    cache_path = f'/tmp/{hashlib.md5(file_path.encode()).hexdigest()[:6]}.cache'

    if not os.path.exists(cache_path):

        xml_path = file_path.replace('.gold.key.txt', '.data.xml')
        key_path = file_path

        lemma2synsets = {}

        for _, _, sentence in tqdm(read_from_raganato(xml_path, key_path), desc="Reading XML"):
            for instance in sentence:
                if instance.labels is not None:
                    if instance.lemma not in lemma2synsets:
                        lemma2synsets[instance.lemma.lower()] = set()
                    for label in instance.labels:
                        label = to_bn_id(label)
                        lemma2synsets[instance.lemma.lower()].add(label)

        with open(cache_path, 'wb') as f:
            pickle.dump(lemma2synsets, f)

    with open(cache_path, 'rb') as f:
        return pickle.load(f)


def read_from_raganato(xml_path: str, key_path: str, instance_transform: Optional[Callable[[WSDInstance], WSDInstance]] = None) -> Iterable[Tuple[str, str, List[WSDInstance]]]:

    def read_by_text_iter(xml_path: str):

        it = ET.iterparse(xml_path, events=('start', 'end'))
        _, root = next(it)

        for event, elem in it:
            if event == 'end' and elem.tag == 'text':
                document_id = elem.attrib['id']
                for sentence in elem:
                    sentence_id = sentence.attrib['id']
                    for word in sentence:
                        yield document_id, sentence_id, word

            root.clear()

    mapping = {}

    with open(key_path) as f:
        for line in f:
            line = line.strip()
            instance, *labels = line.split(' ')
            mapping[instance] = labels

    last_seen_document_id = None
    last_seen_sentence_id = None

    for document_id, sentence_id, element in read_by_text_iter(xml_path):

        if last_seen_sentence_id != sentence_id:

            if last_seen_sentence_id is not None:
                yield last_seen_document_id, last_seen_sentence_id, sentence

            sentence = []
            last_seen_document_id = document_id
            last_seen_sentence_id = sentence_id

        instance = WSDInstance(
            text=element.text,
            pos=element.attrib.get('pos', None),
            lemma=element.attrib.get('lemma', None),
            labels=None if element.tag == 'wf' or element.attrib['id'] not in mapping else mapping[element.attrib['id']]
        )

        if instance_transform is not None:
            instance = instance_transform(instance)

        sentence.append(instance)

    yield last_seen_document_id, last_seen_sentence_id, sentence


class RaganatoBuilder:

    def __init__(self, lang: str, source: str):
        self.corpus = ET.Element('corpus')
        self.corpus.set('lang', lang)
        self.corpus.set('source', source)
        self.current_text_section = None
        self.current_sentence_section = None
        self.gold_senses = []

    def open_text_section(self, text_id: str, text_source: str = None):
        text_section = ET.SubElement(self.corpus, 'text')
        text_section.set('id', text_id)
        if text_source is not None:
            text_section.set('source', text_source)
        self.current_text_section = text_section

    def open_sentence_section(self, sentence_id: str):
        sentence_section = ET.SubElement(self.current_text_section, 'sentence')
        sentence_id = self.compute_id([self.current_text_section.attrib['id'], sentence_id])
        sentence_section.set('id', sentence_id)
        self.current_sentence_section = sentence_section

    def add_annotated_token(self, token: str, lemma: str, pos: str, instance_id: Optional[str] = None, sense: Optional[str] = None):
        if instance_id is not None and sense is not None:
            token_element = ET.SubElement(self.current_sentence_section, 'instance')
            token_id = self.compute_id([self.current_sentence_section.attrib['id'], instance_id])
            token_element.set('id', token_id)
            self.gold_senses.append((token_id, sense))
        else:
            token_element = ET.SubElement(self.current_sentence_section, 'wf')
        token_element.set('lemma', lemma)
        token_element.set('pos', pos)
        token_element.text = token

    @staticmethod
    def compute_id(chain_ids: List[str]) -> str:
        return '.'.join(chain_ids)

    def store(self, data_output_path: str, labels_output_path: str):
        self.__store_xml(data_output_path)
        self.__store_labels(labels_output_path)

    def __store_xml(self, output_path: str):
        corpus_writer = ET.ElementTree(self.corpus)
        with open(output_path, 'wb') as f_xml:
            corpus_writer.write(f_xml, encoding='UTF-8', xml_declaration=True)
        dom = minidom.parse(output_path)
        pretty_xml = dom.toprettyxml()
        with open(output_path, 'w') as f_xml:
            f_xml.write(pretty_xml)

    def __store_labels(self, output_path: str):
        with open(output_path, 'w') as f_labels:
            for gold_sense in self.gold_senses:
                f_labels.write(' '.join(gold_sense))
                f_labels.write('\n')
