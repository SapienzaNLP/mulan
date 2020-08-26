# MuLaN: Multilingual Label propagatioN for Word Sense Disambiguation
MuLaN ([Multilingual Label propagatioN](https://www.ijcai.org/Proceedings/2020/0531.pdf), IJCAI 2020) is a label propagation technique tailored to WSD and capable of automatically producing sense-tagged training datasets in multiple languages. Simply put, by jointly leveraging contextualized word embeddings and the multilingual information enclosed in knowledge bases, MuLaN projects sense information from a source tagged corpus in language L<sub>1</sub> towards a target unlabelled one in language L<sub>2</sub>, possibly different from L<sub>1</sub>. 

If you find either our code or our release datasets useful in your work, please cite us with:
```
@inproceedings{ijcai2020-531,
  title     = {Mu{L}a{N}: Multilingual Label propagatio{N} for Word Sense Disambiguation},
  author    = {Barba, Edoardo and Procopio, Luigi and Campolungo, Niccolò and Pasini, Tommaso and Navigli, Roberto},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  pages     = {3837--3844},
  year      = {2020},
  month     = {7},
  doi       = {10.24963/ijcai.2020/531},
  url       = {https://doi.org/10.24963/ijcai.2020/531},
}
```
## Released Datasets
We release the datasets we referenced within our paper, providing silver-tagged data in German, Spanish, French and Italian:

|    Dataset   | # sentences | # instances | # distinct lemmas | # distinct synsets | # distinct senses | % of transferred synsets |
|:------------:|:-----------:|:-----------:|:-----------------:|:------------------:|:-----------------:|:------------------------:|
| SemCor + WNG |    154835   |    722812   |       59073       |        69404       |       91274       |           100.0          |
|   [mulan-de](https://drive.google.com/file/d/1_vQX0oUYFIyv58e6tbDuin4EtxpKABj-/view?usp=sharing)   |    207801   |    245173   |       19108       |        18676       |       21776       |           26.91          |
|   [mulan-es](https://drive.google.com/file/d/1bTayDizd-1HO3KYokPByck0yIJX2Uape/view?usp=sharing)   |    262391   |    452584   |       30383       |        42618       |       56252       |           61.41          |
|   [mulan-fr](https://drive.google.com/file/d/1BL9esH2iKEVOx-xmvcLLsJPP1gQ6gZ9G/view?usp=sharing)   |    228757   |    310756   |       21218       |        24600       |       28701       |           35.44          |
|   [mulan-it](https://drive.google.com/file/d/1L_EidyONEaX_7sJXozIaN6OcY5_T5Sl1/view?usp=sharing)   |    279320   |    415761   |       28244       |        32489       |       43559       |           46.81          |
## Generating a new corpus
We also release our transferring code, thus allowing to generate new tagged corpora.

### Environment Setup
1. Install [conda](https://docs.conda.io/en/latest/)
2. Run *setup.sh* to setup the environment
    ```
    bash setup.sh
    ```
3. Setup data folder so that it looks like the following:
    ``` bash
    $ tree -L 2 ~/mulan/data
    ├── bn2wn.txt
    ├── mapped-datasets
    │   ├── sample-source
    │   └── sample-target
    └── wsd-datasets
        ├── SemCor
        └── WNGT
    ```
   That is, you have to create the *bn2wn.txt* file. In order to achieve this, you may use the code and instructions in [SapienzaNLP/mwsd-datasets](https://github.com/SapienzaNLP/mwsd-datasets): the section *BabelNet To WordNet mapping* produces exactly this file. The file should look the following:
    ``` bash
    <babelnet-id> <\t> <first-associated-wordnet-id> <\t> <second-associated-wordnet-id> ...
    ```
4. Setup the vocabs folder so that it looks like the following:
    ``` bash
    $ tree -L 1 ~/mulan/vocabs
    ├── ...
    ├── lemma2synsets.<desired-languge>.txt
    └── ...
    ```
   Once again, you may use the code and instructions in [SapienzaNLP/mwsd-datasets](https://github.com/SapienzaNLP/mwsd-datasets) in order to generate the mappings from lemmas to the possible BabelNet synsets in the desired languages (section *Build the Inventory*): take the file *inventory.<language>.withgold.txt* (we suggest sticking to the WordNet subgraph and using the *-s wn* option for most cases), rename it to *lemma2synsets.<desired-languge>.txt* and place it in the *vocabs/* folder. This file should look like the following:
    ``` bash
    <lemma>#<pos> <\t> <first-associated-babelnet-id> <\t> <second-associated-babelnet-id> ...
    ```

### Projection Flow
All the transfering code is organized using a "gun firing" analogy. This metaphor was not meant to stick around; rather, 
it simply made easier talking and reasoning about the various steps. However, we ended up getting attached to it:
1. **load**: MuLaN takes as input a source and a target corpus, vectorizes them and stores them in the *vectorization/*
   folder (section 3.1 in the paper)
2. **spot targets**: MuLaN computes, for each annotated instance in the source corpus, a list of possible projections
   onto the target one. These *transfer coordinates* are stored in the *coordinates/* folder (section 3.2)
3. **aim** (or, informally, compute firing priorities): MuLaN filters the proposed transfers (with a backward check) and
   assigns a globally consistent score to each (x, y), where x is an annotated source instance and y a target instance
   MuLaN proposed transferring x upon. The output of this step is stored inside the *coordinates/* folder (first part of section 3.3)
4. **fire**: MuLaN uses the coordinates file produced at the previous step to automatically generate a new annotated
   corpus (second part of section 3.3)

### Folder Structure
``` bash
$ tree -L 1 ~/mulan
├── mulan
├── cache
├── coordinates
├── data
├── README.md
├── transfer
├── vectorization
└── vocabs
```
* **mulan**: code directory
* **cache**: where some intermediate data structures (i.e. LevelDB databases) are saved
* **coordinates**: where coordinates are saved
* **data**: datasets and mappings
* **transfer**: where transfer results are saved
* **vectorization**: where vectorization results are saved
* **vocab**: where vocabs are saved

### Input Data Preparation
MuLaN's pipeline takes as input 2 corpora (a source and a target one), specified through the Corpus enum in *mulan/corpora.py* 
and with the associated data stored in *data/mapped-datasets* in our predefined input format:
``` bash
$ head -1 ~/mulan/data/mapped-datasets/sample-source/data.txt 
d1.s1 <\t> I I PRON X <\t> have have VERB X <\t> a a DET X <\t> dog dog NOUN bn:00015267n
$ head -1 ~/mulan/data/mapped-datasets/sample-target/0.txt
sample:1.1 <\t> Io io PRON X <\t> ho avere VERB X <\t> un un DET X <\t> cane cane NOUN X
```
MuLaN only accepts input in such format. Most likely, you'll need to preprocess your data into this structure; to pos-lemma tag the file, we suggest using [Stanza](https://stanfordnlp.github.io/stanza/).

### Output Format
As the output format, we follow the scheme introduced in [SemEval 2013 task 12](https://www.aclweb.org/anthology/S13-2040.pdf)
and later chosen as the standard WSD format for the evaluation framework presented in [Raganato et al, 2017](https://www.aclweb.org/anthology/E17-1010.pdf).

A dataset consists of 2 files:
* a .xml file, storing the actual sentences and marking tagged instances (i.e. tokens) with an id
* a .txt file, mapping instances ids with the actual labels (synsets in our case)

To make an example:

``` bash
$ tree -L 1 ~/mulan/transfer/SAMPLE_SOURCE_MBERT-SAMPLE_TARGET_MBERT/
├── transfer.data.xml
└── transfer.gold.key.txt
```
### Running the projection code
You can create a new corpus either using our fine-grained scripts:

``` bash
PYTHONPATH=mulan/ python mulan/transfer/1_retrieve_targets_manifesto.py --language <language>
PYTHONPATH=mulan/ python mulan/transfer/2_load.py <source-corpus-enum>
PYTHONPATH=mulan/ python mulan/transfer/2_load.py <target-corpus-enum>
PYTHONPATH=mulan/ python mulan/transfer/3_spot_targets.py --source-enum <source-corpus-enum> --target-enum <target-corpus-enum> --coordinates-folder <output-coordinates-folder>
PYTHONPATH=mulan/ python mulan/transfer/4_compute_priorities.py --source-enum <semcor-enum> --target-enum <target-corpus-enum> --coordinates-folder <output-coordinates-folder>
PYTHONPATH=mulan/ python mulan/transfer/5_word_fire.py --language <language> --name <name> --coordinates <coordinates-path>,<source-corpus-enum>,<target-corpus-enum> --coordinates <coordinates-path>,<source-corpus-enum>,<target-corpus-enum> --output-folder <output-folder>
```
or with a simpler:
``` bash
bash pipeline.sh <target-language> <source-corpus-enum> <target-corpus-enum>
```
It may be necessary to manually edit the code in order to change some hyperparameters (i.e. the encoder). We plan on moving
our code to an AllenNLP-like structure, with json configurations being given as input, so as to better support hyperparameters
modifications; however, we do not have an ETA for this yet.

### Example
We provide a simple example you may use to better understand the intermediate stages (and, obviously, to check that everything is working correctly):
* **source** (located at *data/mapped-datasets/sample-source*): *I have a (dog, [bn:00015267n](https://babelnet.org/synset?word=bn%3A00015267n&lang=EN))*
* **target** (located at *data/mapped-datasets/sample-target*): *Io ho un cane*
Running:
``` bash
bash pipeline.sh it SAMPLE_SOURCE_MBERT SAMPLE_TARGET_MBERT
```
will project the sense of *dog* in *I have a dog* towards *cane* towards *cane* in *Io ho un cane*, producing:
* **intermediate files** in *vectorization* and *coordinates*

* **final result** in *transfer*
