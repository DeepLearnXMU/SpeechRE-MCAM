# SpeechRE  

TTS for Text-to-Speech;

IWSLT for SpeechRE, and the model is placed in fairseq_modules;

fairseq is a modified version in source code;

**Notice:**
> We implement our model in `Speech_RE/IWSLT/fairseq_modules/models/wav2triplet_s2t.py`

> Our loss function is implemented in `Speech_RE/fairseq/fairseq/criterions/label_smoothed_cross_entropy.py`

> Our cross-modal entity alignment method is implemented in `Speech_RE/fairseq/fairseq/transformer.py (Alignment_forward function)`

> Our configuration file is `Speech_RE/IWSLT/config/speechre_tacred_part_part.yaml`

> The training script for our model is placed in `Speech_RE/IWSLT/run_train.sh`

## Dataset
**Configuration information of the dataset synthesized by TTS:**

conll04.tgz：https://drive.google.com/file/d/1Q5k3eM6WknfjA2DWo19CyTwZngYVXRUL/view?usp=sharing

re-tacred(dev&test_part).tgz：https://drive.google.com/file/d/1qctG-n_W51zp-hiPDS-XEl7jh_bI1l_-/view?usp=sharing

re-tacred(train_part).tgz：https://drive.google.com/file/d/1ainRqlx4h9_HDFtOq8xasN-OLJDNSbwD/view?usp=sharing

<!-- 以conll04为例，数据集目录格式如下： -->
For example, the data of CoNLL04 is organized as:

```
├── conll04
│   ├── audio
│   │   ├── train
│   │   │   ├── train-0.wav
│   │   │   ├── train-1.wav
│   │   │   ├── train-2.wav
│   │   │   ├── ...
│   │   ├── dev
│   │   │   ├── dev-0.wav
│   │   │   ├── ...
│   │   ├── test
│   │   │   ├── test-0.wav
│   │   │   ├── ...
│   ├── train_conll04.tsv
│   ├── dev_conll04.tsv
│   ├── test_conll04.tsv
```

<!-- tsv文件格式如下： -->
The format of tsv files:

| id      | audio                                                      | duration_ms | n_frames | tgt_text                                                     | speaker | tgt_lang |
| ------- | ---------------------------------------------------------- | ----------- | -------- | ------------------------------------------------------------ | ------- | -------- |
| train-0 | /path/to/datasets/conll04/audio/train/train-0.wav:0:239828 | 14989       | 239828   | <triplet> Radio Reloj Network <subj> Havana <obj> OrgBased_In | 0       | en       |
| train-1 | /path/to/datasets/conll04/audio/train/train-1.wav:0:64099  | 4006        | 64099    | <triplet> Bruno Pusterla <subj> Italian Agricultural Confederation <obj> Work_For | 0       | en       |
| ...     |                                                            |             |          |                                                              |         |          |



**Notice:** The real dataset we constructed will be released soon.
