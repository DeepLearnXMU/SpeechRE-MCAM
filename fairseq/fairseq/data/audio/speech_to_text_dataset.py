# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import io
import logging
import os.path as op
import pdb
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from fairseq.data import (
    ConcatDataset,
    Dictionary,
    FairseqDataset,
    ResamplingDataset,
    data_utils as fairseq_data_utils,
)
from fairseq.data.audio.audio_utils import get_fbank, get_waveform
from fairseq.data.audio.feature_transforms import CompositeAudioFeatureTransform
# Prefix="/workdir/liangzhang/SpeechRE/IWSLT"
Prefix = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT"

logger = logging.getLogger(__name__)


class S2TDataConfig(object):
    """Wrapper class for data config YAML"""

    def __init__(self, yaml_path):
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load YAML files for " "S2T data config")
        self.config = {}
        if op.isfile(yaml_path):
            try:
                with open(yaml_path) as f:
                    self.config = yaml.load(f, Loader=yaml.FullLoader)
            except Exception as e:
                logger.info(f"Failed to load config from {yaml_path}: {e}")
        else:
            logger.info(f"Cannot find {yaml_path}")

    @property
    def vocab_filename(self):
        """fairseq vocabulary file under data root"""
        return self.config.get("vocab_filename", "dict.txt")

    @property
    def shuffle(self) -> bool:
        """Shuffle dataset samples before batching"""
        return self.config.get("shuffle", False)

    @property
    def pre_tokenizer(self) -> Dict:
        """Pre-tokenizer to apply before subword tokenization. Returning
        a dictionary with `tokenizer` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        return self.config.get("pre_tokenizer", {"tokenizer": None})

    @property
    def bpe_tokenizer(self) -> Dict:
        """Subword tokenizer to apply after pre-tokenization. Returning
        a dictionary with `bpe` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        return self.config.get("bpe_tokenizer", {"bpe": None})

    @property
    def prepend_tgt_lang_tag(self) -> bool:
        """Prepend target lang ID token as the target BOS (e.g. for to-many
        multilingual setting). During inference, this requires `--prefix-size 1`
        to force BOS to be lang ID token."""
        return self.config.get("prepend_tgt_lang_tag", False)

    @property
    def input_feat_per_channel(self):
        """The dimension of input features (per audio channel)"""
        return self.config.get("input_feat_per_channel", 80)

    @property
    def input_channels(self):
        """The number of channels in the input audio"""
        return self.config.get("input_channels", 1)

    @property
    def sampling_alpha(self):
        """Hyper-parameter alpha = 1/T for temperature-based resampling.
        (alpha = 1 for no resampling)"""
        return self.config.get("sampling_alpha", 1.0)

    @property
    def use_audio_input(self):
        """Needed by the dataset loader to see if the model requires
        raw audio as inputs."""
        return self.config.get("use_audio_input", False)

    @property
    def audio_root(self):
        """Audio paths in the manifest TSV can be relative and this provides
        the root path. Set this to empty string when using absolute paths."""
        return self.config.get("audio_root", "")

    def get_feature_transforms(self, split, is_train):
        """Split-specific feature transforms. Allowing train set wildcard `_train`,
        evaluation set wildcard `_eval` and general wildcard `*` for matching."""
        from copy import deepcopy

        cfg = deepcopy(self.config)
        _cur = cfg.get("transforms", {})
        cur = _cur.get(split)
        cur = _cur.get("_train") if cur is None and is_train else cur
        cur = _cur.get("_eval") if cur is None and not is_train else cur
        cur = _cur.get("*") if cur is None else cur
        cfg["transforms"] = cur
        return cfg


def is_npy_data(data: bytes) -> bool:
    return data[0] == 147 and data[1] == 78


def is_flac_or_wav_data(data: bytes) -> bool:
    is_flac = data[0] == 102 and data[1] == 76
    is_wav = data[0] == 82 and data[1] == 73
    return is_flac or is_wav


def read_from_uncompressed_zip(file_path, offset, file_size) -> bytes:
    with open(file_path, "rb") as f:
        f.seek(offset)
        data = f.read(file_size)
    return data


def get_features_from_npy_or_audio(path):
    ext = op.splitext(op.basename(path))[1]
    if ext not in {".npy", ".flac", ".wav"}:
        raise ValueError(f'Unsupported file format for "{path}"')
    return np.load(path) if ext == ".npy" else get_fbank(path)


def get_features_or_waveform_from_uncompressed_zip(
    path, byte_offset, byte_size, need_waveform=False
):
    assert path.endswith(".zip")
    data = read_from_uncompressed_zip(path, byte_offset, byte_size)
    f = io.BytesIO(data)
    if is_npy_data(data):
        features_or_waveform = np.load(f)
    elif is_flac_or_wav_data(data):
        features_or_waveform = \
            get_waveform(f, always_2d=False)[0] if need_waveform else get_fbank(f)
    else:
        raise ValueError(f'Unknown file format for "{path}"')
    return features_or_waveform


def get_features_or_waveform(path: str, need_waveform=False):
    """Get speech features from .npy file or waveform from .wav/.flac file.
    The file may be inside an uncompressed ZIP file and is accessed via byte
    offset and length.

    Args:
        path (str): File path in the format of "<.npy/.wav/.flac path>" or
        "<zip path>:<byte offset>:<byte length>".
        need_waveform (bool): return waveform instead of features.

    Returns:
        features_or_waveform (numpy.ndarray): speech features or waveform.
    """
    _path, *extra = path.split(":")
    if not op.exists(_path):
        raise FileNotFoundError(f"File not found: {_path}")
    if len(extra) == 0:
        if need_waveform:
            waveform, _ = get_waveform(_path, always_2d=False)
            return waveform
        return get_features_from_npy_or_audio(_path)
    elif len(extra) == 2:
        extra = [int(i) for i in extra]
        if need_waveform:
            waveform, _ = get_waveform(_path, start=extra[0], frames=extra[1], always_2d=False)
            return waveform
        features_or_waveform = get_features_or_waveform_from_uncompressed_zip(
            _path, extra[0], extra[1], need_waveform=need_waveform
        )
    else:
        raise ValueError(f"Invalid path: {path}")
    return features_or_waveform


def _collate_frames(
    frames: List[torch.Tensor], is_audio_input: bool = False
) -> torch.Tensor:
    """
    Convert a list of 2D frames into a padded 3D tensor
    Args:
        frames (list): list of 2D frames of size L[i]*f_dim. Where L[i] is
            length of i-th frame and f_dim is static dimension of features
    Returns:
        3D tensor of size len(frames)*len_max*f_dim where len_max is max of L[i]
    """
    max_len = max(frame.size(0) for frame in frames)
    if is_audio_input:
        out = frames[0].new_zeros((len(frames), max_len))
    else:
        out = frames[0].new_zeros((len(frames), max_len, frames[0].size(1)))
    for i, v in enumerate(frames):
        out[i, : v.size(0)] = v
    return out

# 读取音频数据拼接对目标句子和原句子进行分词。
class SpeechToTextDataset(FairseqDataset):
    LANG_TAG_TEMPLATE = "<lang:{}>"

    def __init__(
        self,
        split: str,
        is_train_split: bool,
        data_cfg: S2TDataConfig,
        audio_paths: List[str],
        n_frames: List[int],
        src_texts: Optional[List[str]] = None,
        tgt_texts: Optional[List[str]] = None,
        speakers: Optional[List[str]] = None,
        src_langs: Optional[List[str]] = None,
        tgt_langs: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        tgt_dict: Optional[Dictionary] = None,
        pre_tokenizer=None,
        bpe_tokenizer=None,
    ):
        self.split, self.is_train_split = split, is_train_split
        self.data_cfg = data_cfg
        self.audio_paths, self.n_frames = audio_paths, n_frames
        self.n_samples = len(audio_paths)
        assert len(n_frames) == self.n_samples > 0
        assert src_texts is None or len(src_texts) == self.n_samples
        assert tgt_texts is None or len(tgt_texts) == self.n_samples
        assert speakers is None or len(speakers) == self.n_samples
        assert src_langs is None or len(src_langs) == self.n_samples
        assert tgt_langs is None or len(tgt_langs) == self.n_samples
        assert ids is None or len(ids) == self.n_samples
        assert (tgt_dict is None and tgt_texts is None) or (
            tgt_dict is not None and tgt_texts is not None
        )
        self.src_texts, self.tgt_texts = src_texts, tgt_texts
        self.src_langs, self.tgt_langs = src_langs, tgt_langs
        self.tgt_dict = tgt_dict
        self.check_tgt_lang_tag()
        self.ids = ids
        self.shuffle = data_cfg.shuffle if is_train_split else False

        self.feature_transforms = CompositeAudioFeatureTransform.from_config_dict(
            self.data_cfg.get_feature_transforms(split, is_train_split)
        )

        self.pre_tokenizer = pre_tokenizer
        self.bpe_tokenizer = bpe_tokenizer

        # 直接定义分词器
        from transformers import BartTokenizer
        self.tokenizer = BartTokenizer.from_pretrained(Prefix + "/Data/bartre")
        # self.tokenizer.add_special_tokens({"additional_special_tokens": ["<triplet>", "<subj>", "<obj>"]})
        # self.src_text = self.get_src_text()
        logger.info(self.__repr__())

    def get_src_text(self):
        import json
        src_text = dict()
        if self.split.startswith("train"):
            with open("/data/wangguitao/TTS/Mozilla/data/CONLL04/conll04_train.json")as f:
                lines = json.load(f)
                for id, line in enumerate(lines):
                    src_text[id] = " ".join(line["tokens"])
        elif self.split.startswith("dev"):
            with open("/data/wangguitao/TTS/Mozilla/data/CONLL04/conll04_dev.json")as f:
                lines = json.load(f)
                for id, line in enumerate(lines):
                    src_text[id] = " ".join(line["tokens"])
        elif self.split.startswith("test"):
            with open("/data/wangguitao/TTS/Mozilla/data/CONLL04/conll04_test.json")as f:
                lines = json.load(f)
                for id, line in enumerate(lines):
                    src_text[id] = " ".join(line["tokens"])
        else:
            print("Dataset Error!")
        return src_text

    def __repr__(self):
        return (
            self.__class__.__name__
            + f'(split="{self.split}", n_samples={self.n_samples}, '
            f"prepend_tgt_lang_tag={self.data_cfg.prepend_tgt_lang_tag}, "
            f"shuffle={self.shuffle}, transforms={self.feature_transforms})"
        )

    @classmethod
    def is_lang_tag(cls, token):
        pattern = cls.LANG_TAG_TEMPLATE.replace("{}", "(.*)")
        return re.match(pattern, token)

    def check_tgt_lang_tag(self):
        if self.data_cfg.prepend_tgt_lang_tag:
            assert self.tgt_langs is not None and self.tgt_dict is not None
            tgt_lang_tags = [
                self.LANG_TAG_TEMPLATE.format(t) for t in set(self.tgt_langs)
            ]
            assert all(t in self.tgt_dict for t in tgt_lang_tags)

    def tokenize_text(self, text: str):
        # print(text)
        if self.pre_tokenizer is not None:
            text = self.pre_tokenizer.encode(text)
        if self.bpe_tokenizer is not None:
            text = self.bpe_tokenizer.encode(text)
        # if self.tokenizer is not None:
        #     text = self.tokenizer.tokenize(text)
        #     text = " ".join(text)
        # print(text)
        return text

    # 真正处理数据的函数
    def __getitem__(
        self, index: int
    ) -> Tuple[int, torch.Tensor, Optional[torch.Tensor]]:
        # 读取音频向量
        source = get_features_or_waveform(self.audio_paths[index], need_waveform=self.data_cfg.use_audio_input)
        if self.feature_transforms is not None:
            assert not self.data_cfg.use_audio_input
            source = self.feature_transforms(source)
        source = torch.from_numpy(source).float()

        # 分词三元组序列
        target = None
        if self.tgt_texts is not None:
            target = self.tokenize_text(self.tgt_texts[index])
            # pdb.set_trace()
            # target = torch.IntTensor(target).long()
            # text = self.tokenizer.tokenize(self.tgt_texts[index])
            # text = " ".join(text) + " </s>"
            # target1 = self.tokenizer.convert_tokens_to_ids(text.split(" "))
            src_get =  self.tokenize_text(self.src_texts[index])
            # src_get = torch.IntTensor(src_get).long()

            # target = self.tgt_dict.encode_line(
            #     tokenized, add_if_not_exist=True, append_eos=True
            # ).long()
            # breakpoint()
            if self.data_cfg.prepend_tgt_lang_tag:
                lang_tag = self.LANG_TAG_TEMPLATE.format(self.tgt_langs[index])
                lang_tag_idx = self.tgt_dict.index(lang_tag)
                target = torch.cat((torch.LongTensor([lang_tag_idx]), target), 0)

        return index, source, target, src_get

    def __len__(self):
        return self.n_samples

    # 是数据控制处理器
    def collater(self, samples: List[Tuple[int, torch.Tensor, torch.Tensor]]) -> Dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([i for i, _, _, _ in samples], dtype=torch.long)
        # 将帧填充到最大长度
        frames = _collate_frames([s for _, s, _, _ in samples], self.data_cfg.use_audio_input)
        # sort samples by descending number of frames
        n_frames = torch.tensor([s.size(0) for _, s, _, _ in samples], dtype=torch.long)

        n_frames, order = n_frames.sort(descending=True)  #把从小到大变成从大到小到排序
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        target, target_lengths = None, None
        prev_output_tokens = None
        ntokens = None
        if self.tgt_texts is not None:
            # 训练解码结果
            target = fairseq_data_utils.collate_tokens(
                [t for _, _, t, _ in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            target = target.index_select(0, order)
            target_lengths = torch.tensor(
                [t.size(0) for _, _, t, _ in samples], dtype=torch.long
            ).index_select(0, order)
            # 训练输入
            prev_output_tokens = fairseq_data_utils.collate_tokens(
                [t for _, _, t, _ in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, order)
            ntokens = sum(t.size(0) for _, _, t, _ in samples)

        srcget, srcget_lengths = None, None
        if self.src_texts is not None:
            # 训练解码结果
            srcget = fairseq_data_utils.collate_tokens(
                [s for _, _, _, s in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            srcget = srcget.index_select(0, order)
            srcget_lengths = torch.tensor(
                [s.size(0) for _, _, _, s in samples], dtype=torch.long
            ).index_select(0, order)

        out = {
            "id": indices,
            "net_input": {
                "src_tokens": frames,
                "src_lengths": n_frames,
                "prev_output_tokens": prev_output_tokens,
                # "src_text": [self.src_text[int(indice)] for indice in indices],
                "src_text": srcget,
                "src_text_lengths": srcget_lengths
            },
            "target": target,
            "target_lengths": target_lengths,
            "ntokens": ntokens,
            "nsentences": len(samples),
        }
        return out

    def num_tokens(self, index):
        return self.n_frames[index]

    def size(self, index):
        t_len = 0
        if self.tgt_texts is not None:
            tokenized = self.tokenize_text(self.tgt_texts[index])
            # print(tokenized)
            # exit(0)
            t_len = len(tokenized)
            # t_len = len(tokenized.split(" "))
        return self.n_frames[index], t_len

    @property
    def sizes(self):
        return np.array(self.n_frames)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        # first by descending order of # of frames then by original/random order
        order.append([-n for n in self.n_frames])
        return np.lexsort(order)

    def prefetch(self, indices):
        raise False


class SpeechToTextDatasetCreator(object):
    # mandatory columns
    KEY_ID, KEY_AUDIO, KEY_N_FRAMES = "id", "audio", "n_frames"
    KEY_TGT_TEXT = "tgt_text"
    # optional columns
    KEY_SPEAKER, KEY_SRC_TEXT = "speaker", "src_text"
    KEY_SRC_LANG, KEY_TGT_LANG = "src_lang", "tgt_lang"
    # default values
    DEFAULT_SPEAKER = DEFAULT_SRC_TEXT = DEFAULT_LANG = ""

    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[List[Dict]],
        data_cfg: S2TDataConfig,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
    ) -> SpeechToTextDataset:
        # 将文件分解成多个list,SpeechToTextDataset考虑到几个字段
        audio_paths, n_frames, src_texts, tgt_texts, ids = [], [], [], [], []
        speakers, src_langs, tgt_langs = [], [], []

        for s in samples:
            print("\n\n================================")
            print(s[0])
            print("================================\n\n")
            ids.extend([ss[cls.KEY_ID] for ss in s])
            audio_paths.extend(
                [op.join(data_cfg.audio_root, ss[cls.KEY_AUDIO]) for ss in s]
            )
            # [ss['n_frames'] for ss in s]
            # pdb.set_trace()
            n_frames.extend([int(ss[cls.KEY_N_FRAMES]) for ss in s])
            tgt_texts.extend([ss[cls.KEY_TGT_TEXT] for ss in s])
            # 后面是可选字段
            src_texts.extend([ss.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for ss in s])
            speakers.extend([ss.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for ss in s])
            src_langs.extend([ss.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for ss in s])
            tgt_langs.extend([ss.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for ss in s])


        # 读取文件并生成数据迭代器
        return SpeechToTextDataset(
            split_name,
            is_train_split,
            data_cfg,
            audio_paths,
            n_frames,
            src_texts,
            tgt_texts,
            speakers,
            src_langs,
            tgt_langs,
            ids,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
        )

    @classmethod
    def _get_size_ratios(cls, ids: List[str], sizes: List[int], alpha: float = 1.0):
        """Size ratios for temperature-based sampling
        (https://arxiv.org/abs/1907.05019)"""
        _sizes = np.array(sizes)
        prob = _sizes / _sizes.sum()
        smoothed_prob = prob ** alpha
        smoothed_prob = smoothed_prob / smoothed_prob.sum()
        size_ratio = (smoothed_prob * _sizes.sum()) / _sizes

        o_str = str({_i: f"{prob[i]:.3f}" for i, _i in enumerate(ids)})
        logger.info(f"original sampling probability: {o_str}")
        p_str = str({_i: f"{smoothed_prob[i]:.3f}" for i, _i in enumerate(ids)})
        logger.info(f"balanced sampling probability: {p_str}")
        sr_str = str({_id: f"{size_ratio[i]:.3f}" for i, _id in enumerate(ids)})
        logger.info(f"balanced sampling size ratio: {sr_str}")
        return size_ratio.tolist()

    @classmethod
    def from_tsv(
        cls,
        root: str,
        data_cfg: S2TDataConfig,
        splits: str,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
        is_train_split: bool,
        epoch: int,
        seed: int,
    ) -> SpeechToTextDataset:
        samples = []
        _splits = splits.split(",") # 多个文件使用“,”分开
        # 是将每个数据文件的数据合并为：sampel=List[List[Dict]]  每个数据文件是一个list
        for split in _splits:
            tsv_path = op.join(root, f"{split}.tsv")
            if not op.isfile(tsv_path):
                raise FileNotFoundError(f"Dataset not found: {tsv_path}")
            with open(tsv_path) as f:
                reader = csv.DictReader(
                    f,
                    delimiter="\t",
                    quotechar=None,
                    doublequote=False,
                    lineterminator="\n",
                    quoting=csv.QUOTE_NONE,
                )
                samples.append([dict(e) for e in reader])
                assert len(samples) > 0

        # 读取数据变成可迭代对象
        datasets = [
            cls._from_list(
                name,
                is_train_split,
                [s],
                data_cfg,
                tgt_dict,
                pre_tokenizer,
                bpe_tokenizer,
            )
            for name, s in zip(_splits, samples)
        ]
        if is_train_split and len(_splits) > 1 and data_cfg.sampling_alpha != 1.0:
            # temperature-based sampling
            size_ratios = cls._get_size_ratios(
                _splits, [len(s) for s in samples], alpha=data_cfg.sampling_alpha
            )
            datasets = [
                ResamplingDataset(
                    d, size_ratio=r, seed=seed, epoch=epoch, replace=(r >= 1.0)
                )
                for d, r in zip(datasets, size_ratios)
            ]
        # ConcatDataset合并多个SpeechToTextDataset，并记录下来，然后提供一些操作函数
        return ConcatDataset(datasets)
