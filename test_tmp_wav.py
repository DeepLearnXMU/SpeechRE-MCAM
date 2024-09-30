from pathlib import Path
from typing import BinaryIO, Optional, Tuple, Union
import numpy as np
import torch
SF_AUDIO_FILE_EXTENSIONS = {".wav", ".flac", ".ogg"}


def _convert_to_mono(
        waveform: torch.FloatTensor, sample_rate: int
) -> torch.FloatTensor:
    if waveform.shape[0] > 1:
        try:
            import torchaudio.sox_effects as ta_sox
        except ImportError:
            raise ImportError(
                "Please install torchaudio to convert multi-channel audios"
            )
        effects = [['channels', '1']]
        return ta_sox.apply_effects_tensor(waveform, sample_rate, effects)[0]
    return waveform


def convert_to_mono(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    if waveform.shape[0] > 1:
        _waveform = torch.from_numpy(waveform)
        return _convert_to_mono(_waveform, sample_rate).numpy()
    return waveform


def get_waveform(
        path_or_fp: Union[str, BinaryIO], normalization=True, mono=True,
        frames=-1, start=0, always_2d=True
) -> Tuple[np.ndarray, int]:
    """Get the waveform and sample rate of a 16-bit WAV/FLAC/OGG Vorbis audio.

    Args:
        path_or_fp (str or BinaryIO): the path or file-like object
        normalization (bool): Normalize values to [-1, 1] (Default: True)
        mono (bool): convert multi-channel audio to mono-channel one
        frames (int): the number of frames to read. (-1 for reading all)
        start (int): Where to start reading. A negative value counts from the end.
        always_2d (bool): always return 2D array even for mono-channel audios
    Returns:
        waveform (numpy.ndarray): 1D or 2D waveform (channels x length)
        sample_rate (float): sample rate
    """
    if isinstance(path_or_fp, str):
        ext = Path(path_or_fp).suffix
        if ext not in SF_AUDIO_FILE_EXTENSIONS:
            raise ValueError(f"Unsupported audio format: {ext}")

    try:
        import soundfile as sf
    except ImportError:
        raise ImportError(
            "Please install soundfile to load WAV/FLAC/OGG Vorbis audios"
        )

    waveform, sample_rate = sf.read(
        path_or_fp, dtype="float32", always_2d=True, frames=frames, start=start
    )
    waveform = waveform.T  # T x C -> C x T
    if mono and waveform.shape[0] > 1:
        waveform = convert_to_mono(waveform, sample_rate)
    if not normalization:
        waveform *= 2 ** 15  # denormalized to 16-bit signed integers
    if not always_2d:
        waveform = waveform.squeeze(axis=0)
    return waveform, sample_rate


def _get_kaldi_fbank(
        waveform: np.ndarray, sample_rate: int, n_bins=80
) -> Optional[np.ndarray]:
    """Get mel-filter bank features via PyKaldi."""
    try:
        from kaldi.feat.mel import MelBanksOptions
        from kaldi.feat.fbank import FbankOptions, Fbank
        from kaldi.feat.window import FrameExtractionOptions
        from kaldi.matrix import Vector

        mel_opts = MelBanksOptions()
        mel_opts.num_bins = n_bins
        frame_opts = FrameExtractionOptions()
        frame_opts.samp_freq = sample_rate
        opts = FbankOptions()
        opts.mel_opts = mel_opts
        opts.frame_opts = frame_opts
        fbank = Fbank(opts=opts)
        features = fbank.compute(Vector(waveform.squeeze()), 1.0).numpy()
        return features
    except ImportError:
        return None


def _get_torchaudio_fbank(
        waveform: np.ndarray, sample_rate, n_bins=80
) -> Optional[np.ndarray]:
    """Get mel-filter bank features via TorchAudio."""
    try:
        import torchaudio.compliance.kaldi as ta_kaldi
        waveform = torch.from_numpy(waveform)
        features = ta_kaldi.fbank(
            waveform, num_mel_bins=n_bins, sample_frequency=sample_rate
        )
        return features.numpy()
    except ImportError:
        return None


def get_fbank(path_or_fp: Union[str, BinaryIO], n_bins=80) -> np.ndarray:
    """Get mel-filter bank features via PyKaldi or TorchAudio. Prefer PyKaldi
    (faster CPP implementation) to TorchAudio (Python implementation). Note that
    Kaldi/TorchAudio requires 16-bit signed integers as inputs and hence the
    waveform should not be normalized."""
    waveform, sample_rate = get_waveform(path_or_fp, normalization=False)

    features = _get_kaldi_fbank(waveform, sample_rate, n_bins)
    if features is None:
        features = _get_torchaudio_fbank(waveform, sample_rate, n_bins)
    if features is None:
        raise ImportError(
            "Please install pyKaldi or torchaudio to enable "
            "online filterbank feature extraction"
        )

    return features


import os
import pdb
import re
import csv
import json
import torch


tsv_path = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Data/train_raw.tsv"
with open(tsv_path) as f:
    reader = csv.DictReader(
        f,
        delimiter="\t",
        quotechar=None,
        doublequote=False,
        lineterminator="\n",
        quoting=csv.QUOTE_NONE,
    )
    samples_train=[dict(e) for e in reader]

tsv_path1 = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Data/dev_raw.tsv"
with open(tsv_path1) as f:
    reader = csv.DictReader(
        f,
        delimiter="\t",
        quotechar=None,
        doublequote=False,
        lineterminator="\n",
        quoting=csv.QUOTE_NONE,
    )
    samples_dev=[dict(e) for e in reader]

# id	audio	duration_ms	n_frames	tgt_text	speaker	tgt_lang	src_text
# dict_keys(['id', 'audio', 'n_frames', 'src_text', 'tgt_text', 'speaker'])
print("========================")
print(len(samples_train))
print(len(samples_dev))
print("========================")

# def Data_Static():
#     Total = 0
#     No_rel = 0
#     for i in range(len(samples)):
#         Speech = samples[i]
#         R = extract_triplets(Speech['tgt_text'])
#         Total += len(R)
#         for r in R:
#             if r['type'] == 'no relation':
#                 No_rel += 1
#
#     print("total:",Total)
#     print("No Relation:", No_rel)
#     print(No_rel/Total)


# import asyncio
# import edge_tts
# VOICE = "en-GB-SoniaNeural"
# path_aud = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Datasets/conll04/audio/train_new/"
#
# async def my_function(text, voice, output, rate = '-20%', volume = '+0%'):
#     tts = edge_tts.Communicate(text=text, voice=voice, rate=rate)
#     await tts.save(output)
#
# for i in range(len(fcc_data)):
#     Soure = fcc_data[i]
#     sentence = " ".join(Soure['tokens'])
#     print(sentence)
#
#     Speech = samples[i]
#     OUTPUT_FILE = path_aud + Speech['id'] + ".mp3"
#
#     asyncio.run(my_function(sentence,VOICE,OUTPUT_FILE))

# entity_num=[]
# token_num = []
# for i in range(len(fcc_data)):
#     Soure = fcc_data[i]
#     entity_num.append(len(Soure['entities']))
#     token_num.append(len(Soure['tokens']))
#
# entity_num = torch.tensor(entity_num).float()
# print(entity_num.max())
# print(entity_num.mean())
#
#
# token_num = torch.tensor(token_num).float()
# print(token_num.max())
# print(token_num.mean())
def Data_marge1():
    Id_list = {}
    samples = []
    for i in range(len(samples_train)):
        Id = samples_train[i]['id'].split("_")[1]
        if Id not in Id_list:
            Id_list[Id]=[i]
        else:
            Id_list[Id].append(i)

    Id_n = []
    for key in Id_list.keys():
        Id_n.append(len(Id_list[key]))
    print(len(Id_list))

    for key in Id_list.keys():
        for i in Id_list[key][:34]:
            samples_train[i]['tgt_text'] = samples_train[i]['src_text']
            samples_train[i]['duration_ms'] = 11041
            samples_train[i]['tgt_lang'] = 'en'
            samples.append(samples_train[i])
    print(len(samples))

    tsv_path_new = tsv_path[:-4] + "_new" + tsv_path[-4:]
    print("数据保存到：", tsv_path_new)
    columns_name = list(samples[1].keys())
    with open(tsv_path_new, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns_name,
                                delimiter="\t",
                                quotechar=None,
                                doublequote=False,
                                lineterminator="\n",
                                quoting=csv.QUOTE_NONE
                                )
        writer.writeheader()
        for e in samples:
            # col = [e[key] for key in columns_name]
            writer.writerow(e)

def Data_marge():
    samples = []
    samples_len = []
    for i in range(len(samples_train)):
        if samples_train[i]['src_text'] and samples_train[i]['n_frames']:
            l = len(samples_train[i]['src_text'].split())
            if(l>10):
                samples_train[i]['tgt_text'] = samples_train[i]['src_text']
                samples_train[i]['duration_ms'] = 11041
                samples_train[i]['tgt_lang'] = 'en'
                samples_train[i]['speaker'] = samples_train[i]['speaker'].split(".")[-1]
                samples.append(samples_train[i])

                samples_train[i]['audio']
            samples_len.append(l)
    samples_len = torch.tensor(samples_len).float()

    # samples = samples[:100000]
    tsv_path_new = tsv_path[:-4] + "_new" + tsv_path[-4:]
    print("数据保存到:", tsv_path_new)
    print("数据量:",len(samples))
    columns_name = list(samples[1].keys())
    with open(tsv_path_new, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns_name,
                                delimiter="\t",
                                quotechar=None,
                                doublequote=False,
                                lineterminator="\n",
                                quoting=csv.QUOTE_NONE
                                )
        writer.writeheader()
        for e in samples:
            # col = [e[key] for key in columns_name]
            writer.writerow(e)

if __name__ == '__main__':
    Data_marge()


# import torch
# from TTS.api import TTS
#
# # Get device
# # device = "cuda" if torch.cuda.is_available() else "cpu"
#
# # List available 🐸TTS models
# tts_model_list = TTS().list_models()
# for model_name in tts_model_list:
#     name = model_name.split("/")
#     if(name[1] in ["multilingual","en"] and "vits" in model_name):
#         print(model_name)

# tts = TTS("tts_models/en/multi-dataset/tortoise-v2")
# print("模型speakers:", tts.speakers)
# Init TTS
# tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
# wav = tts.tts(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en")
# Text to speech to a file
# tts.tts_to_file(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en", file_path="output.wav")