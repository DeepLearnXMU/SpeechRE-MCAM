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


Soure_path = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Datasets/conll04/soure_data/"
with open(Soure_path+"conll04_dev.json", 'r') as fcc_file:
    fcc_data = json.load(fcc_file)

with open(Soure_path + "conll04_types.json", 'r') as types_file:
    types = json.load(types_file)

with open(Soure_path + "conll04_prediction_example.json", 'r') as example_file:
    example = json.load(example_file)


tsv_path = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Data/dev_conll04_map.tsv"
with open(tsv_path) as f:
    reader = csv.DictReader(
        f,
        delimiter="\t",
        quotechar=None,
        doublequote=False,
        lineterminator="\n",
        quoting=csv.QUOTE_NONE,
    )
    samples=[dict(e) for e in reader]


# Function to parse the generated text and extract the triplets
def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
    return triplets

print("========================")
print(len(fcc_data))
print(len(samples))
print("========================")

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

for i in range(len(fcc_data)):
    Soure = fcc_data[i]
    Speech = samples[i]
    R = extract_triplets(Speech['tgt_text'])
    # ========音频长度处理============
    path = Speech['audio']
    _path, *extra = path.split(":")
    extra = [int(i) for i in extra]
    waveform, _ = get_waveform(_path, always_2d=False)
    L = -1
    waveform = abs(waveform * 10000.0)  # 音频元素放大好判断0
    for l in range(11,len(waveform)):
        if((waveform[-l:]).sum()==0.0):
            L = l
        else:
            break

    if(L>10):
        Speech['n_frames'] = str((int(Speech['n_frames']) - L + 10))
    Speech['audio'] = _path +":0:"+Speech['n_frames']

    Flag_t = True
    for j, r in enumerate(Soure['relations']):
        h_idx = r['head']
        t_idx = r['tail']

        h_start_idx = Soure['entities'][h_idx]['start']
        h_end_idx = Soure['entities'][h_idx]['end']

        t_start_idx = Soure['entities'][t_idx]['start']
        t_end_idx = Soure['entities'][t_idx]['end']

        h_name = " ".join(Soure['tokens'][h_start_idx:h_end_idx])
        t_name = " ".join(Soure['tokens'][t_start_idx:t_end_idx])

        r['head'] = h_name
        r['tail'] = t_name
        r_name = r['type'].split("_")[0]

        # print(r)
        Flag = False
        for R_temp in R:
            # if(R_temp['type']==r['type'] and R_temp['head']==r['head'] and R_temp['tail']==r['tail']):
            if (R_temp['head'] == r['head'] and R_temp['tail'] == r['tail']):
                Flag = True

        if not Flag:
            # Flag_t = False
            R[0]['head'] = r['head']
            R[0]['tail'] = r['tail']
            Speech['tgt_text'] = "<triplet> " + r['head'] + " <subj> " + r['tail'] + " <obj> " + R[0]['type']
            print("============",i,"===============")
            print(Speech['tgt_text'])
            print(R)
            print(r)
            print("========================")

    # ========源语言处理============
    tokens = [[w] for w in Soure['tokens']]
    for e in enumerate(Soure['entities']):
        e = e[-1]
        e_t = e["type"].lower()
        mask_start = "<entity>"
        mask_end = "</entity>"
        # mask_start = "<" + e_t + ">"
        # mask_end = "</" + e_t + ">"
        e_start = int(e["start"])
        e_end = int(e["end"]) - 1

        tokens[e_start].insert(0, mask_start)
        tokens[e_end].append(mask_end)

    tokens_p = []
    for w_s in tokens:
        tokens_p = tokens_p + w_s
    Soure['tokens'] = tokens_p
    # entity = [None for t in Soure['tokens']]
    # for e in enumerate(Soure['entities']):
    #     e = e[-1]
    #     e_t = e["type"].lower()
    #     mask_start = "<" + e_t + ">"
    #     mask_end = "</" + e_t + ">"
    #     e_start = int(e["start"])
    #     e_end = int(e["end"])
    #
    #     if (len(entity) <= e_start):
    #         entity.append(mask_start)
    #     else:
    #         if(entity[e_start] == None):
    #             entity[e_start] = mask_start
    #         else:
    #             entity.insert(e_start+1, mask_start)
    #
    #     if (len(entity) <= e_end):
    #         entity.append(mask_end)
    #     else:
    #         if(entity[e_end] == None):
    #             entity[e_end] = mask_end
    #         else:
    #             entity.insert(e_end, mask_end)
    #
    # if(i==800):
    #     print(entity)
    #
    # Shift = 0
    # for j in range(len(entity)):
    #     if (entity[j]):
    #         Soure['tokens'].insert(Shift + j, entity[j])
    #         Shift += 1

    # entity = [None for t in Soure['tokens']]
    # for e in enumerate(Soure['entities']):
    #     e = e[-1]
    #     e_t = e["type"].lower()
    #     mask_start = "<" + e_t + ">"
    #     mask_end = "</" + e_t + ">"
    #     e_start = int(e["start"])
    #     e_end = int(e["end"])
    #
    #     if (len(entity) <= e_start):
    #         entity.append(mask_start)
    #     else:
    #         entity[e_start] = mask_start
    #
    #     if (len(entity) <= e_end):
    #         entity.append(mask_end)
    #     else:
    #         entity[e_end] = mask_end
    #
    # Shift = 0
    # for i in range(len(entity)):
    #     if (entity[i]):
    #         Soure['tokens'].insert(Shift + i, entity[i])
    #         Shift += 1

    # print(" ".join(Soure['tokens']))
    # print(R)

    if(Flag_t):
        Speech["src_text"] = " ".join(Soure['tokens'])

    # print(Speech)

tsv_path_new = tsv_path[:-4]+"_new"+tsv_path[-4:]
columns_name = list(Speech.keys())
with open(tsv_path_new,"w") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=columns_name,
                            delimiter = "\t",
                            quotechar = None,
                            doublequote = False,
                            lineterminator = "\n",
                            quoting = csv.QUOTE_NONE
                            )
    writer.writeheader()
    for e in samples:
        # col = [e[key] for key in columns_name]
        writer.writerow(e)

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