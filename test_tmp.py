import random
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

Soure_path = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Datasets/re-tacred/soure_data/"
Rel_Org = []
Rel_dict = {"org":"organization","per":"person"}
with open(Soure_path+"test.txt", 'r') as fcc_file:
    fcc_data = fcc_file.readlines()
    fcc_data = [eval(line) for line in fcc_data]

    Sentence_id = {}
    for i, data in enumerate(fcc_data):
        sentence = " ".join(data['token'])
        if(sentence not in Sentence_id):
            Sentence_id[sentence]=[i]
        else:
            Sentence_id[sentence].append(i)

        if (":" in data['relation']):
            tmp = data['relation'].split(":")
            tmp[0] = Rel_dict[tmp[0]]
            data['relation'] = "_".join(tmp)

        data['relation'] = " ".join(re.split("_|/",data['relation']))
        data['relation'] = data['relation'].replace('stateorprovince','state or province')
        if(data['relation'] not in Rel_Org):
            Rel_Org.append(data['relation'])

    score_tris = []
    sentences = []
    for key in Sentence_id.keys():
        tokens = [[w] for w in fcc_data[Sentence_id[key][0]]['token']]

        Relation_list = []
        rep = []
        for s_i in Sentence_id[key]:
            head_n = fcc_data[s_i]['h']['name'].replace('-RRB- ','')
            tail_n = fcc_data[s_i]['t']['name'].replace('-RRB- ','')
            head_s, head_e = fcc_data[s_i]['h']['pos']
            tail_s, tail_e = fcc_data[s_i]['t']['pos']
            head_e = head_e - 1
            tail_e = tail_e - 1

            Relation_list.append({'head': head_n, 'tail': tail_n, 'type': fcc_data[s_i]['relation']})
            mask_start = "<entity>"
            mask_end = "</entity>"
            if (head_s,head_e) not in rep:
                rep.append((head_s,head_e))
                tokens[head_s].insert(0, mask_start)
                # if(head_e>=len(tokens)):
                #     head_e = head_e - 1
                tokens[head_e].append(mask_end)
            if (tail_s,tail_e) not in rep:
                rep.append((tail_s,tail_e))
                tokens[tail_s].insert(0, mask_start)
                # if(tail_e>=len(tokens)):
                #     tail_e = tail_e - 1
                tokens[tail_e].append(mask_end)

        tokens_p = []
        for w_s in tokens:
            tokens_p = tokens_p + w_s

        score_tris.append(Relation_list)
        sentences.append(" ".join(tokens_p))

with open(Soure_path + "rel2id.json", 'r') as types_file:
    types = json.load(types_file)

# tsv_path = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Data/train_tacred.tsv"
# with open(tsv_path) as f:
#     reader1 = csv.DictReader(
#         f,
#         delimiter="\t",
#         quotechar=None,
#         doublequote=False,
#         lineterminator="\n",
#         quoting=csv.QUOTE_NONE,
#     )
#     samples1=[dict(e) for e in reader1]


tsv_path = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Datasets/re-tacred/test_re-tacred_new.tsv"
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
print(len(Sentence_id))
print("========================")


def Generate_label_data():
    tsv_path = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Data/train_conll04_map_new.tsv"
    with open(tsv_path) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        samples_train = [dict(e) for e in reader]

    tsv_path = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Data/dev_conll04_map_new.tsv"
    with open(tsv_path) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        samples_dev = [dict(e) for e in reader]

    tsv_path = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Data/test_conll04_map_new.tsv"
    with open(tsv_path) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        samples_test = [dict(e) for e in reader]


    Samples_new = []
    for samples in [samples_train,samples_dev,samples_test]:
        for i in range(len(samples)):
            Speech = {
                'id':samples[i]['id'],
                'src_text': samples[i]['src_text'].replace(' </entity>', '').replace('<entity> ', '').replace('</entity> ', '').replace(' <entity>', '')
            }
            Samples_new.append(Speech)

    print("===============================")
    print("句子数：",len(Samples_new))
    print("===============================")
    samples = Samples_new
    # tsv_path_new = tsv_path[:-4] + "_label_data" + tsv_path[-4:]
    tsv_path_new = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Data/test_conll04_label_data.tsv"
    print("数据保存到：", tsv_path_new)
    columns_name = list(Speech.keys())
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

def Data_Static():
    Total = 0
    No_rel = 0
    for i in range(len(samples)):
        Speech = samples[i]
        R = extract_triplets(Speech['tgt_text'])
        Total += len(R)
        for r in R:
            if r['type'] == 'no relation':
                No_rel += 1

    print("total:",Total)
    print("No Relation:", No_rel)
    print(No_rel/Total)

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
def Data_marge():
    Rel_name = []
    Samples_new = []
    for i in range(len(samples)):
        Speech = samples[i]
        R = extract_triplets(Speech['tgt_text'])

        # ========音频长度处理============
        path = Speech['audio']
        _path, *extra = path.split(":")
        extra = [int(i) for i in extra]
        if (not os.path.exists(_path)):
            print(_path)
            continue
        waveform, _ = get_waveform(_path, always_2d=False)
        L = -1
        waveform = abs(waveform * 10000.0)  # 音频元素放大好判断0
        for l in range(11, len(waveform)):
            if ((waveform[-l:]).sum() == 0.0):
                L = l
            else:
                break

        if (L > 10):
            Speech['n_frames'] = str((int(Speech['n_frames']) - L + 10))
        Speech['audio'] = _path + ":0:" + Speech['n_frames']

        for r in R:
            r['type'] = r['type'].replace('stateorprovince', 'state or province')
            if (r['type'] not in Rel_name):
                Rel_name.append(r['type'])

        N_R = 0
        for r in R:
            if r not in score_tris[i]:
                N_R += 1
        if (N_R > 0):
            print(R)
            print(score_tris[i])
            print("\t\t:", N_R)
        else:
            Speech["src_text"] = sentences[i]

        Samples_new.append(Speech)
        if (i % 1000 == 0):
            print("当前步数：", i)

        # if Flag:
        #     if(i!=K):
        #         print(i, "=>", K)
        #     Speech["src_text"] = sentences[K]
        #     break
        #
        # for K, R_m in enumerate(score_tris):
        #     Flag = True
        #     if(len(R)==len(R_m)):
        #         for r in R:
        #             if r not in R_m:
        #                 Flag = False
        #                 break
        #     else:
        #         Flag = False
        #
        #     if Flag:
        #         if(i!=K):
        #             print(i, "=>", K)
        #         Speech["src_text"] = sentences[K]
        #         break

        # if(not Flag):
        #     print("=============关系三元组匹配错误==================")
        #     print(R)
        #     print(i)
        #     for K, sen in enumerate(sentences):
        #         Flag = True
        #         for r in R:
        #             if r['head'] not in sen or r['tail'] not in sen:
        #                 Flag = False
        #                 break
        #         if Flag:
        #             print("=============句子匹配正确==================")
        #             print(i, "=>", K)
        #             pdb.set_trace()
        #             break
        #
        #     if(not Flag):
        #         print("=============句子匹配错误==================")
        #         pdb.set_trace()

        # sent_id = []
        # for k, sentence in enumerate(Sentence_id.keys()):
        #     Flag = True
        #     for r in R:
        #         if(r['head'] not in sentence or r['tail'] not in sentence):
        #             Flag = False
        #             break
        #
        #     if Flag:
        #         sent_id.append(k)
        #
        # if(len(sent_id)!=1):
        #     print(R)
    # print("\n\n")
    # print("=====================")
    # for R in Rel_Org:
    #     if R not in Rel_name:
    #         print(R)
    # print("=====================")
    # for R in Rel_name:
    #     if R not in Rel_Org:
    #         print(R)

    # pdb.set_trace()
    samples = Samples_new
    tsv_path_new = tsv_path[:-4] + "_new" + tsv_path[-4:]
    print("数据保存到：", tsv_path_new)
    columns_name = list(Speech.keys())
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

def Generate_Hunan_data():
    tsv_path = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Data/test_conll04_map_new.tsv"
    with open(tsv_path) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        samples = [dict(e) for e in reader]

    print("========================")
    print(len(samples))
    print("========================")

    for i in range(len(samples)):
        Speech = samples[i]
        R = extract_triplets(Speech['tgt_text'])

        # ========音频长度处理============
        path = Speech['audio']
        _path, *extra = path.split(":")
        extra = [int(i) for i in extra]
        waveform, _ = get_waveform(_path, always_2d=False)
        L = -1

        waveform = abs(waveform * 10000.0)  # 音频元素放大好判断0
        start = 0
        for l in range(len(waveform)):
            if (waveform[l] != 0.0):
                start = max(start, l - 10)
                break

        end = len(waveform)-1
        for l in range(end, -1, -1):
            if (waveform[l] != 0.0):
                end = min(end, l + 10)
                break
        end += 1
        Speech['n_frames'] = str(end - start)
        start = str(start)
        Speech['audio'] = _path + ":" + start + ":" + Speech['n_frames']

    tsv_path_new = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Data/test_conll04_map_new2.tsv"
    print("数据保存到：", tsv_path_new)
    columns_name = list(Speech.keys())
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


def read_tsv_file(tsv_path):
    with open(tsv_path) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
            quotechar=None,
            doublequote=False,
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
        )
        samples = [dict(e) for e in reader]
    return samples

def writer_tsv_file(tsv_path_new,samples):
    print("数据保存到：", tsv_path_new)
    columns_name = list(samples[0].keys())
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
def Generate_Hunan_data_banlance():
    file_path = '/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Datasets/Speaker/CoNLL04/'
    speaker = ['女7','女8','男7','男8']
    list_speaker = []
    for i in range(len(speaker)):
        Path = file_path + speaker[i]
        all_file_name = os.listdir(Path)
        list_speaker.append([d[:-4] for d in all_file_name])


    test_path = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Data/test_conll04_map_new2.tsv"
    samples_test = read_tsv_file(test_path)
    train_path = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Data/train_conll04_map_new2.tsv"
    samples_train = read_tsv_file(train_path)
    dev_path = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Data/dev_conll04_map_new2.tsv"
    samples_dev = read_tsv_file(dev_path)

    print("===========训练数据=============")
    print(len(samples_train))
    print("===============================")
    print("===========验证数据=============")
    print(len(samples_dev))
    print("===============================")
    print("===========测试数据=============")
    print(len(samples_test))
    print("===============================")
    # pdb.set_trace()
    # [x['speaker'] for x in samples_train]
    # for samples in [samples_train, samples_dev, samples_test]:
    #     for i in range(len(samples)):
    #         Id = samples[i]['id']
    #         flag = True
    #         for j in range(len(speaker)):
    #             if Id in list_speaker[j]:
    #                 samples[i]['speaker']=str(j)
    #                 flag=False
    #                 break
    #         if flag:
    #             print("当前样本不匹配")
    #             print(samples[i])
    samples_train1, samples_dev1, samples_test1=[],[],[]
    for samples in [samples_train, samples_dev, samples_test]:
        for i in range(len(samples)):
            R = random.random()
            if (R <= 0.64):
                samples_train1.append(samples[i])
            if (0.64 < R and R <=(0.64 + 0.20)):
                samples_test1.append(samples[i])
            if ((0.64 + 0.20) < R):
                samples_dev1.append(samples[i])

    print("\n\n===========训练数据=============")
    print(len(samples_train1))
    print("===============================")
    print("===========验证数据=============")
    print(len(samples_dev1))
    print("===============================")
    print("===========测试数据=============")
    print(len(samples_test1))
    print("===============================")

    test_path = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Data/test_conll04_map_new3.tsv"
    train_path = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Data/train_conll04_map_new3.tsv"
    dev_path = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Data/dev_conll04_map_new3.tsv"
    writer_tsv_file(dev_path, samples_dev1)
    writer_tsv_file(test_path, samples_test1)
    writer_tsv_file(train_path, samples_train1)

def Generate_Hunan_data_spraker():
    file_path = '/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Datasets/Speaker/CoNLL04/'
    speaker = ['男7','男8','女7','女8']
    list_speaker = []
    for i in range(len(speaker)):
        Path = file_path + speaker[i]
        all_file_name = os.listdir(Path)
        list_speaker.append([d[:-4] for d in all_file_name])

    test_path = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Data/test_conll04_map_new2.tsv"
    samples_test = read_tsv_file(test_path)
    train_path = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Data/train_conll04_map_new2.tsv"
    samples_train = read_tsv_file(train_path)
    dev_path = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Data/dev_conll04_map_new2.tsv"
    samples_dev = read_tsv_file(dev_path)

    print("===========训练数据=============")
    print(len(samples_train))
    print("===============================")
    print("===========验证数据=============")
    print(len(samples_dev))
    print("===============================")
    print("===========测试数据=============")
    print(len(samples_test))
    print("===============================")
    # [x['speaker'] for x in samples_train]
    samples_train1, samples_dev1, samples_test1 = [], [], []
    for samples in [samples_train, samples_dev, samples_test]:
        for i in range(len(samples)):
            Id = samples[i]['id']
            flag = True
            for j in range(len(speaker)):
                if Id in list_speaker[j]:
                    samples[i]['speaker']=str(j)
                    flag=False
                    break
            if flag:
                print("当前样本不匹配")
                print(samples[i])

            if(j==0 or j==1):
                # new_id = "train-" + str(len(samples_train1))
                # samples[i]['id'] = new_id
                samples_train1.append(samples[i])
            if(j==2):
                # new_id = "dev-" + str(len(samples_dev1))
                # samples[i]['id'] = new_id
                samples_dev1.append(samples[i])
            if(j==3):
                # new_id = "test-" + str(len(samples_test1))
                # samples[i]['id'] = new_id
                samples_test1.append(samples[i])

    # for samples in [samples_train, samples_dev, samples_test]:
    #     for i in range(len(samples)):
    #         R = random.random()
    #         if (R <= 0.64):
    #             samples_train1.append(samples[i])
    #         if (0.64 < R and R <=(0.64 + 0.20)):
    #             samples_test1.append(samples[i])
    #         if ((0.64 + 0.20) < R):
    #             samples_dev1.append(samples[i])

    print("\n\n===========训练数据=============")
    print(len(samples_train1))
    print("===============================")
    print("===========验证数据=============")
    print(len(samples_dev1))
    print("===============================")
    print("===========测试数据=============")
    print(len(samples_test1))
    print("===============================")

    test_path = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Data/test_conll04_map_new4.tsv"
    train_path = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Data/train_conll04_map_new4.tsv"
    dev_path = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Data/dev_conll04_map_new4.tsv"
    writer_tsv_file(dev_path, samples_dev1)
    writer_tsv_file(test_path, samples_test1)
    writer_tsv_file(train_path, samples_train1)

def check_Hunan_data_banlance():
    test_path = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Data/test_conll04_map_new3.tsv"
    samples_test = read_tsv_file(test_path)
    train_path = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Data/train_conll04_map_new3.tsv"
    samples_train = read_tsv_file(train_path)
    dev_path = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Data/dev_conll04_map_new3.tsv"
    samples_dev = read_tsv_file(dev_path)

    print("===========训练数据=============")
    print(len(samples_train))
    print("===============================")
    print("===========验证数据=============")
    print(len(samples_dev))
    print("===============================")
    print("===========测试数据=============")
    print(len(samples_test))
    print("===============================")

    for i in range(len(samples_train)):
        Id = samples_train[i]['id']
        pass


def Generate_Hunan_data_re():
    tsv_path = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Data/test_re-tacred_new.tsv"
    samples = read_tsv_file(tsv_path)

    print("========================")
    print(len(samples))
    print("========================")

    for i in range(len(samples)):
        Speech = samples[i]

        # ========音频长度处理============
        path = Speech['audio']
        _path, *extra = path.split(":")
        extra = [int(i) for i in extra]
        waveform, _ = get_waveform(_path, always_2d=False)

        waveform = abs(waveform * 10000.0)  # 音频元素放大好判断0
        start = 0
        for l in range(len(waveform)):
            if (waveform[l] != 0.0):
                start = max(start, l - 10)
                break

        end = len(waveform)-1
        for l in range(end, -1, -1):
            if (waveform[l] != 0.0):
                end = min(end, l + 10)
                break
        end += 1
        Speech['n_frames'] = str(end - start)
        start = str(start)
        Speech['audio'] = _path + ":" + start + ":" + Speech['n_frames']
    pdb.set_trace()
    tsv_path_new = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Data/test_re-tacred_new1.tsv"
    print("数据保存到：", tsv_path_new)
    columns_name = list(Speech.keys())
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

def Del_Error_data_re():
    tsv_path = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Data/train_re-tacred_new.tsv"
    samples = read_tsv_file(tsv_path)

    print("========================")
    print(len(samples))
    print("========================")

    n_frames, n_tokens = [], []
    for i in range(len(samples)):
        Speech = samples[i]
        n_frames.append(int(Speech['n_frames']))
        n_tokens.append(len(Speech['tgt_text'].split()))

    n_frames = torch.tensor(n_frames)
    n_tokens = torch.tensor(n_tokens)
    n_frames1, order = n_frames.sort(descending=True)

    Del_id = []
    for i in order:
        Flag1, Flag2 = False, False
        if(i<(len(order) - 1)):
            if (n_tokens[i] > n_tokens[i + 1] + 40):
                Flag1 = True

        if (i > 0):
            if (n_tokens[i - 1] + 40 < n_tokens[i]):
                Flag2 = True

        if(Flag1 and Flag2):
            Del_id.append(i)
            print(n_tokens[i].item(), [x.item() for x in n_tokens[i-2:i+2]], [x.item() for x in n_frames[i-2:i+2]])
            # pdb.set_trace()

    print("=========错误实例个数============")
    print(len(Del_id))
    print("===============================")

    samples_new = []
    for i in range(len(samples)):
        if(i not in Del_id):
            samples_new.append(samples[i])

    print("========================")
    print(len(samples_new))
    print("========================")

    samples = samples_new
    tsv_path_new = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Data/train_re-tacred_new1.tsv"
    print("数据保存到：", tsv_path_new)
    columns_name = list(Speech.keys())
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

def generate_tgt_text(rel_list):
    Flag = [1] * len(rel_list)

    tgt_text = ""
    for i in range(len(rel_list)):
        if Flag[i]==1:
            H, R, T = rel_list[i]['head'], rel_list[i]['type'], rel_list[i]['tail']
            tgt_text += " <triplet> " + H + " <subj> " + T + " <obj> " + R
            Flag[i] = 0
            for j in range(len(rel_list)):
                if i < j and Flag[j] == 1:
                    H1, R1, T1 = rel_list[j]['head'], rel_list[j]['type'], rel_list[j]['tail']
                    if H == H1:
                        tgt_text += " <subj> " + T + " <obj> " + R
                        Flag[j] = 0
    return tgt_text[1:]


def Generate_None_data():
    tsv_path = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Data/train_re-tacred_new.tsv"
    samples = read_tsv_file(tsv_path)
    print("========================")
    print(len(samples))
    print("========================")

    for i in range(len(samples)):
        Speech = samples[i]
        R_list = extract_triplets(Speech['tgt_text'])

        rel_list = []
        for R in R_list:
            if(R['type']!='no relation'):
                rel_list.append(R)

        tgt_text = generate_tgt_text(rel_list)
        rel_list1 = extract_triplets(tgt_text)
        if(len(rel_list)==0):
            pdb.set_trace()

        if(i%100):
            print(rel_list)
            print(rel_list1)

        Flag = False
        if len(rel_list1) != len(rel_list):
            print("11111111111")
            Flag = True

        for j in range(len(rel_list1)):
            Flag1 = True
            for k in range(len(rel_list)):
                if (rel_list1[j]['head'] == rel_list[k]['head'] and rel_list1[j]['tail'] == rel_list[k]['tail'] and rel_list1[j]['type'] == rel_list[k]['type']):
                    Flag1 = False
                    break
            if Flag1:
                Flag = True

        if(Flag):
            print("存在转化错误")
            pdb.set_trace()
        else:
            Speech['tgt_text'] = tgt_text

    pdb.set_trace()
    tsv_path_new = "/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Data/train_re-tacred_new1.tsv"
    writer_tsv_file(tsv_path_new, samples)


import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
def create_garph():
    x = [18.87, 55.66, 10.41, 17.21, 43.37, 3.20, 24.89, 59.57, 12.50]
    y = [40.06, 77.12, 23.32, 34.12, 58.74, 7.91, 40.86, 80.54, 23.72]

    for i in range(len(x)):
        print(y[i]-x[i])

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 12,
             }
    # 折线图
    x = [1, 2, 3, 4, 5, 6]  # 点的横坐标
    k1 = [38.08, 39.17, 40.06, 38.55, 39.42, 38.09]  # 线1的纵坐标
    k2 = [73.00, 75.73, 77.12, 76.36, 76.91, 75.42]  # 线2的纵坐标
    k3 = [20.79, 22.54, 23.32, 22.42, 23.07, 21.42]  # 线2的纵坐标
    plt.plot(x, k1, marker='s', linestyle='--', label=r'            ')
    plt.plot(x, k2, marker='^', linestyle=':', label=r'         ')
    plt.plot(x, k3, marker='o', linestyle='-', label=r'         ')

    # plt.plot(x, k1, 'b*-', label="Entity")  # s-:方形
    # plt.plot(x, k2, 'rs-', label="Relation")  # o-:圆形
    # plt.plot(x, k3, marker='s', linestyle='--', label="Triplet")  # o-:圆形
    plt.xlabel("Number of learnable vectors K",font1)  # 横坐标名字
    plt.ylabel("F1 Score",font1)  # 纵坐标名字
    plt.legend(loc="lower right",prop=font1)  # 图例
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)

    plt.grid(linestyle='-.')  # 生成网格
    plt.show()


def Create_graph1():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from collections import namedtuple
    from matplotlib import font_manager
    from matplotlib import rcParams

    import seaborn as sns
    # font_manager.fontManager.addfont('analysis/fonts/Times-New-Roman.ttf')

    # plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['font.serif'] = ['analysis/fonts/Times-New-Roman.ttf']  # 替换为实际的字体文件路径
    plt.rc('font', family='Times New Roman')

    plt.rc('font', size=8)
    config = {
        "mathtext.fontset": 'stix',
    }
    rcParams.update(config)
    import seaborn as sns

    # 输入数据
    x = ['0.25', '0.5', '0.75', '1', '2']

    # kd bleu
    y1_bleu = [13.85, 14.24, 14.54, 15.39, 15.21]

    # ocr bleu
    y2_bleu = [14.64, 14.89, 15.11, 15.39, 14.57]

    # tit bleu
    y3_bleu = [14.54, 14.75, 15.06, 15.39, 14.40]

    # 设置颜色代码
    palette = sns.color_palette()

    # 设置字体
    font = {'family': 'Times New Roman',
            'size': 14}
    plt.rc('font', **font)
    plt.figure(figsize=(7, 5), dpi=300)

    # 绘制柱状图
    bar_width = 0.35
    index = [0, 1, 2, 3, 4]
    plt.plot(x, y2_bleu, marker='s', linestyle='--', label=r'$\alpha$', color=palette[1])
    plt.plot(x, y3_bleu, marker='^', linestyle=':', label=r'$\beta$', color=palette[2])
    plt.plot(x, y1_bleu, marker='o', linestyle='-', label=r'$\gamma$', color=palette[0])

    # 添加标题和标签
    plt.xlabel("Coefficient", fontsize=16)
    plt.ylabel("BLEU", fontsize=16)

    # 添加图例
    plt.legend(loc="lower right", frameon=True, fontsize=12)

    # 设置刻度字体和范围
    # plt.xticks([i + bar_width / 2 for i in index], x, fontsize=14)
    # plt.yticks(fontsize=14)

    plt.ylim(13.0, 15.5)

    # 在柱状图上显示具体数值
    # def add_labels(rects):
    #     for rect in rects:
    #         height = rect.get_height()
    #         plt.text(rect.get_x() + rect.get_width() / 2, height, '{:.2f}'.format(height), ha='center', va='bottom', fontsize=12)
    #         rect.set_edgecolor('white')

    # add_labels(bars1)
    # add_labels(bars2)

    # plt.savefig('analysis/loss_weight.png', bbox_inches='tight')
    # 显示图像
    plt.show()


if __name__ == '__main__':
    # Del_Error_data_re()
    # Generate_None_data()
    # Generate_Hunan_data_spraker()
    # Generate_Hunan_data_re()
    create_garph()

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