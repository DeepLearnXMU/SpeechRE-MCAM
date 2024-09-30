import pdb
import re
import copy
import logging
from typing import Optional
import math
import psutil
from omegaconf import DictConfig, open_dict
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from random import choice
import random

from fairseq import checkpoint_utils
from fairseq.tasks import FairseqTask
from fairseq.modules import (
    PositionalEmbedding,
    LayerNorm,
    FairseqDropout,
)
from fairseq.models import register_model
from fairseq.models.transformer import TransformerDecoder
from fairseq.models.wav2vec import (
    Wav2Vec2Seq2SeqConfig,
    Wav2Vec2Seq2SeqModel,
    Wav2VecEncoder,
    Embedding,
)
from fairseq.models.speech_to_text import (
    lengths_to_padding_mask,
    Conv1dSubsampler,
)
from fairseq.modules import MultiheadAttention

import csv

logger = logging.getLogger(__name__)

BLOCKS2REGEX = {
    "encoder.feat_extr": r"encoder.*\.feature_extractor\..*|"
                         r"encoder.*\.post_extract_proj\..*|"
                         r"encoder.*\.pos_conv\..*",
    "encoder.self_attn": r"encoder.*\.self_attn\..*",
    "encoder.layer_norm": r"encoder.*layer_norm.*",
    "encoder.ffn": r"encoder.*\.fc[1-2]\..*",
    "adapter": r"encoder\.adapter.*",
    "len_adaptor": r"encoder\.len_adaptor.*",
    "decoder.embedding": r"decoder\.embed_tokens.*|"
                         r"decoder\.embed_positions.*|"
                         r"decoder\.layernorm_embedding.*",
    "decoder.self_attn": r"decoder.*\.self_attn\..*",
    "decoder.layer_norm": r"decoder.*layer_norm.*",
    "decoder.encoder_attn": r"decoder.*\.encoder_attn\..*",
    "decoder.ffn": r"decoder.*\.fc[1-2]\..*",
}
# '<mask>': 50264, '<triplet>': 50265, '<subj>': 50266, '<obj>': 50267, '<entity>': 50268, '</entity>': 50269

special_tokens_id={'<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3, '<mask>': 50264, '<triplet>': 50265, '<subj>': 50266,
                   '<obj>': 50267, '<peop>': 50268, '</peop>': 50269,'<org>': 50270, '</org>': 50271, '<other>': 50272,
                   '</other>': 50273, '<loc>': 50274, '</loc>': 50275}
# entity_ids = [50268, 50269, 1]
entity_ids = [50268, 50269, 1, 0, 2, 50270]
entity_start = ['<peop>','<org>','<other>','<loc>']
entity_end = {'<peop>':'</peop>','<org>':'</org>','<other>':'</other>','<loc>':'</loc>'}

entity_start_id = [50268,50270,50272,50274]
entity_end_id = [50269,50271,50273,50275]

@dataclass
class Wav2Vec2Seq2SeqModConfig(Wav2Vec2Seq2SeqConfig):
    freeze_layers: str = field(
        default="",
        metadata={"help": "finetune only LayerNorm and Attention (LNA) layers"}
    )
    adapter_dim: Optional[int] = field(
        default=None,
        metadata={"help": "projection size of the Adapter"}
    )
    adapter_post: bool = field(
        default=False,
        metadata={"help": "if true, the Adapter is placed after the "
                          "Length Adaptor"}
    )
    adapter_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability for the encoder-decoder "
                          "attention weights (if it's not specified, the "
                          "decoder_attention_dropout is used)"}
    )
    len_adaptor_kernel_sizes: str = field(
        default="3,3",
        metadata={"help": "kernel sizes of the Length Adaptor (Conv1d)"}
    )
    len_adaptor_channels: int = field(
        default=1024,
        metadata={"help": "# of channels in the Length Adaptor (Conv1d)"}
    )
    load_pretrained_decoder_from: Optional[str] = field(
        default=None,
        metadata={"help": "model to take decoder weights from"}
    )
    adapter_input_dim: int = field(
        default=1024,
        metadata={"help": "encoder dimension"}
    )
    adapter_output_dim: int = field(
        default=100,
        metadata={"help": "decoder dimension"}
    )
    decoder_output_dim: int = field(
        default=768,
        metadata={"help": "decoder output dimension (extra linear layer "
                          "if different from decoder embed dim)"}
    )
    decoder_enc_attention_dropout: Optional[float] = field(
        default=None,
        metadata={"help": "dropout probability for the encoder-decoder "
                          "attention weights (if it's not specified, the "
                          "decoder_attention_dropout is used)"}
    )


@register_model("wav2vec_seq2seq_speechre", dataclass=Wav2Vec2Seq2SeqModConfig)
class Wav2Vec2Seq2SeqModModelRE(Wav2Vec2Seq2SeqModel):
    """
    Modified version of the wav2vec_seq2seq model.

    It adds these functionalities:
      - Use with the speech_to_text pipeline
      - Loading pretrained decoder
      - Finetuning only LNA layers
      - Using adapter and length_adaptor modules
    """
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.CE_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='sum')
        # model_ckpt = torch.load("/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Save/load_model/checkpoint_ASR_1.pt")
        # self.load_state_dict(model_ckpt['model'], strict=False)
        # self.encoder.Generate_len_adaptor()

    @classmethod
    def build_model(cls, cfg: Wav2Vec2Seq2SeqModConfig, task: FairseqTask):
        """Build a new model instance."""
        encoder = cls.build_encoder(cfg)  # 必须是FairseqEncoder类型
        decoder = cls.build_decoder(cfg)  # 必须是FairseqDecoder类型
        encoder.text_encoder = cls.encoder_text
        encoder.Seq_compressor = cls.Seq_compressor
        # encoder.text_adapter = cls.encoder_text.layers[-4:]
        encoder.projection = copy.deepcopy(decoder.output_projection)

        # encoder.text_adapter = copy.deepcopy(cls.encoder_text.layers[-4:])
        # encoder.seq_compressor = cls.seq_compressor
        # encoder.Blank = nn.Parameter(torch.FloatTensor(1, cls.encoder_text.layers[-1].embed_dim))
        # nn.init.uniform_(encoder.Blank)
        # encoder.Output_layer = Output_layer(decoder.layers[-1], decoder.output_projection)

        # encoder.Query_embs = cls.Query_embs
        # encoder.Qery_dropout = copy.deepcopy(cls.encoder_text.dropout_module)
        # encoder.Qery_layernorm = copy.deepcopy(cls.encoder_text.layernorm_embedding)
        # encoder.coss_att_layer = MultiHeadAttention(cls.encoder_text.layers[-1])
        # encoder.coss_att_layer = Self_Attention(decoder.layers[0])
        # encoder.emb_indxs = cls.emb_indxs
        # encoder.self_att_layer = MultiHeadAttention(in_dim=decoder.layers[0].embed_dim)
        # encoder.self_att_layer.self_attn_layer_norm = copy.deepcopy(decoder.layers[0].self_attn_layer_norm)
        # encoder.coss_att_layer = MultiHeadAttention(cls.encoder_text.layers[-1])
        # encoder.coss_att_layer.self_attn_layer_norm = copy.deepcopy(decoder.layers[0].self_attn_layer_norm)
        # encoder.output_layer = Output_layer(in_dim=decoder.layers[0].embed_dim)
        # encoder.output_layer.final_layer_norm = copy.deepcopy(decoder.layers[0].final_layer_norm)
        # encoder.output_layer.act = copy.deepcopy(decoder.layers[0].activation_fn)

        # encoder.coss_att_layer.compression_forward

        # encoder.text_project1 = copy.deepcopy(cls.encoder_text.layers[-1].fc1)
        # encoder.text_act = copy.deepcopy(cls.encoder_text.layers[-1].activation_fn)
        # encoder.text_drop = copy.deepcopy(cls.encoder_text.layers[-1].dropout_module)
        # encoder.text_project2 = copy.deepcopy(cls.encoder_text.layers[-1].fc2)
        # encoder.text_layer_norm = copy.deepcopy(cls.encoder_text.layers[-1].final_layer_norm)

        # cls.W = decoder.output_projection.weight.detach()       #.transpose(0, 1)

        task.tgt_dict = cls.bart.task.source_dictionary
        # 检查encoder和decoder类型并记录
        model = Wav2Vec2Seq2SeqModModelRE(encoder, decoder)
        model.freeze_blocks(cfg)
        model.unfreeze_model(encoder.len_adaptor)
        model.unfreeze_model(encoder.projection)
        model.unfreeze_model(encoder.Seq_compressor)
        # model.unfreeze_model(encoder.text_adapter)
        # model.unfreeze_model(encoder.w2v_model)
        # model.unfreeze_model(encoder.text_encoder)
        # model.unfreeze_model(decoder)

        # model.freeze_model(cls.bart.model.decoder)
        # model.freeze_model(cls.bart.model.encoder)
        # model.freeze_model(encoder.text_encoder)
        # model.freeze_model(encoder.text_adapter)
        # model.freeze_model(encoder.seq_compressor)
        # model.freeze_model(encoder.w2v_model)
        # model.freeze_model(encoder.len_adaptor)
        # model.freeze_model(decoder)
        return model

    @classmethod
    def build_encoder(cls, cfg: Wav2Vec2Seq2SeqModConfig):
        return Wav2VecEncoderMod(cfg)

    @classmethod
    def build_text_encoder(cls, cfg: Wav2Vec2Seq2SeqModConfig):
        from fairseq.models.bart import BARTModel
        bart = BARTModel.from_pretrained(cfg.load_pretrained_decoder_from,checkpoint_file='model.pt')
        bart.eval()
        return bart

    @classmethod
    def statistics_toke(cls):
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

        L = len(samples_train)
        Num = [0 for i in range(50270)]
        for i in range(L):
            Sample = samples_train[i]
            tokens = cls.bart.encode(Sample['src_text'])
            for t in tokens:
                Num[t]+=1
        for t in entity_ids:
            Num[t] = 0
        Num = torch.tensor(Num)
        Num_p = F.softmax(Num.float(),dim=-1)
        # Num = Num/L
        # Num.topk(100, dim=-1)
        pdb.set_trace()

    @classmethod
    def build_decoder(cls, cfg: Wav2Vec2Seq2SeqModConfig):
        from fairseq.models.bart import BARTModel
        bart = BARTModel.from_pretrained(cfg.load_pretrained_decoder_from, checkpoint_file='model.pt')
        # special_tokens = ['<triplet>', '<subj>', '<obj>', '<peop>', '</peop>', '<org>', '</org>', '<other>',
        #                     '</other>', '<loc>', '</loc>']
        # special_words = ["triplet", "subject", "object", "people", "organization", "location", "other"]

        # =============== 添加特殊tokes ====================
        special_tokens = ['<triplet>', '<subj>', '<obj>', "<entity>", "</entity>","<blank>"]  #+ ['<query_%d>'%i for i in range(30)]

        for word_s in special_tokens:
            bart.task.source_dictionary.add_symbol(word_s)

        # 更新embedding
        tgt_dict = bart.task.source_dictionary
        padding_idx = tgt_dict.pad()
        num_embeddings = len(tgt_dict)
        num_old, embed_dim =  bart.model.encoder.embed_tokens.weight.data.size()
        new_embs_layer = Embedding(num_embeddings, embed_dim, padding_idx)
        new_embs_layer.weight.data[:num_old] = bart.model.encoder.embed_tokens.weight.data
        bart.model.encoder.embed_tokens = new_embs_layer
        bart.model.decoder.embed_tokens = new_embs_layer

        # Query embedding
        # cls.Query_embs = Embedding(2, embed_dim)
        # cls.emb_indxs = Embedding(num_embeddings, num_embeddings)
        # cls.emb_indxs.weight.data = torch.eye(num_embeddings)
        # tokens = bart.encode('Lee Harvey Oswald Kennedy Kill')
        # tokens = bart.encode('<triplet> Lee Harvey Oswald <subj> Kennedy <obj> Kill')
        # sentence = bart.decode(tokens)
        # tokens_embs = []
        # for i, w in enumerate(special_words):
        #     tokens = bart.encode(w)
        #     tokens_emb = bart.model.encoder.embed_tokens(tokens)[1:-1]
        #     tokens_embs.append(tokens_emb.mean(0))
        #     if(i>2):
        #         tokens_embs.append(tokens_emb.mean(0))
        # new_embs = torch.stack(tokens_embs, dim=0)
        # bart.model.encoder.embed_tokens.weight.data = torch.cat((bart.model.encoder.embed_tokens.weight.data,new_embs),dim=0)
        #
        #
        # new_embs_layer.weight.data = bart.model.encoder.embed_tokens.weight.data
        # bart.model.encoder.embed_tokens = new_embs_layer
        # bart.model.decoder.embed_tokens = new_embs_layer

        # 定义Query的embedding层
        # text = "<triplet> AP <subj> NEW YORK <obj> OrgBased_In"
        # tensor([0,
        #         41552, 21237, 26151, 15698,     -> < triplet >
        #         1480,                           ->  AP
        #         28696, 10936, 267, 15698,       ->  < subj >
        #         5178, 4180,                     ->  NEW YORK
        #         28696, 46134, 15698,            ->  < obj >
        #         1793, 571, 20920, 1215, 1121,   ->  OrgBased_In
        #         2])
        # print(text)
        # text = bart.encode(text)
        # print(text)
        # exit(0) '<mask> Lee Harvey Oswald <mask> Kennedy <mask> Kill'
        # bart = BARTModel.from_pretrained("/workdir/liangzhang/SpeechRE/IWSLT/Pre-trained_models/BART-large", checkpoint_file='model.pt')
        # x = bart.fill_mask(['The cat <mask> on the <mask>.'], topk=3, beam=10)
        # tokens = bart.encode('<triplet> Lee Harvey Oswald <subj> Kennedy <obj> Kill')
        # tokens1 = bart.encode("<peop> Winter </peop> , 53 , a former <org>  UniversityYale </org> law professor who took the bench in " \
        # "<other> 1982 </other> , and <peop> Starr </peop> , a fellow appointee of President <peop> Reagan </peop> , " \
        # "are both known as judicial conservatives .")
        # sentence = bart.decode(tokens)
        # sentence1 = bart.decode(tokens1)
        # bart.task.source_dictionary.indices['</s>']
        #
        # last_layer_features1 = bart.extract_features(tokens1)
        # last_layer_features = bart.extract_features(tokens)
        # pdb.set_trace()
        # last_layer_features1 = bart.model.encoder(tokens, token_embeddings=torch.cat([tokens_emb, tokens_emb, tokens_emb], dim=0))
        # pdb.set_trace()

        # cls.decoder_tokenizer = bart
        # old_embed = decoder.embed_tokens
        # new_embed = embed_tokens    # nn.Embedding
        # old_embed_dim = old_embed.weight.data.shape[0]
        # # 向BART的embedding层中为额外添加的几个token添加embedding向量
        # new_embed.weight.data[:old_embed_dim, :] = old_embed.weight.data[:old_embed_dim, :]
        # decoder.embed_tokens = new_embed
        # 使用embedding层构建一个共享参数的输出投影层

        decoder = bart.model.decoder
        decoder.build_output_projection(decoder.args, tgt_dict, new_embs_layer)
        cls.bart = bart
        cls.encoder_text = bart.model.encoder

        # build generator using current args as well as any kwargs
        gen_args = copy.deepcopy(bart.cfg.generation)
        with open_dict(gen_args):
            gen_args.beam = 5
        cls.generator_trip = bart.task.build_generator(bart.models, gen_args)
        cls.Seq_compressor = Cross_Attention(copy.deepcopy(cls.bart.model.args))

        # audio_args = copy.deepcopy(cls.bart.model.args)
        # args.decoder_layers = 4

        # cls.statistics_toke()
        # pdb.set_trace()
        # decoder = TransformerDecoderMod(cfg, tgt_dict, embed_tokens)
        # if getattr(cfg, "load_pretrained_decoder_from", None):
        #     decoder = checkpoint_utils.load_pretrained_component_from_model(
        #         component=decoder, checkpoint=cfg.load_pretrained_decoder_from
        #     )
        #     logger.info(
        #         f"loaded pretrained decoder from: "
        #         f"{cfg.load_pretrained_decoder_from}"
        #     )
        return decoder

    def extract_triplets(self,text):
        triplets = []
        relation, subject, relation, object_ = '', '', '', ''
        text = text.strip()
        current = 'x'
        for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
            if token == "<triplet>":
                current = 't'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
                    relation = ''
                subject = ''
            elif token == "<subj>":
                current = 's'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
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
            triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})

        # 去重复
        triplets_str = []
        for t in triplets:
            triplet = 'head: ' + t['head'] + ' type: ' + t['type'] + ' tail: ' + t['tail']
            if triplet not in triplets_str:
                triplets_str.append(triplet)

        return triplets_str, triplets

    def extract_entities(self,text):
        entities = []
        text = text.strip()
        tokens = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split()
        L = len(tokens)

        i = 0
        entity_e = None
        entity = None
        while(i<L):
            if (tokens[i] in entity_start):
                entity = tokens[i]
                entity_e = entity_end[tokens[i]]

            elif(tokens[i] == entity_e):
                entity = entity + " " + tokens[i]

                if entity not in entities:
                    entities.append(entity)

                entity_e = None
                entity = None

            elif(entity is not None):
                entity = entity + " " + tokens[i]

            i += 1

        return entities

    def valid_step_RE2(self, sample):
        net_input = sample["net_input"]
        src_text, src_text_lengths, src_tokens = net_input['src_text'], net_input['src_text_lengths'], net_input['src_tokens']
        src_lengths, prev_output_tokens = net_input['src_lengths'], net_input['prev_output_tokens']

        # 输入生成
        bs, l = src_text.size()
        src_text_mask = torch.ones((bs, l + 20)).to(src_text)
        entity_mask = torch.tensor(entity_ids).to(src_text)
        bs = src_text.size(0)
        Mask_seq = torch.ones((bs, 20)).to(src_text) * 50264
        Max_l = -1
        for i in range(src_text.size(0)):
            temp = src_text[i][~torch.isin(src_text[i], entity_mask)]
            temp = torch.cat((temp, Mask_seq[i]), dim=-1)
            l = temp.size(0)
            src_text_mask[i][:l] = temp

            if (l > Max_l):
                Max_l = l

        src_text_mask = src_text_mask[:, :Max_l]
        src_text_mask_lengths = (src_text_mask[:, :Max_l] != 1).sum(-1)

        # 模型生成
        Sample_infer = {
            'id': sample['id'],
            'target': sample['target']
        }
        Sample_infer['net_input'] = {
            'src_tokens': src_text_mask,
            'src_lengths': src_text_mask_lengths,
            'token_embeddings': None,
            'prev_output_tokens': prev_output_tokens
        }
        prefix_tokens = sample['target'][:, :1]
        Output_temp = self.generator_trip._generate(Sample_infer, prefix_tokens)
        Output = []
        for i in range(len(Output_temp)):
            Output.append(Output_temp[i][0]['tokens'])

        # =================== 关系的解码 ===================
        triplet_p = triplet_r = 0
        triplet_total_p = triplet_total_r = 1e-5
        relation_p = relation_r = 0
        relation_total_p = relation_total_r = 1e-5
        entity_p = entity_r = 0
        entity_total_p = entity_total_r = 1e-5
        for i in range(len(Output)):
            pred, pred_list = self.extract_triplets(self.bart.decode(Output[i]))
            target, target_list = self.extract_triplets(self.bart.decode(sample['target'][i]))

            triplet_total_p += len(pred)
            triplet_total_r += len(target)
            for e in target:
                if e in pred:
                    triplet_r += 1
            for e in pred:
                if e in target:
                    triplet_p += 1

            Entity_pred, Entity_target = [], []
            Relation_pred, Relation_target = [], []
            for r in pred_list:
                if r['head'] not in Entity_pred:
                    Entity_pred.append(r['head'])
                if r['tail'] not in Entity_pred:
                    Entity_pred.append(r['tail'])
                if r['type'] not in Relation_pred:
                    Relation_pred.append(r['type'])
            for r in target_list:
                if r['head'] not in Entity_target:
                    Entity_target.append(r['head'])
                if r['tail'] not in Entity_target:
                    Entity_target.append(r['tail'])
                if r['type'] not in Relation_target:
                    Relation_target.append(r['type'])

            entity_total_p += len(Entity_pred)
            entity_total_r += len(Entity_target)
            for e in Entity_target:
                if e in Entity_pred:
                    entity_r += 1
            for e in Entity_pred:
                if e in Entity_target:
                    entity_p += 1

            relation_total_p += len(Relation_pred)
            relation_total_r += len(Relation_target)
            for r in Relation_target:
                if r in Relation_pred:
                    relation_r += 1
            for r in Relation_pred:
                if r in Relation_target:
                    relation_p += 1

        total_outout = {
            'entity_p': entity_r,
            'entity_r': entity_p,
            'entity_total_p': entity_total_p,
            'entity_total_r': entity_total_r,
            'relation_r': relation_r,
            'relation_p': relation_p,
            'relation_total_p': relation_total_p,
            'relation_total_r': relation_total_r,
            'triplet_p': triplet_p,
            'triplet_r': triplet_r,
            'triplet_total_p': triplet_total_p,
            'triplet_total_r': triplet_total_r
        }
        return total_outout

    def valid_step_RE(self,sample):
        net_input = sample["net_input"]
        src_text, src_text_lengths, src_tokens = net_input['src_text'], net_input['src_text_lengths'], net_input['src_tokens']
        src_lengths, prev_output_tokens = net_input['src_lengths'], net_input['prev_output_tokens']
        audi_encoder_out = self.encoder(src_text, src_tokens, src_lengths=src_lengths)

        # speech = audi_encoder_out["encoder_out"][0]
        # pad_mask_s = audi_encoder_out["encoder_padding_mask"][0]
        # audi_encoder_out["encoder_out"][0] = self.bart.model.encoder.Audio_forward(speech, pad_mask_s)

        # ========= 标注实体位置 ================
        entity_start_mask = (src_text == 50268)
        entity_end_mask = (src_text == 50269)

        src_text_tmp = torch.zeros_like(src_text)
        src_text_tmp = (src_text_tmp + entity_start_mask.to(src_text_tmp) - entity_end_mask.to(src_text_tmp))
        entity_mask_M = (src_text_tmp.cumsum(-1) > 0.5)

        # ========= 纯文本:去掉实体mask ================
        # b, l = src_text.size()
        # src_text_c = torch.ones_like(src_text)
        # src_text_mask = torch.ones((b, 2 * l)).to(src_text)
        # entity_mask = (torch.ones((b, 2 * l)).to(src_text) < 0.5)
        # entity_pos = torch.tensor(entity_ids).to(src_text)
        # max_len, max_len_c = -1, -1
        # for i in range(src_text.size(0)):
        #     mask = (~torch.isin(src_text[i], entity_pos))
        #     temp = src_text[i][mask]
        #     temp_b = entity_mask_M[i][mask]
        #
        #     l = temp.size(0)
        #     src_text_c[i][:l] = temp.clone()
        #     if (max_len_c < l):
        #         max_len_c = l
        #
        #     temp, temp_b = self.add_noise(temp, temp_b)
        #     l = temp.size(0)
        #     src_text_mask[i][:l] = temp
        #     entity_mask[i][:l] = temp_b
        #
        #     if (max_len < l):
        #         max_len = l
        #
        # src_text_mask = src_text_mask[:, :max_len]
        # entity_mask = entity_mask[:, :max_len]
        # src_text_len = (src_text_mask != 1).sum(-1).long()
        #
        # src_text_c = src_text_c[:, :max_len_c]
        # src_text_len_c = (src_text_c != 1).sum(-1).long()
        #
        # audi_out = audi_encoder_out['encoder_out'][0].transpose(0, 1)
        # audi_out_pad = audi_encoder_out['encoder_padding_mask'][0].to(src_text_len_c)
        # audi_pred = self.decoder.output_projection(audi_out)
        # audi_len = (1 - audi_out_pad).sum(-1)
        #
        # # audi_pred_I = F.gumbel_softmax(audi_pred, tau=1, hard=False, dim=-1)
        # audi_pred_I = F.softmax(audi_pred, dim=-1)
        #
        # suffix = torch.zeros_like(audi_pred_I[:,:1,:])
        # suffix_pad = torch.ones_like(audi_out_pad[:, :1]).to(audi_out_pad)
        #
        # audi_pred_I = torch.cat((audi_pred_I, suffix.clone()), dim=1)
        # audi_out_pad = torch.cat((audi_out_pad, suffix_pad.clone()), dim=1)
        #
        # suffix[:, :, 2:3] = suffix[:, :, 2:3] + 1
        # suffix_pad = torch.zeros_like(audi_out_pad[:, :1]).to(audi_out_pad)
        #
        # x_id = torch.arange(audi_pred_I.size(0)).to(src_text)
        # audi_pred_I[x_id, audi_len] = suffix[:,0,:]
        # audi_out_pad[x_id, audi_len] = suffix_pad[:,0]
        #
        # prefix = torch.zeros_like(audi_pred_I[:,:1,:])
        # prefix[:, :, :1] = prefix[:, :, :1] + 1
        # prefix_pad = torch.zeros_like(audi_out_pad[:, :1]).to(audi_out_pad)
        #
        # audi_pred_I = torch.cat((prefix, audi_pred_I), dim=1)
        # audi_out_pad = torch.cat((prefix_pad, audi_out_pad), dim=1)
        # audi_len_I = audi_len + 2
        #
        # W = self.decoder.output_projection.weight.detach()
        # audi_embs = torch.matmul(audi_pred_I, W)
        #
        # audi_tokens = audi_pred_I.max(-1)[-1]
        # pad = torch.ones_like(audi_tokens)
        # audi_tokens = (1 - audi_out_pad) * audi_tokens + audi_out_pad * pad
        # audi_tokens = [audi_tokens, audi_token_embs]

        # ========= 全mask输入 ================
        Qerys = torch.ones_like(src_text) * 50264
        mask = (src_text != 1).to(src_text)  # 不为1的地方进行mask
        Qerys = (mask * Qerys + (1 - mask) * src_text).to(src_text)
        padding_mask = (Qerys == 1)
        # padding_mask = (torch.zeros_like(Qerys) > 0.5).bool()

        token_embs = {
            'encoder_padding_mask' : [padding_mask],
            'encoder_out' : [Qerys],
        }

        # ============toke生成性能===========
        # for rate in [0.3, 0.6, 1.0]:
        #     encoder_out = self.bart.model.encoder(
        #         src_text_mask,
        #         src_lengths=src_text_len,
        #         token_embeddings=token_embs,
        #         # return_all_hiddens=True,
        #         type_embs=Type_embs1,
        #         # compression_F= self.encoder.compression_forward
        #         # compression_F=compression_F
        #     )
        #     enc_out = encoder_out["encoder_out"][0].transpose(0, 1)
        #     enc_pred = self.decoder.output_projection(self.encoder.Output_layer(enc_out[:, -src_text.size(-1):]))
        #     enc_pred = F.softmax(enc_pred, dim=-1)
        #     enc_pred_val, enc_pred_id = enc_pred.max(-1)
        #
        #     enc_out_p = (mask * enc_pred_val + (1 - mask) * (-1)).to(enc_pred_val)  # 已经被预测的地方是-1，需要被预测的地方为概率
        #     text_lengths = mask.sum(-1)  # 需要被预测的token数
        #     Top_Ks = []
        #     for j in range(b):
        #         K = int(text_lengths[j] * rate)
        #         top_k = enc_out_p[j].topk(K, dim=-1)[0][-1]
        #         Top_Ks.append(top_k)
        #     Top_Ks = torch.tensor(Top_Ks).unsqueeze(-1).to(enc_out_p)
        #     mask_top = (enc_out_p >= Top_Ks).long()
        #
        #     token_embs['encoder_out'][0] = mask_top * enc_pred_id + (1 - mask_top) * token_embs['encoder_out'][0]
        #     mask = mask - mask_top

        # ==============解码输入=====================
        enc_pred = self.decoder.output_projection(audi_encoder_out["encoder_out"][0]).transpose(0,1).max(-1)[-1]
        enc_pred_len = (1 - audi_encoder_out["encoder_padding_mask"][0].long()).sum(-1)

        Sample_infer = {
            'id': sample['id'],
            'target': sample['target']
        }
        Sample_infer['net_input'] = {
            # 'src_tokens': src_text,
            # 'src_lengths': src_text_lengths,
            'src_tokens': enc_pred,
            'src_lengths': enc_pred_len,
            # 'token_embeddings': token_embs,
            'prev_output_tokens': prev_output_tokens
        }
        prefix_tokens = sample['target'][:, :1]
        Output_temp = self.generator_trip._generate(Sample_infer, prefix_tokens, encoder_outs=[audi_encoder_out])
        # Output_temp = self.generator_trip._generate(Sample_infer, prefix_tokens)
        Output = []
        for i in range(len(Output_temp)):
            Output.append(Output_temp[i][0]['tokens'])

        # =================== 关系的解码 ===================
        triplet_p = triplet_r = 0
        triplet_total_p = triplet_total_r = 1e-5
        relation_p = relation_r = 0
        relation_total_p = relation_total_r = 1e-5
        entity_p = entity_r = 0
        entity_total_p = entity_total_r = 1e-5
        for i in range(len(Output)):
            pred, pred_list = self.extract_triplets(self.bart.decode(Output[i]))
            target, target_list = self.extract_triplets(self.bart.decode(sample['target'][i]))
            triplet_total_p += len(pred)
            triplet_total_r += len(target)
            for e in target:
                if e in pred:
                    triplet_r += 1
            for e in pred:
                if e in target:
                    triplet_p += 1

            Entity_pred, Entity_target = [], []
            Relation_pred,  Relation_target = [], []
            for r in pred_list:
                if r['head'] not in Entity_pred:
                    Entity_pred.append(r['head'])
                if r['tail'] not in Entity_pred:
                    Entity_pred.append(r['tail'])
                if r['type'] not in Relation_pred:
                    Relation_pred.append(r['type'])
            for r in target_list:
                if r['head'] not in Entity_target:
                    Entity_target.append(r['head'])
                if r['tail'] not in Entity_target:
                    Entity_target.append(r['tail'])
                if r['type'] not in Relation_target:
                    Relation_target.append(r['type'])

            entity_total_p += len(Entity_pred)
            entity_total_r += len(Entity_target)
            for e in Entity_target:
                if e in Entity_pred:
                    entity_r += 1
            for e in Entity_pred:
                if e in Entity_target:
                    entity_p += 1

            relation_total_p += len(Relation_pred)
            relation_total_r += len(Relation_target)
            for r in Relation_target:
                if r in Relation_pred:
                    relation_r += 1
            for r in Relation_pred:
                if r in Relation_target:
                    relation_p += 1

        # encoder_out = self.bart.model.encoder(
        #     src_text_mask,
        #     src_lengths=src_text_len,
        #     token_embeddings=token_embs,
        #     # return_all_hiddens=True,
        #     type_embs=Type_embs1,
        #     # compression_F= self.encoder.compression_forward
        #     # compression_F=compression_F
        # )
        # enc_out = encoder_out["encoder_out"][0].transpose(0, 1)
        # # enc_pred = self.decoder.output_projection(self.encoder.Output_layer(enc_out[:, -src_text.size(-1):]))
        # enc_pred = self.encoder.Output_layer(enc_out[:, -src_text.size(-1):])
        # enc_pred = enc_pred.max(-1)[-1]
        # mask = (src_text != 1)
        # toke_t = mask.sum()
        # toke_p = (enc_pred == src_text)[mask].sum()

        total_outout = {
            'entity_p': entity_r,
            'entity_r': entity_p,
            'entity_total_p': entity_total_p,
            'entity_total_r': entity_total_r,
            'relation_r': relation_r,
            'relation_p': relation_p,
            'relation_total_p': relation_total_p,
            'relation_total_r': relation_total_r,
            'triplet_p': triplet_p,
            'triplet_r': triplet_r,
            'triplet_total_p': triplet_total_p,
            'triplet_total_r': triplet_total_r,
            'token_p': 0.0,
            'token_t': 1.0
        }
        return total_outout

    def BART_forward(self, src_text, src_text_lengths, prev_output_tokens, entity_mask, encoder_out_s=None):
        # if encoder_out is None:
        encoder_out = self.bart.model.encoder(
            src_text,
            src_lengths=src_text_lengths,
            # token_embeddings= token_embs,
            return_all_hiddens=True,
            # compression_F= self.encoder.compression_forward
            # compression_F=compression_F
        )

        # Alig_input={
        #     "speech":encoder_out_s["encoder_embedding"][0],
        #     "text":encoder_out['encoder_states'][-5],
        #     "pad_mask_s":encoder_out_s['encoder_padding_mask'][0],
        #     "pad_mask_t":encoder_out['encoder_padding_mask'][0],
        #     "entity_mask":entity_mask,
        #     "Audio_forward":self.encoder.Audio_forward
        # }
        Alig_input={
            "speech":encoder_out_s["encoder_out"][0],
            "text":encoder_out['encoder_out'][0],
            "pad_mask_s":encoder_out_s['encoder_padding_mask'][0],
            "pad_mask_t":encoder_out['encoder_padding_mask'][0],
            "entity_mask":entity_mask,
        }
        x_s, x_t, mask_t = self.bart.model.encoder.Alignment_forward(**Alig_input)
        en_out_s = [encoder_out_s["encoder_out"][0].transpose(0, 1),  x_s.transpose(0, 1)]
        en_out_t = [encoder_out['encoder_out'][0].transpose(0, 1), x_t.transpose(0, 1)]

        # 文本编码器输出
        de_out = []
        decoder_out1 = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            # features_only=features_only,
            # alignment_layer=alignment_layer,
            # alignment_heads=alignment_heads,
            # src_lengths=src_text_lengths,
            # return_all_hiddens=return_all_hiddens,
        )
        de_out.append(decoder_out1)

        encoder_out["encoder_out"][0] = x_t
        decoder_out2 = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            # features_only=features_only,
            # alignment_layer=alignment_layer,
            # alignment_heads=alignment_heads,
            # src_lengths=src_text_lengths,
            # return_all_hiddens=return_all_hiddens,
        )
        de_out.append(decoder_out2)

        decoder_out3 = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out_s,
            # features_only=features_only,
            # alignment_layer=alignment_layer,
            # alignment_heads=alignment_heads,
            # src_lengths=src_text_lengths,
            # return_all_hiddens=return_all_hiddens,
        )
        de_out.append(decoder_out3)

        # encoder_out_s["encoder_out"][0] = x_s
        # decoder_out4 = self.decoder(
        #     prev_output_tokens,
        #     encoder_out=encoder_out_s,
        #     # features_only=features_only,
        #     # alignment_layer=alignment_layer,
        #     # alignment_heads=alignment_heads,
        #     # src_lengths=src_text_lengths,
        #     # return_all_hiddens=return_all_hiddens,
        # )
        # de_out.append(decoder_out4)

        Output = {
            "en_out_t": en_out_t,
            "en_out_s": en_out_s,
            "de_out": de_out,
            "att_e": encoder_out["att_score"][-1],
            "att_d": decoder_out3[1]['attn'][-1],
            'mask_t': mask_t
        }

        # Output = {
        #     "en_out_t": [encoder_out['encoder_states'][-5].transpose(0, 1),
        #                  encoder_out['encoder_out'][0].transpose(0, 1),
        #                  x_t.transpose(0, 1)],
        #     "en_out_s": [encoder_out_s["encoder_embedding"][0].transpose(0, 1),
        #                  encoder_out_s["encoder_out"][0].transpose(0, 1),
        #                  x_s.transpose(0, 1)],
        #     "de_out": [decoder_out1, decoder_out2],
        #     "att_e": encoder_out["att_score"][-1],
        #     "att_d": decoder_out2[1]['attn'][-1],
        # }

        return Output


    def Updata_W(self, src_text, src_text_rep):
        # 更新词汇表权重
        src_text = src_text.detach()
        src_text_rep = src_text_rep.detach()

        d = src_text_rep.size(-1)
        l1 = self.W.size(0)
        col = src_text.view(-1, 1)
        l2 = col.size(0)
        src_text_rep_temp = src_text_rep.contiguous().view(-1, d)

        col = col.expand(-1, l1)
        row = torch.tensor([i for i in range(l1)]).to(col).unsqueeze(0).expand(l2,-1)
        c_r = (col == row).to(src_text_rep_temp).transpose(0, 1)
        new_W = torch.matmul(c_r, src_text_rep_temp)
        S = c_r.sum(-1).to(new_W).unsqueeze(-1) + 1e-5
        new_W = new_W / S
        alph = (S > 0.5) * 0.1
        self.W = self.W.to(new_W)
        self.W = alph * new_W + (1 - alph) * self.W

    def loss_fn(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        # return 2 - 2 * (x * y).sum(dim=-1)
        return (x * y).sum(dim=-1)


    def consistence_loss1(self, x, y, x_mask, y_mask):
        l_x = x.size(1)
        l_y = y.size(1)
        y_x_mask = x_mask.unsqueeze(1).expand(-1, l_y, -1).to(x) * (-1e8)
        x_x_mask = x_mask.unsqueeze(1).expand(-1, l_x, -1).to(x) * (-1e8)
        x_y_mask = y_mask.unsqueeze(1).expand(-1, l_x, -1).to(x) * (-1e8)
        y_y_mask = y_mask.unsqueeze(1).expand(-1, l_y, -1).to(x) * (-1e8)

        # x_n = F.normalize(x, dim=-1, p=2)
        # y_n = F.normalize(y, dim=-1, p=2)
        x_n = x
        y_n = y
        x_y = F.softmax(torch.matmul(x_n, y_n.transpose(-2, -1)) + x_y_mask, dim=-1)
        x_x = F.softmax(torch.matmul(x_n, x_n.transpose(-2, -1)) + x_x_mask, dim=-1)

        y_x = F.softmax(torch.matmul(y_n, x_n.transpose(-2, -1)) + y_x_mask, dim=-1)
        y_y = F.softmax(torch.matmul(y_n, y_n.transpose(-2, -1)) + y_y_mask, dim=-1)

        x_y_p = torch.matmul(x_y, y)
        x_x_p = torch.matmul(x_x, x)
        y_x_p = torch.matmul(y_x, x)
        y_y_p = torch.matmul(y_y, y)

        Mask_x = (x_mask < 0.5)
        Loss = F.pairwise_distance(x_y_p[Mask_x], x_x_p[Mask_x], p=2).sum()
        # Loss = self.loss_fn(x_y_p[Mask_x], x_x_p[Mask_x]).sum()
        # Loss = torch.sqrt(((x_y_p[Mask_x] - x_x_p[Mask_x]) ** 2).sum(-1)).sum()
        # Loss = F.kl_div(F.log_softmax(x_y_p[Mask_x], dim=-1), F.softmax(x_x_p[Mask_x], dim=-1), reduction='sum')
        # Loss += F.kl_div(F.log_softmax(x_x_p[Mask_x], dim=-1), F.softmax(x_y_p[Mask_x], dim=-1), reduction='sum')

        Mask_y = (y_mask < 0.5)
        Loss += F.pairwise_distance(y_x_p[Mask_y], y_y_p[Mask_y], p=2).sum()
        # Loss += self.loss_fn(y_x_p[Mask_y], y_y_p[Mask_y]).sum()
        # Loss += torch.sqrt(((y_x_p[Mask_y] - y_y_p[Mask_y]) ** 2).sum(-1)).sum()
        # Loss += F.kl_div(F.log_softmax(y_x_p[Mask_y], dim=-1), F.softmax(y_y_p[Mask_y], dim=-1), reduction='sum')
        # Loss += F.kl_div(F.log_softmax(y_y_p[Mask_y], dim=-1), F.softmax(y_x_p[Mask_y], dim=-1), reduction='sum')

        return Loss/12.0

    def consistence_loss(self, x, y, x_mask, y_mask):
        x = x.transpose(0, 1)
        y = y.transpose(0, 1)
        x_q = self.encoder.Seq_compressor([x], x_mask)
        y_q = self.encoder.Seq_compressor([y], y_mask)
        Loss = self.loss_fn(x_q, y_q).sum()
        # Loss = -1 * F.pairwise_distance(x_q, y_q, p=2).sum()
        # Loss = -1 * F.kl_div(F.log_softmax(x_q, dim=-1), F.softmax(y_q, dim=-1), reduction='sum')
        # Loss += -1 * F.kl_div(F.log_softmax(y_q, dim=-1), F.softmax(x_q, dim=-1), reduction='sum')
        return Loss

    def add_noise(self, input, input_mask, noise_rate=0.2):
        temp = [[d.item()] for d in input]
        temp_mask = [[d.item()] for d in input_mask]
        input_t = [temp[0][0]]
        input_mask_t = [temp_mask[0][0]]
        for i in range(1, len(temp) - 1):
            if random.random()<=0.2:
                temp[i].append(temp[i][0])
                temp_mask[i].append(temp_mask[i][0])
            if random.random()<=0.2:
                temp[i].append(50270)
                temp_mask[i].append(False)
            if random.random()<=0.2:
                temp[i][0] = 50264

            input_t.extend(temp[i])
            input_mask_t.extend(temp_mask[i])
        input_t.extend(temp[-1])
        input_mask_t.extend(temp_mask[-1])
        input_t = torch.tensor(input_t).to(input)
        input_mask_t = torch.tensor(input_mask_t).to(input_mask)

        return input_t, input_mask_t

    def get_entity_pos(self, input, start_id, end_id):
        entity_start_mask = (input == start_id)
        entity_end_mask = (input == end_id)

        src_text_tmp = torch.zeros_like(input)
        src_text_tmp = (src_text_tmp + entity_start_mask.to(src_text_tmp) - entity_end_mask.to(src_text_tmp))
        src_text_tmp = src_text_tmp.cumsum(-1) - entity_start_mask.to(src_text_tmp)

        entity_mask_M = (src_text_tmp > 0.5)
        return entity_mask_M

    def get_CTC_input1(self, src_text, src_text_lengths, enc_out_t, enc_out_s, audio_len):
        # 取出bank
        x_id = torch.arange(src_text.size(0)).to(src_text)
        W_b = enc_out_t[x_id, src_text_lengths].detach().unsqueeze(1).transpose(-1, -2)

        # 去掉停用token
        stop_token = torch.tensor(entity_ids).to(src_text)
        max_len = -1
        src_text_lengths_p = []
        for i in range(src_text.size(0)):
            mask = (~torch.isin(src_text[i], stop_token))
            temp = src_text[i][mask]
            temp_b = enc_out_t[i][mask]

            l = temp.size(0)
            src_text_lengths_p.append(l)

            src_text[i][:l] = temp
            enc_out_t[i][:l] = temp_b

        src_text_lengths = torch.tensor(src_text_lengths_p).to(src_text_lengths)
        max_len = src_text_lengths.max()
        src_text = src_text[:, :max_len]
        enc_out_t = enc_out_t[:, :max_len]

        mask = ((src_text != 0) & (src_text != 1) & (src_text != 2) & (src_text != 50270) & (src_text != 50269) & (src_text != 50268))
        b, l = src_text.size()
        W = enc_out_t[mask].detach().transpose(-1, -2)
        # W_b = enc_out_t[:, 0].detach().unsqueeze(1).transpose(-1, -2)

        CTC_label = torch.tensor([[i for i in range(l)]] * b).to(src_text)
        CTC_label_b = torch.tensor([0] + src_text_lengths.tolist()[:-1]).to(src_text).cumsum(dim=-1)
        CTC_label = CTC_label + CTC_label_b.unsqueeze(-1)

        audio_pred = torch.cat((torch.matmul(enc_out_s, W), torch.matmul(enc_out_s, W_b)), dim=-1)

        return [audio_pred, audio_len, CTC_label, src_text_lengths]

    def get_CTC_input2(self, src_text, src_text_lengths, enc_out_t, enc_out_s, audio_len):
        mask = (src_text != 1)
        b, l = src_text.size()
        x_id = torch.arange(src_text.size(0)).to(src_text)
        W_b = enc_out_t[x_id, src_text_lengths].detach().unsqueeze(1).transpose(-1, -2)

        W = enc_out_t[:, :l][mask].detach().transpose(-1, -2)

        CTC_label = torch.tensor([[i for i in range(l)]] * b).to(src_text)
        CTC_label_b = torch.tensor([0] + src_text_lengths.tolist()[:-1]).to(src_text).cumsum(dim=-1)
        CTC_label = CTC_label + CTC_label_b.unsqueeze(-1)

        audio_pred = torch.cat((torch.matmul(enc_out_s, W), torch.matmul(enc_out_s, W_b)), dim=-1)

        return [audio_pred, audio_len, CTC_label, src_text_lengths]

    def get_CTC_input(self, src_text, src_text_lengths, enc_out_t, enc_out_s, audio_len):
        mask = (src_text != 1)
        b, l = src_text.size()

        x_id = torch.arange(src_text.size(0)).to(src_text)
        W_b = enc_out_t[x_id, src_text_lengths].detach().unsqueeze(1).transpose(-1, -2)

        enc_out_t = enc_out_t[:, :l].detach()
        W = []
        Neg_examples = self.encoder.projection.weight.detach()[:-1]
        L = Neg_examples.size(0)
        token_id = torch.arange(L).to(src_text)
        for i in range(b):
            mask1 = (~torch.isin(token_id, src_text[i]))
            Neg_example = Neg_examples[mask1]
            Pos_example = enc_out_t[i][mask[i]]
            W.append(torch.cat((Pos_example, Neg_example, Neg_example), dim=0)[:l + L])
        W = torch.stack(W, dim=0).transpose(-1, -2)

        CTC_label = torch.tensor([[i for i in range(l)]] * b).to(src_text)
        audio_pred = torch.cat((torch.matmul(enc_out_s, W), torch.matmul(enc_out_s, W_b)), dim=-1)

        # 更新负例的损失
        Predict = self.encoder.projection(enc_out_t)
        Loss = self.CE_loss(Predict[mask],src_text[mask])

        return [audio_pred, audio_len, CTC_label, src_text_lengths], Loss

    def decoder_alignment_loss(self, decoder_out0, decoder_out1, mask):
        # 解码器对齐
        Loss2 = F.kl_div(F.log_softmax(decoder_out0[0][mask], dim=-1),F.softmax(decoder_out1[0][mask].detach(), dim=-1), reduction='sum')
        Loss1 = 0
        tem = [1.0, 0.5, 0.25]
        for j, k in enumerate([12, 8, 4]):
            h_out1 = decoder_out0[1]['inner_states'][k].transpose(0, 1)[mask]
            h_out2 = decoder_out1[1]['inner_states'][k].transpose(0, 1)[mask].detach()
            Loss1 += tem[j] * F.pairwise_distance(h_out1, h_out2, p=2).sum()
            # Loss1 += tem[j] * self.loss_fn(h_out1, h_out2).sum()

        Loss1 = Loss1 / 6.0
        return Loss2

    def forward(self, sample, step):
        # src_text, src_text_lengths, src_tokens, src_lengths, prev_output_tokens, ** kwargs
        net_input = sample["net_input"]
        src_text, src_text_lengths, src_tokens = net_input['src_text'], net_input['src_text_lengths'], net_input['src_tokens']
        src_lengths, prev_output_tokens = net_input['src_lengths'], net_input['prev_output_tokens']
        audi_encoder_out = self.encoder(src_text, src_tokens, src_lengths=src_lengths)

        # ========= 标注实体位置 ================
        entity_mask_M = self.get_entity_pos(src_text, start_id=50268, end_id=50269)

        # ========= 纯文本:去掉实体mask ================
        b, l = src_text.size()
        src_text_c = torch.ones_like(src_text)
        src_text_mask = torch.ones((b, 2 * l)).to(src_text)
        entity_mask = (torch.ones((b, 2 * l)).to(src_text) < 0.5)
        entity_pos = torch.tensor(entity_ids).to(src_text)
        max_len, max_len_c = -1, -1
        for i in range(src_text.size(0)):
            mask = (~torch.isin(src_text[i], entity_pos))
            temp = src_text[i][mask]
            temp_b = entity_mask_M[i][mask]

            l = temp.size(0)
            src_text_c[i][:l] = temp.clone()
            if (max_len_c < l):
                max_len_c = l

            temp, temp_b = self.add_noise(temp, temp_b)
            l = temp.size(0)
            src_text_mask[i][:l] = temp
            entity_mask[i][:l] = temp_b

            if (max_len < l):
                max_len = l

        src_text_mask = src_text_mask[:, :max_len]
        entity_mask = entity_mask[:, :max_len]
        src_text_len = (src_text_mask != 1).sum(-1).long()

        src_text_c = src_text_c[:, :max_len_c]
        src_text_len_c = (src_text_c != 1).sum(-1).long()

        # 添加Bank到输出到最后面
        x_id = torch.arange(src_text.size(0)).to(src_text)
        src_text_p = torch.cat((src_text, torch.ones_like(src_text[:, :1])), dim=-1)
        src_text_p[x_id, src_text_lengths] = src_text_p[x_id, src_text_lengths] * 50270
        src_text_lengths_p = src_text_lengths + 1
        entity_mask = torch.cat((entity_mask_M, entity_mask_M[:, -1:]), dim=-1)
        # entity_mask = entity_mask_M
        BART_Output = self.BART_forward(src_text_p, src_text_lengths_p, prev_output_tokens, entity_mask, encoder_out_s=audi_encoder_out)

        # CTC计算需要的输入
        audio_len = (1 - audi_encoder_out['encoder_padding_mask'][0].long()).sum(-1)
        CTC_inputs, Loss0 = [], 0
        for k in range(1):
            enc_out_t, enc_out_s =  BART_Output["en_out_t"][k], BART_Output["en_out_s"][k]
            # CTC_input = self.get_CTC_input2(src_text.clone(), src_text_lengths.clone(), enc_out_t.clone(), enc_out_s.clone(), audio_len.clone())
            # CTC_inputs.append(CTC_input)
            # CTC_input = self.get_CTC_input1(src_text_p, src_text_lengths, enc_out_t, enc_out_s, audio_len)
            # enc_out_t, enc_out_s = BART_Output["en_out_t"][k+1], BART_Output["en_out_s"][k+1]
            CTC_input, loss = self.get_CTC_input(src_text, src_text_lengths, enc_out_t, enc_out_s, audio_len)
            CTC_inputs.append(CTC_input)
            Loss0 += loss

        # 编码器对齐
        # enc_out_s1, enc_out_s2 = BART_Output["en_out_s"][1], BART_Output["en_out_s"][2]
        # mask = (audi_encoder_out['encoder_padding_mask'][0].long() < 0.5)
        # Loss0 = F.pairwise_distance(enc_out_s1[mask], enc_out_s2[mask], p=2).sum()
        enc_out_t1, enc_out_t2 = BART_Output["en_out_t"][1], BART_Output["en_out_t"][0].detach()
        mask_t = BART_Output['mask_t']
        Loss1 = F.pairwise_distance(enc_out_t1[mask_t], enc_out_t2[mask_t], p=2).sum()/3.0
        # Loss1 = self.loss_fn(enc_out_t1[entity_mask], enc_out_t2[entity_mask]).sum()

        mask = ((src_text_p != 1) & (src_text_p != 50270))
        Speech, Text = BART_Output["en_out_s"][0], BART_Output["en_out_t"][0].detach()
        Speech_mask, Text_mask = audi_encoder_out['encoder_padding_mask'][0].to(Speech), mask
        Loss1 += self.consistence_loss(Speech, Text, Speech_mask.bool(), Text_mask.bool())

        # 解码器对齐
        mask = ((sample['target'] != 1) & (sample['target'] != 0) & (sample['target'] != 2))
        Loss = []
        for k in range(2):
            Loss.append(self.decoder_alignment_loss(BART_Output["de_out"][k+1], BART_Output["de_out"][0], mask))

        Loss = Loss[1] + Loss[0]
        return [BART_Output["de_out"][2], BART_Output["de_out"][1], BART_Output["de_out"][0]], CTC_inputs, Loss, Loss1, Loss0

    def freeze_model(self, model):
        for n, p in model.named_parameters():
            p.requires_grad = False

    def unfreeze_model(self, model):
        for n, p in model.named_parameters():
            p.requires_grad = True

    def freeze_blocks(self, cfg: Wav2Vec2Seq2SeqModConfig):
        regex_to_freeze = re.compile(
            "|".join([BLOCKS2REGEX[b] for b in cfg.freeze_layers.split(',')])
        )
        for n, p in self.named_parameters():
            if re.match(regex_to_freeze, n):
                # print("\t\t   ",n)
                p.requires_grad = False
            # else:
            #     print(n,":",p.requires_grad)

        # for n, p in self.bart.model.encoder.named_parameters():
        #     n = 'encoder.' + n
        #     if re.match(regex_to_freeze, n):
        #         p.requires_grad = False
        #     else:
        #         print(n)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

# from fairseq.models import FairseqDecoder, FairseqEncoder
# from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
# special_tokens_list = ['<triplet>','<subj>','<obj>','<peop>', '</peop>','<org>', '</org>','<other>','</other>', '<loc>','</loc>']
# class Bart_Decoder(FairseqDecoder):
#     def __init__(self, model_name_or_path, dictionary = None,):
#         super().__init__(dictionary)
#         self.config = AutoConfig.from_pretrained(
#             model_name_or_path,
#             decoder_start_token_id=0,
#             early_stopping=False,
#             no_repeat_ngram_size=0,
#             # dropout=conf.dropout,
#             forced_bos_token_id=None,
#         )
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
#         self.tokenizer.add_tokens(special_tokens_list, special_tokens=True)
#         self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=self.config)
#         self.model.resize_token_embeddings(len(self.tokenizer))


import torch.nn as nn
from typing import List, Dict, Optional, Any
from fairseq import utils


class Wav2VecEncoderMod(Wav2VecEncoder):
    """
    Modification of the Wav2VecEncoder

    It modifies it to work with the speech_to_text pipeline.
    Moreover, it includes the adapter and length adaptor modules.
    """
    def __init__(self, cfg: Wav2Vec2Seq2SeqModConfig, tgt_dict=None):
        super().__init__(cfg, tgt_dict)
        # 额外的长度压缩器
        self.len_adaptor = Conv1dSubsampler(
            cfg.decoder_embed_dim,
            cfg.len_adaptor_channels,
            cfg.decoder_embed_dim,
            [int(k) for k in cfg.len_adaptor_kernel_sizes.split(",")],
        )
        # self.Q_Former = MultiheadAttention(embed_dim=cfg.decoder_embed_dim, num_heads=16)
        # self.cfg = cfg
        # self.Query_n = 30
        # self.len_adaptor = Conv1dSubsampler(
        #     cfg.decoder_embed_dim,
        #     cfg.len_adaptor_channels,
        #     cfg.decoder_embed_dim,
        # )
        # from fairseq.models.bart import BARTModel
        # self.bart = BARTModel.from_pretrained("/data/wangguitao/IWSLT/Pre-trained_models/BART-large",
        #                                  checkpoint_file='model.pt')
        # self.bart.eval()
        # encoder.Query_embs = cls.Query_embs
        # encoder.coss_att_layer = copy.deepcopy(decoder.layers[0])
        # encoder.coss_att_layer.compression_forward

    def Generate_len_adaptor(self):
        self.len_adaptor = Conv1dSubsampler(
            self.cfg.decoder_embed_dim,
            self.cfg.len_adaptor_channels,
            self.cfg.decoder_embed_dim,
            [int(k) for k in self.cfg.len_adaptor_kernel_sizes.split(",")],
        )

    def Audio_forward(self, speech, pad_mask_s):
        x = speech
        for l, layer in enumerate(self.text_adapter):
            x = layer(x, encoder_padding_mask=pad_mask_s)
        return x

    def forward(self, src_text, src_tokens, src_lengths, **kwargs):
        encoder_out = super().forward(
            source=src_tokens,
            padding_mask=lengths_to_padding_mask(src_lengths),
            tbc=False,
            **kwargs
        )
        encoder_out["encoder_padding_mask"] = encoder_out.pop("padding_mask")
        encoder_out["encoder_out"], lengths = self.len_adaptor(
            encoder_out["encoder_out"],
            (~encoder_out["encoder_padding_mask"]).sum(dim=1)
        )
        encoder_out["encoder_padding_mask"] = lengths_to_padding_mask(lengths)
        speech = encoder_out["encoder_out"]
        pad_mask_s = encoder_out["encoder_padding_mask"]
        # x = self.Audio_forward(speech, pad_mask_s)

        return {
            "encoder_out": [speech],  # T x B x C
            "encoder_padding_mask": [pad_mask_s],  # B x T
            "encoder_embedding": [speech],  # B x T x C
            "encoder_states": [speech],  # List[T x B x C]
            "att_score": [],
            "src_tokens": [],
            "src_lengths": [],
        }

        # Out = self.seq_compressor(src_text, encoder_out["encoder_out"], encoder_out["encoder_padding_mask"])
        # Out["encoder_out"] = [encoder_out["encoder_out"]]
        # Out["encoder_padding_mask"] = [encoder_out["encoder_padding_mask"]]
        # encoder_out["encoder_out"], encoder_out["encoder_padding_mask"] = self.seq_compressor(src_text,
        #                                                                                       encoder_out["encoder_out"],
        #                                                                                       encoder_out["encoder_padding_mask"])
        #
        # encoder_out["encoder_embedding"] = []
        # encoder_out["encoder_states"] = []
        # encoder_out["att_score"] = []
        # encoder_out["src_tokens"] = []
        # encoder_out["src_lengths"] = []

        # for k, v in encoder_out.items():
        #     print(k)
        #     print(v.shape)
        #     print(v)
        # pad = nn.ZeroPad2d((0, 0, 0, 1024 - encoder_out["encoder_out"].shape[1]))
        # encoder_out["encoder_out"] = pad(encoder_out["encoder_out"])
        # for k, v in encoder_out.items():
        #     print(k)
        #     print(v.shape)
        #     print(v)
        # encoder_out["encoder_padding_mask"] = encoder_out.pop("padding_mask")
        # encoder_out["encoder_out"] = self.len_adapter(encoder_out["encoder_out"])
        # encoder_out["encoder_padding_mask"] = lengths_to_padding_mask(
        #     torch.tensor([encoder_out["encoder_out"].shape[0]] * encoder_out["encoder_out"].shape[1]).cuda()
        # )
        # encoder_out["padding_mask"] = encoder_out["encoder_padding_mask"].transpose(0, 1)
        # for k, v in encoder_out.items():
        #     print(k)
        #     print(v.shape)
        #     print(v)
        # return {k: [v] for k, v in encoder_out.items()}
        # return Out

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out, new_order):
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
        }



from torch.autograd import Function
from typing import Any, Optional, Tuple
import torch.nn as nn
import torch
import torch.nn.functional as F
class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

class GRL_Layer(nn.Module):
    def __init__(self):
        super(GRL_Layer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class Cross_Attention(nn.Module):
    def __init__(self, args, memory_num=20):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.attention_heads = args.encoder_attention_heads
        self.memory_num = memory_num
        self.grl = GRL_Layer()

        self.memory_module = nn.Parameter(torch.FloatTensor(self.memory_num, 1,self.embed_dim))
        nn.init.uniform_(self.memory_module)
        self.memory_attention = MultiheadAttention(self.embed_dim, self.attention_heads)

    def forward(self, keys, kay_padding_mask):
        T, B, C = keys[0].size()
        x = self.memory_module.repeat(1, B, 1)

        for key in keys:
            key = self.grl(key)
            x, attn_score = self.memory_attention(
                query=x,
                key=key,
                value=key,
                key_padding_mask=kay_padding_mask,
                need_weights=True
            )
        return x


class Cross_Self_layer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.attention_heads = args.encoder_attention_heads

        self.dropout_module = FairseqDropout(args.dropout)
        self.self_attn = MultiheadAttention(self.embed_dim, self.attention_heads, dropout=args.attention_dropout)

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )

        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.encoder_attn = MultiheadAttention(self.embed_dim, self.attention_heads, dropout=args.attention_dropout)
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = nn.Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = nn.Linear(args.decoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)

    def forward(self, x, encoder_out, encoder_padding_mask, self_attn_padding_mask, self_attn_mask=None):
        residual = x
        x, attn = self.encoder_attn(
            query=x,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            need_weights=True,
            static_kv=True
        )
        x = self.dropout_module(x) + residual
        x = self.encoder_attn_layer_norm(x)

        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=True,
            attn_mask=None,  # self_attn_mask
        )
        x = self.dropout_module(x) + residual
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x) + residual
        x = self.final_layer_norm(x)

        return x, attn

class Cross_Self_Att(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.embed_tokens = Embedding(1024, self.embed_dim, 1)
        self.layers = nn.ModuleList([Cross_Self_layer(args) for _ in range(2)])
        self.layernorm_embedding = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(args.dropout)

    def forward(self, src_tokens, encoder_out, encoder_padding_mask):
        b, l = src_tokens.size()
        x = torch.tensor([[i for i in range(l)]]).expand(b, l).to(src_tokens)
        # self_attn_padding_mask = (torch.ones_like(x) < 0.5)
        self_attn_padding_mask = (src_tokens==1)

        x = self.embed_tokens(x)
        encoder_embedding = x

        x = self.dropout_module(self.layernorm_embedding(x))

        x = x.transpose(0, 1)
        self_attn_mask = torch.triu(torch.ones(l, l), diagonal=1).to(x) * (-1e8)
        encoder_states = [x]
        for idx, layer in enumerate(self.layers):
            x, attn_score = layer(x, encoder_out, encoder_padding_mask, self_attn_padding_mask, self_attn_mask)
            encoder_states.append(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [self_attn_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "att_score": [attn_score],
            "src_tokens": [],
            "src_lengths": [],
        }



class Output_layer(nn.Module):
    def __init__(self, base, output_projection):
        super().__init__()
        self.hidden_size = base.embed_dim
        self.final_layer_dropout = copy.deepcopy(base.activation_dropout_module)
        self.final_layer_fc1 = copy.deepcopy(base.fc1)
        self.final_layer_fc2 = copy.deepcopy(base.fc2)
        self.final_layer_act = copy.deepcopy(base.activation_fn)
        self.final_layer_norm = copy.deepcopy(base.final_layer_norm)
        self.W = output_projection.weight.detach().transpose(0, 1)


    def forward(self, input_tensor):
        hidden_states = self.final_layer_fc1(input_tensor)
        hidden_states = self.final_layer_act(hidden_states)
        hidden_states = self.final_layer_fc2(hidden_states)
        hidden_states = self.final_layer_dropout(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states + input_tensor)
        hidden_states = torch.matmul(hidden_states, self.W.to(hidden_states))
        return hidden_states

class Self_Attention(nn.Module):
    def __init__(self, base, attention_heads=8, memory_num=20):
        super().__init__()
        self.memory_num = memory_num
        self.dim = base.embed_dim
        self.attention_heads = attention_heads
        self.memory_module = nn.Parameter(torch.FloatTensor(self.memory_num, 1, self.dim))
        nn.init.uniform_(self.memory_module)
        self.memory_attention = MultiheadAttention(self.dim, self.attention_heads)

    def forward(self, keys, kay_padding_mask):
        T, B, C = keys[0].size()
        x = self.memory_module.repeat(1, B, 1)
        for key in keys:
            x, attn_score = self.memory_attention(
                query=x,
                key=key,
                value=key,
                key_padding_mask=kay_padding_mask,
                need_weights=True
            )
        # encoder_padding_mask = torch.zeros(B, x.size(0), dtype=torch.bool).to(x.device)
        encoder_padding_mask = torch.zeros(B, x.size(0)).to(kay_padding_mask)

        return x, encoder_padding_mask, attn_score

class cross_Attention1(nn.Module):
    def __init__(self, base, num_attention_heads=8):
        super().__init__()
        self.hidden_size = base.embed_dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads

        self.self_query = copy.deepcopy(base.self_attn.q_proj)
        self.self_key = copy.deepcopy(base.self_attn.k_proj)
        self.self_value = copy.deepcopy(base.self_attn.v_proj)
        self.self_drop = copy.deepcopy(base.self_attn.dropout_module)
        self.self_dense = copy.deepcopy(base.self_attn.out_proj)
        self.self_layer_norm = copy.deepcopy(base.self_attn_layer_norm)

        self.cross_query = copy.deepcopy(base.encoder_attn.q_proj)
        self.cross_key = copy.deepcopy(base.encoder_attn.k_proj)
        self.cross_value = copy.deepcopy(base.encoder_attn.v_proj)
        self.cross_drop = copy.deepcopy(base.encoder_attn.dropout_module)
        self.cross_dense = copy.deepcopy(base.encoder_attn.out_proj)
        self.cross_layer_norm = copy.deepcopy(base.encoder_attn_layer_norm)

        self.final_layer_dropout = copy.deepcopy(base.activation_dropout_module)
        self.final_layer_fc1 = copy.deepcopy(base.fc1)
        self.final_layer_fc2 = copy.deepcopy(base.fc2)
        self.final_layer_act = copy.deepcopy(base.activation_fn)
        self.final_layer_norm = copy.deepcopy(base.final_layer_norm)

    def transpose_for_scores(self, x):
        attention_head_size = x.size(-1) // self.num_attention_heads
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def att_forward(self, query_layer, key_layer, value_layer, kay_padding_mask):
        # [B,h,N*K,N*K]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [b,h,l,l]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_mask = kay_padding_mask.unsqueeze(1).expand(attention_scores.size()).to(attention_scores)

        attention_scores = attention_scores - (attention_mask * 1e5)
        attention_probs = F.softmax(attention_scores, dim=-1)
        # attention_probs = self.dropout(attention_probs)

        # [b,l,d]
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

    def forward(self, query, key, kay_padding_mask, query_padding_mask):
        # query/key => [b,l,d]
        # kay_padding_mask = [b,l]
        #  ======== Cross-attention ==============
        query_layer = self.transpose_for_scores(self.cross_query(query))  # [b,h,l,d]
        key_layer = self.transpose_for_scores(self.cross_key(key))
        value_layer = self.transpose_for_scores(self.cross_value(key))
        context_layer = self.att_forward(query_layer, key_layer, value_layer, kay_padding_mask)
        hidden_states = self.cross_dense(context_layer)
        hidden_states = self.cross_drop(hidden_states)
        input_tensor = self.cross_layer_norm(hidden_states + context_layer)

        #  ======== Self-attention ==============
        key = query = input_tensor
        query_layer = self.transpose_for_scores(self.self_query(query))  # [b,h,l,d]
        key_layer = self.transpose_for_scores(self.self_key(key))
        value_layer = self.transpose_for_scores(self.self_value(key))
        context_layer = self.att_forward(query_layer, key_layer, value_layer, query_padding_mask)
        hidden_states = self.self_dense(context_layer)
        hidden_states = self.self_drop(hidden_states)
        input_tensor = self.self_layer_norm(hidden_states + context_layer)

        hidden_states = self.final_layer_fc1(input_tensor)
        hidden_states = self.final_layer_act(hidden_states)
        hidden_states = self.final_layer_fc2(hidden_states)
        hidden_states = self.final_layer_dropout(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states + input_tensor)

        return hidden_states

class MultiHeadAttention(nn.Module):
    def __init__(self, base, num_attention_heads=8):
        super().__init__()
        self.hidden_size = base.embed_dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads

        self.query = copy.deepcopy(base.self_attn.q_proj)
        self.key = copy.deepcopy(base.self_attn.k_proj)
        self.value = copy.deepcopy(base.self_attn.v_proj)
        self.self_drop = copy.deepcopy(base.self_attn.dropout_module)
        self.self_dense = copy.deepcopy(base.self_attn.out_proj)
        self.self_layer_norm = copy.deepcopy(base.self_attn_layer_norm)

        self.final_layer_dropout = copy.deepcopy(base.activation_dropout_module)
        self.final_layer_fc1 = copy.deepcopy(base.fc1)
        self.final_layer_fc2 = copy.deepcopy(base.fc2)
        self.final_layer_act = copy.deepcopy(base.activation_fn)
        self.final_layer_norm = copy.deepcopy(base.final_layer_norm)

    def transpose_for_scores(self, x):
        attention_head_size = x.size(-1) // self.num_attention_heads
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, kay_padding_mask):
        # query/key => [b,l,d]
        # kay_padding_mask = [b,l]
        query_layer = self.transpose_for_scores(self.query(query))  # [b,h,l,d]
        key_layer = self.transpose_for_scores(self.key(key))
        value_layer = self.transpose_for_scores(self.value(key))

        # [B,h,N*K,N*K]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [b,h,l,l]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_mask = kay_padding_mask.unsqueeze(1).expand(attention_scores.size()).to(attention_scores)

        attention_scores = attention_scores - (attention_mask * 1e5)
        attention_probs = F.softmax(attention_scores, dim=-1)
        # attention_probs = self.dropout(attention_probs)

        # [b,l,d]
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        hidden_states = self.self_dense(context_layer)
        hidden_states = self.self_drop(hidden_states)
        input_tensor = self.self_layer_norm(hidden_states + context_layer)

        hidden_states = self.final_layer_fc1(input_tensor)
        hidden_states = self.final_layer_act(hidden_states)
        hidden_states = self.final_layer_fc2(hidden_states)
        hidden_states = self.final_layer_dropout(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states + input_tensor)

        return hidden_states, attention_scores




# Wav2VecEncoder from wt
'''
class Wav2VecEncoder(nn.Module):
    """
    base encoder: Wav2Vec2
    """

    def __init__(self, *args, **kwargs):
        super(Wav2VecEncoder, self).__init__()
        self.encoder = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h',
                                                      cache_dir='../cached_models/')
        self.config = self.encoder.config
        self.processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h',
                                                           cache_dir='../cached_models/')
        if "sampling_rate" in kwargs:
            self.sampling_rate = kwargs["sampling_rate"]
        else:
            self.sampling_rate = 16000

    def forward(self, **kwargs):
        """
        encode raw audio
        Args:
            encoder_input

        Returns:
        output (Wav2Vec2BaseModelOutput)
        """
        output = self.encoder(output_hidden_states=True, output_attentions=True, **kwargs)
        return output
'''


class TransformerDecoderMod(TransformerDecoder):
    """
    Modification of the TransformerDecoder

    It is adapted to the argument names defined in Wav2Vec2Seq2SeqModConfig.
    """
    def __init__(self, cfg, dictionary, embed_tokens, no_encoder_attn=False):
        transformer_cfg = copy.deepcopy(cfg)
        with open_dict(transformer_cfg):
            transformer_cfg.dropout = transformer_cfg.decoder_dropout
            transformer_cfg.attention_dropout = (
                transformer_cfg.decoder_attention_dropout
            )
            transformer_cfg.activation_dropout = (
                transformer_cfg.decoder_activation_dropout
            )
            transformer_cfg.layernorm_embedding = True
            transformer_cfg.adaptive_input = False
            transformer_cfg.no_scale_embedding = False
            transformer_cfg.quant_noise_pq = 0.0
            transformer_cfg.adaptive_softmax_cutoff = None
        super().__init__(transformer_cfg, dictionary, embed_tokens, no_encoder_attn)
        if cfg.decoder_enc_attention_dropout is not None:
            for layer in self.layers:
                layer.encoder_attn.dropout_module.p = \
                    cfg.decoder_enc_attention_dropout

    def load_state_dict(self, state_dict, strict=True):
        state_dict["output_projection.weight"] = state_dict["embed_tokens.weight"]
        super().load_state_dict(state_dict, strict)


# BartDecoder from wt
'''
class BartDecoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BartDecoder, self).__init__()
        self.lm = BartForConditionalGeneration.from_pretrained('facebook/bart-base', cache_dir='../cached_models')
        self.config = self.lm.config
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', cache_dir="../cached_models")

        # add special token
        self.num_new_tokens = self.tokenizer.add_tokens(['<subj>', '<obj>', "<triplet>"])
        # self.lm.resize_token_embeddings(self.num_new_tokens)
        self.lm.resize_token_embeddings(10000)  # 这里改了
        self.decoder = deepcopy(self.lm.get_decoder())
        self.decoder_head = deepcopy(self.lm.lm_head)
        self.decoder_final_logits_bias = deepcopy(self.lm.final_logits_bias)

        del self.lm

        self.modify_architecture()  # if necessary
        pass

    def forward(self, *args, **kwargs):
        decoder_outputs = self.decoder(
            input_ids=kwargs["input_ids"],  #
            attention_mask=kwargs["attention_mask"],  #
            encoder_hidden_states=kwargs["encoder_hidden_states"][-1],  #
            encoder_attention_mask=kwargs["encoder_attentions"],  #
            head_mask=None,  #
            cross_attn_head_mask=None,
            past_key_values=None,
            inputs_embeds=None,  # important, used for generation
            use_cache=None,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        return decoder_outputs

    def modify_architecture(self):
        """
        modify the architecture of bart decoder -- insert adapter
        Returns:
        None
        """
        pass

    @classmethod
    def shift_tokens_right(cls, input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    @classmethod
    def prepare_decoder_input_ids_from_labels(cls, labels: torch.Tensor):
        return cls.shift_tokens_right(labels,
                                      cls.config.pad_token_id,
                                      cls.config.decoder_start_token_id)
'''


class Adapter(nn.Module):
    """
    Adapter for model finetuning, as described in:
    https://arxiv.org/pdf/1909.08478.pdf
    """
    def __init__(self, embed_dim, proj_dim, dropout):
        super().__init__()
        self.layer_norm = LayerNorm(embed_dim)
        self.down_proj = nn.Linear(embed_dim, proj_dim)
        self.up_proj = nn.Linear(proj_dim, embed_dim)
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.down_proj(x)
        x = F.relu(x)
        x = self.up_proj(x)
        x = self.dropout_module(x)
        x += residual
        return x


# LengthAdapter from wt
class LengthAdapter(nn.Module):
    def __init__(self, *args, **kwargs):
        super(LengthAdapter, self).__init__()
        self.input_dim = kwargs["adapter_input_dim"]
        self.output_dim = kwargs["adapter_output_dim"]
        self.mid_dim = kwargs["adapter_hidden_size"]
        self.adapter = nn.Sequential(
            nn.Linear(self.input_dim, self.mid_dim),
            nn.ReLU(),
            nn.LayerNorm(self.mid_dim),
            nn.Linear(self.mid_dim, self.output_dim)
        )
        pass

    def forward(self, batch_data):
        """
        transfer input data

        Args:
            batch_data (Tensor): the hidden states of encoder

        Returns:
        tensor
        """
        bsz, in_seq_len, _ = batch_data.size()  # B x T x (C x D)
        x = batch_data.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        x = self.adapter(x)
        output = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        return output
