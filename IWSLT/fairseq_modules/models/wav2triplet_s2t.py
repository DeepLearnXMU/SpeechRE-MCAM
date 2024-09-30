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
# from fairseq.modules import MultiheadAttention

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
# '<mask>': 50264, '<triplet>': 50265, '<subj>': 50266, '<obj>': 50267, '<entity>': 50268, '</entity>': 50269, '<blank>':50270

# special_tokens_id={'<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3, '<mask>': 50264, '<triplet>': 50265, '<subj>': 50266,
#                    '<obj>': 50267, '<peop>': 50268, '</peop>': 50269,'<org>': 50270, '</org>': 50271, '<other>': 50272,
#                    '</other>': 50273, '<loc>': 50274, '</loc>': 50275}
# entity_ids = [50268, 50269, 1]
# entity_ids = [50268, 50269, 1, 0, 2, 50270]
entity_ids = [50268, 50269]
entity_start = ['<peop>','<org>','<other>','<loc>']
entity_end = {'<peop>':'</peop>','<org>':'</org>','<other>':'</other>','<loc>':'</loc>'}

entity_start_id = [50268,50270,50272,50274]
entity_end_id = [50269,50271,50273,50275]

import math


def log_sum_exp(lps):
    _inf = -float('inf')
    if all(lp == _inf for lp in lps): return _inf
    mlp = max(lps)
    return mlp + math.log(sum(math.exp(lp - mlp) for lp in lps))


def beam_search_ctc(probs, bms=5, blank=50271):
    '''
    probs: 概率空间，shape为[sequence_len,vocab_size]的torch tensor
    bms: beam_size
    blank: blank index
    '''
    _inf = -float("inf")
    seqs = [((idx.item(),), (lp.item(), _inf)) if idx.item() != blank
            else (tuple(), (_inf, lp.item()))
            for lp, idx in zip(*probs[0].topk(bms))]
    for i in range(1, probs.size(0)):
        new_seqs = {}
        for seq, (lps, blps) in seqs:
            last = seq[-1] if len(seq) > 0 else None
            for lp, idx in zip(*probs[i].topk(bms)):
                lp = lp.item()
                idx = idx.item()
                if idx == blank:
                    nlps, nblps = new_seqs.get(seq, (_inf, _inf))
                    new_seqs[seq] = (nlps, log_sum_exp([nblps, lps + lp, blps + lp]))
                elif idx == last:
                    # aa
                    nlps, nblps = new_seqs.get(seq, (_inf, _inf))
                    new_seqs[seq] = (log_sum_exp([nlps, lps + lp]), nblps)
                    # a-a
                    new_seq = seq + (idx,)
                    nlps, nblps = new_seqs.get(new_seq, (_inf, _inf))
                    new_seqs[new_seq] = (log_sum_exp([nlps, blps + lp]), nblps)
                else:
                    new_seq = seq + (idx,)
                    nlps, nblps = new_seqs.get(new_seq, (_inf, _inf))
                    new_seqs[new_seq] = (log_sum_exp([nlps, lps + lp, blps + lp]), nblps)
        new_seqs = sorted(
            new_seqs.items(),
            key=lambda x: log_sum_exp(list(x[1])),
            reverse=True)
        seqs = new_seqs[:bms]
    return seqs



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
        model_ckpt = torch.load("/mnt/chongqinggeminiceph1fs/geminicephfs/pr-training-mt/liangzhang/SpeechRE/IWSLT/Save/load_model/checkpoint_ASR_8.pt",map_location ='cpu')
        self.load_state_dict(model_ckpt['model'], strict=False)
        # self.encoder.Generate_len_adaptor()

    @classmethod
    def build_model(cls, cfg: Wav2Vec2Seq2SeqModConfig, task: FairseqTask):
        """Build a new model instance."""
        encoder = cls.build_encoder(cfg)  # must FairseqEncoder type
        decoder = cls.build_decoder(cfg)  # must FairseqDecoder type
        encoder.text_encoder = cls.encoder_text
        encoder.Rel_compressor = cls.Rel_compressor

        encoder.Semantic_share_layer = cls.encoder_text.layers[-(cls.share_layer_n):]
        encoder.projection = copy.deepcopy(decoder.output_projection)
        cls.W = encoder.projection

        task.tgt_dict = cls.bart.task.source_dictionary
        model = Wav2Vec2Seq2SeqModModelRE(encoder, decoder)


        model.freeze_blocks(cfg)
        # model.freeze_model(cls.bart.model.decoder)
        # model.freeze_model(cls.bart.model.encoder)
        # model.freeze_model(encoder.projection)

        model.unfreeze_model(encoder.Semantic_share_layer)
        model.unfreeze_model(encoder.len_adaptor)
        model.unfreeze_model(encoder.projection)
        model.unfreeze_model(encoder.Rel_compressor)
        model.unfreeze_model(encoder.w2v_model.encoder.layers[-6:])
        model.unfreeze_model(cls.bart.model.decoder.embed_tokens)

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
    def build_decoder(cls, cfg: Wav2Vec2Seq2SeqModConfig):
        from fairseq.models.bart import BARTModel
        bart = BARTModel.from_pretrained(cfg.load_pretrained_decoder_from, checkpoint_file='model.pt')
        # special_tokens = ['<triplet>', '<subj>', '<obj>', '<peop>', '</peop>', '<org>', '</org>', '<other>',
        #                     '</other>', '<loc>', '</loc>']
        # special_words = ["triplet", "subject", "object", "people", "organization", "location", "other"]

        # =============== add special tokes ====================
        special_tokens = ['<triplet>', '<subj>', '<obj>', "<entity>", "</entity>","<blank>"]  #+ ['<query_%d>'%i for i in range(30)]
        for word_s in special_tokens:
            bart.task.source_dictionary.add_symbol(word_s)

        # update embedding
        tgt_dict = bart.task.source_dictionary
        padding_idx = tgt_dict.pad()
        num_embeddings = len(tgt_dict)
        num_old, embed_dim =  bart.model.encoder.embed_tokens.weight.data.size()
        new_embs_layer = Embedding(num_embeddings, embed_dim, padding_idx)
        new_embs_layer.weight.data[:num_old] = bart.model.encoder.embed_tokens.weight.data
        bart.model.encoder.embed_tokens = new_embs_layer
        bart.model.decoder.embed_tokens = new_embs_layer

        cls.token_list_test = [0 for i in range(num_embeddings)]
        cls.token_flog_test = False

        decoder = bart.model.decoder
        decoder.build_output_projection(decoder.args, tgt_dict, new_embs_layer)
        cls.bart = bart
        cls.encoder_text = bart.model.encoder

        # build generator using current args as well as any kwargs
        gen_args = copy.deepcopy(bart.cfg.generation)
        with open_dict(gen_args):
            gen_args.beam = 5
        cls.generator_trip = bart.task.build_generator(bart.models, gen_args)

        cls.share_layer_n = 4
        # cls.Rel_compressor = Cross_Attention(copy.deepcopy(cls.bart.model.args))  #cls.encoder_text.layers[-1]
        cls.Rel_compressor = Cross_Attention(copy.deepcopy(cls.bart.model.args), cls.encoder_text.layers[-(cls.share_layer_n + 1)])
        cls.Blank = cls.Rel_compressor.Blank_token
        cls.Blank.data = bart.model.decoder.embed_tokens.weight.data[0]
        cls.bart.model.encoder.add_blank_tokens(cls.Blank)
        cls.GRL = cls.Rel_compressor.grl
        cls.T = cls.Rel_compressor.T

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

    def valid_step_RE(self,sample):
        net_input = sample["net_input"]
        src_text, src_text_lengths, src_tokens = net_input['src_text'], net_input['src_text_lengths'], net_input['src_tokens']
        src_lengths, prev_output_tokens = net_input['src_lengths'], net_input['prev_output_tokens']
        audi_encoder_out = self.encoder(src_text, src_tokens, src_lengths=src_lengths)

        # Mark the location of entities in the input text
        entity_mask_M = self.get_entity_pos(src_text, start_id=50268, end_id=50269)

        # Remove entity mask '<entity>' and '</entity>'
        src_text_c, entity_mask_c, src_text_len_c = self.Remove_entity_mask(src_text, entity_mask_M)

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

        # =================== decoding ===================
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
            # self.bart.decode(src_text[i])
            # pdb.set_trace()

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
        # text encoder
        encoder_out = self.bart.model.encoder(
            src_text,
            src_lengths=src_text_lengths,
            # token_embeddings= token_embs,
            return_all_hiddens=True,
            # compression_F= self.encoder.compression_forward
            # compression_F=compression_F
        )

        Alig_input={
            "speech":encoder_out_s["encoder_embedding"][0],
            "text":encoder_out['encoder_states'][-(self.share_layer_n+1)],
            "pad_mask_s":encoder_out_s['encoder_padding_mask'][1],
            "pad_mask_t":encoder_out['encoder_padding_mask'][0],
            "entity_mask":entity_mask,
            "share_layer_forward":self.encoder.share_layer_forward,
            "Rel_compressor":self.encoder.Rel_compressor,
            "Contra_loss": self.Contra_loss_C,
        }

        Q_n = encoder_out_s["encoder_out"][0].size(0) - encoder_out_s['encoder_states'][0].size(0)

        # mixup text and speech，and Get text/mixup feature sequence
        x_t, x_ts, mask_t, pad_mask_t, Loss = self.bart.model.encoder.Alignment_forward(**Alig_input)

        # some feature sequence recoder
        en_out_s = [encoder_out_s["encoder_embedding"][0].transpose(0, 1), encoder_out_s['encoder_states'][0].transpose(0, 1)]
        en_out_t = [encoder_out['encoder_states'][-(self.share_layer_n+1)].transpose(0, 1), x_t[Q_n:].transpose(0, 1), x_ts[Q_n:].transpose(0, 1)]
        # pad_mask_s1 = encoder_out_s['encoder_padding_mask'][0]
        pad_mask_t1 = encoder_out['encoder_padding_mask'][0]
        en_out_pro = [x_t[:Q_n].transpose(0, 1), x_ts[:Q_n].transpose(0, 1),encoder_out_s["encoder_out"][0][:Q_n].transpose(0, 1)]

        # text decoder
        de_out = []
        encoder_out["encoder_out"][0] = x_t
        encoder_out['encoder_padding_mask'][0] = pad_mask_t
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

        encoder_out["encoder_out"][0] = x_ts
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

        # encoder_out_s["encoder_out"][0] = x_s
        # encoder_out_s['encoder_padding_mask'][0] = pad_mask_s
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

        # encoder_out_s['encoder_padding_mask'][0] = pad_mask_s1
        encoder_out_s['encoder_padding_mask'][0] = encoder_out_s['encoder_padding_mask'][1]
        encoder_out['encoder_padding_mask'][0] = pad_mask_t1

        Output = {
            "en_out_t": en_out_t,
            "en_out_s": en_out_s,
            "de_out": de_out,
            "att_e": encoder_out["att_score"][-1],
            "att_d": decoder_out3[1]['attn'][-1],
            'mask_t': mask_t,
            'en_out_pro':en_out_pro,
            'Loss':Loss
        }

        return Output

    def KD_loss(self, decoder_s, decoder_t, mask, T = 1.0):
        # KL
        p_s = decoder_s[0][mask]
        # p_t = decoder_t[0][mask]
        p_t = decoder_t[0][mask].detach()
        Loss1 = F.kl_div(F.log_softmax(p_s / T, dim=-1), F.softmax(p_t / T, dim=-1), reduction='sum')
        # Loss1 += F.kl_div(F.log_softmax(p_t / T, dim=-1), F.softmax(p_s / T, dim=-1), reduction='sum')

        # L2
        Loss2 = 0.0
        tem = [1.0, 0.5, 0.25]
        for j, k in enumerate([-1, -4, -8]):
            h_out1 = decoder_s[1]['inner_states'][k].transpose(0, 1)[mask]
            h_out2 = decoder_t[1]['inner_states'][k].transpose(0, 1)[mask].detach()
            # h_out2 = decoder_t[1]['inner_states'][k].transpose(0, 1)[mask]
            Loss2 += tem[j] * F.pairwise_distance(h_out1, h_out2, p=2).sum()
            # Loss1 += tem[j] * self.loss_fn(h_out1, h_out2).sum()
        Loss2 = Loss2 / decoder_s[0].size(0)

        return Loss1

    def Contra_loss_C(self, x, y, mask=None, T=0.1):
        y = y.detach()
        Loss1 = (F.pairwise_distance(x, y, p=2) * (1.0 - mask.to(x))).sum() / x.size(0)

        x = F.normalize(x.float(), dim=-1, p=2)
        y = F.normalize(y.float(), dim=-1, p=2)
        # T = self.GRL(self.T[1])[0]
        mask_p = mask.to(x).unsqueeze(1) * (-1e6)
        x_y = -1 * torch.log(F.softmax(torch.matmul(x, y.transpose(-2, -1)) / T + mask_p, dim=-1) + 1e-6)
        y_x = -1 * torch.log(F.softmax(torch.matmul(y, x.transpose(-2, -1)) / T + mask_p, dim=-1) + 1e-6)
        label = torch.eye(y_x.size(-1)).unsqueeze(0).expand(x_y.size(0), -1, -1).to(y_x) * (1.0 - mask.unsqueeze(-1).float())
        Loss = (x_y * label + y_x * label).sum() / x.size(0)

        # return (Loss + Loss1) / 2.0
        return Loss

    def Contra_loss(self, x, y, T=0.1):
        # y = y.detach()
        # L2
        Loss1 = F.pairwise_distance(x, y, p=2).sum()
        #L_CL
        x = F.normalize(x.float(), dim=-1, p=2)
        y = F.normalize(y.float(), dim=-1, p=2)
        # T = self.GRL(self.T[0])[0]
        x_y = -1 * torch.log(F.softmax(torch.matmul(x, y.transpose(-2, -1)) / T, dim=-1) + 1e-6)
        y_x = -1 * torch.log(F.softmax(torch.matmul(y, x.transpose(-2, -1)) / T, dim=-1) + 1e-6)
        label = torch.eye(y_x.size(-1)).unsqueeze(0).expand(x_y.size(0), -1, -1).to(y_x)
        Loss = (x_y * label + y_x * label).sum()

        return Loss


    def get_entity_pos(self, input, start_id, end_id):
        entity_start_mask = (input == start_id)
        entity_end_mask = (input == end_id)

        src_text_tmp = torch.zeros_like(input)
        src_text_tmp = (src_text_tmp + entity_start_mask.to(src_text_tmp) - entity_end_mask.to(src_text_tmp))
        src_text_tmp = src_text_tmp.cumsum(-1) - entity_start_mask.to(src_text_tmp)

        entity_mask_M = (src_text_tmp > 0.5)
        return entity_mask_M


    def get_CTC_input2(self, src_text, src_text_lengths, enc_out_t, enc_out_s, audio_len):
        enc_out_t = enc_out_t.detach()

        mask = (((src_text != 1).float() * (src_text != 50270).float()) > 0.5)
        b, l = src_text.size()

        W_b = self.Blank.unsqueeze(-1).expand(b, -1, -1).to(enc_out_s)
        W = enc_out_t[mask].unsqueeze(0).expand(b, -1, -1).transpose(-1, -2)
        # x_id = torch.arange(src_text.size(0)).to(src_text)
        # W_b = enc_out_t[x_id, src_text_lengths].unsqueeze(1).transpose(-1, -2)
        # W_b = enc_out_t[x_id, src_text_lengths]
        # Mask_label = mask.to(enc_out_t).unsqueeze(-1)
        # W_b = (enc_out_t * Mask_label).sum(1)/Mask_label.sum(1)

        # W_b_list = []
        # for i in range(b):
        #     W_b_list.append(torch.cat((W_b[:i], W_b[i + 1:], W_b[i:i + 1]), dim=0))
        # W_b_list = torch.stack(W_b_list,dim=0)
        # W_b = W_b_list.transpose(-1, -2)
        # W = enc_out_t[mask].detach().unsqueeze(0).expand(b, -1, -1).transpose(-1, -2)

        # l = l - 1
        CTC_label = torch.tensor([[i for i in range(l)]] * b).to(src_text)
        CTC_label_b = torch.tensor([0] + src_text_lengths.tolist()[:-1]).to(src_text).cumsum(dim=-1)
        CTC_label = CTC_label + CTC_label_b.unsqueeze(-1)
        audio_pred = torch.cat((torch.matmul(enc_out_s, W), torch.matmul(enc_out_s, W_b)), dim=-1)

        # W_b[CTC_label[0][:src_text_lengths[0]]] == enc_out_t[0][:src_text_lengths[0]]
        return [audio_pred, audio_len, CTC_label, src_text_lengths]


    def Remove_entity_mask(self, src_text, entity_mask_M):
        # Remove entity mask '<entity>' and '</entity>'
        b, l = src_text.size()
        src_text_c = torch.ones_like(src_text)
        src_text_mask = torch.ones((b, 2 * l)).to(src_text)
        entity_mask = (torch.ones((b, 2 * l)).to(src_text) < 0.5)
        entity_pos = torch.tensor(entity_ids+[1,]).to(src_text)
        max_len, max_len_c = -1, -1
        for i in range(src_text.size(0)):
            mask = (~torch.isin(src_text[i], entity_pos))
            temp = src_text[i][mask]
            temp_b = entity_mask_M[i][mask]

            l = temp.size(0)
            src_text_c[i][:l] = temp.clone()
            entity_mask[i][:l] = temp_b.clone()
            if (max_len_c < l):
                max_len_c = l

            # add noise
            # temp, temp_b = self.add_noise(temp, temp_b)
            # l = temp.size(0)
            # src_text_mask[i][:l] = temp
            # # entity_mask[i][:l] = temp_b
            #
            # if (max_len < l):
            #     max_len = l

        # src_text_mask = src_text_mask[:, :max_len]
        # entity_mask = entity_mask[:, :max_len]
        # src_text_len = (src_text_mask != 1).sum(-1).long()

        src_text = src_text_c[:, :max_len_c]
        entity_mask_M = entity_mask[:, :max_len_c]
        src_text_lengths = (src_text != 1).sum(-1).long()

        src_text_p, entity_mask, src_text_lengths = src_text, entity_mask_M, src_text_lengths

        return src_text_p, entity_mask, src_text_lengths

    def forward(self, sample, step, train):
        # src_text, src_text_lengths, src_tokens, src_lengths, prev_output_tokens, ** kwargs
        net_input = sample["net_input"]
        src_text, src_text_lengths, src_tokens = net_input['src_text'], net_input['src_text_lengths'], net_input['src_tokens']
        src_lengths, prev_output_tokens = net_input['src_lengths'], net_input['prev_output_tokens']
        audi_encoder_out = self.encoder(src_text, src_tokens, src_lengths=src_lengths)

        # Mark the location of entities in the input text
        entity_mask = self.get_entity_pos(src_text, start_id=50268, end_id=50269)
        # Remove entity mask '<entity>' and '</entity>' and add noise
        # src_text, entity_mask, src_text_lengths = self.Remove_entity_mask(src_text, entity_mask)
        BART_Output = self.BART_forward(src_text, src_text_lengths, prev_output_tokens, entity_mask, encoder_out_s=audi_encoder_out)

        # Speech to text distributed for CTC loss
        audio_len = (1 - audi_encoder_out['encoder_padding_mask'][0].long()).sum(-1)
        enc_out_t, enc_out_s = BART_Output["en_out_t"][0], BART_Output["en_out_s"][0]
        CTC_input = self.get_CTC_input2(src_text.clone(), src_text_lengths.clone(), enc_out_t.clone(), enc_out_s.clone(), audio_len.clone())

        # Relation Alignment
        Loss1 = self.Contra_loss(BART_Output["en_out_pro"][-1], BART_Output["en_out_pro"][0]) #+ BART_Output["Loss"]
        # Loss1 += self.Contra_loss(BART_Output["en_out_pro"][1], BART_Output["en_out_pro"][0])
        # Loss1 = Loss1 / 2.0

        # KL loss
        mask = sample['target'] != 1
        Loss = self.KD_loss(BART_Output["de_out"][1], BART_Output["de_out"][0], mask)
        Loss += self.KD_loss(BART_Output["de_out"][-1], BART_Output["de_out"][0], mask)
        # Loss = Loss / 2.0
        return BART_Output["de_out"], CTC_input, Loss, Loss1, BART_Output["Loss"]

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

# Speech Encoder (Wav2Vec model)
class Wav2VecEncoderMod(Wav2VecEncoder):
    """
    Modification of the Wav2VecEncoder

    It modifies it to work with the speech_to_text pipeline.
    Moreover, it includes the adapter and length adaptor modules.
    """
    def __init__(self, cfg: Wav2Vec2Seq2SeqModConfig, tgt_dict=None):
        super().__init__(cfg, tgt_dict)
        # Length Adapter (CNN)
        self.len_adaptor = Conv1dSubsampler(
            cfg.decoder_embed_dim,
            cfg.len_adaptor_channels,
            cfg.decoder_embed_dim,
        )

    def share_layer_forward(self, speech, pad_mask_s):
        x = speech
        for l, layer in enumerate(self.Semantic_share_layer):
            x = layer(x, encoder_padding_mask=pad_mask_s)
        return x

    def forward(self, src_text, src_tokens, src_lengths, **kwargs):
        # Get speech feature sequence by speech encoder
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

        # add soft_prompt(relation)
        soft_prompt = self.Rel_compressor([speech], pad_mask_s)
        speech_1 = torch.cat((soft_prompt, speech), dim=0)
        mask_pro = torch.zeros(soft_prompt.size(1), soft_prompt.size(0)).to(pad_mask_s)
        pad_mask_s = torch.cat((mask_pro, pad_mask_s), dim=-1)

        # Semantic share layer
        x = self.share_layer_forward(speech_1, pad_mask_s)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [pad_mask_s, encoder_out["encoder_padding_mask"]],  # B x T
            "encoder_embedding": [speech],  # B x T x C
            "encoder_states": [x[soft_prompt.size(0):]],  # List[T x B x C]
            "att_score": [],
            "src_tokens": [],
            "src_lengths": [],
        }

        # Out = self.Rel_compressor(src_text, encoder_out["encoder_out"], encoder_out["encoder_padding_mask"])
        # Out["encoder_out"] = [encoder_out["encoder_out"]]
        # Out["encoder_padding_mask"] = [encoder_out["encoder_padding_mask"]]
        # encoder_out["encoder_out"], encoder_out["encoder_padding_mask"] = self.Rel_compressor(src_text,
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
        return input

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        # pdb.set_trace()
        return grad_output.neg() * ctx.coeff, None

class GRL_Layer(nn.Module):
    def __init__(self):
        super(GRL_Layer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)

class MultiHeadAttention(nn.Module):
    def __init__(self, base, num_attention_heads=8):
        super().__init__()
        self.normalize_before = copy.deepcopy(base.normalize_before)
        self.self_attn = copy.deepcopy(base.self_attn)
        self.dropout_module = copy.deepcopy(base.dropout_module)
        self.self_attn.dropout_module.p = 0.0
        self.self_attn_layer_norm = copy.deepcopy(base.self_attn_layer_norm)
        self.final_layer_norm = copy.deepcopy(base.final_layer_norm)
        self.activation_fn = copy.deepcopy(base.activation_fn)
        self.activation_dropout_module = copy.deepcopy(base.activation_dropout_module)
        self.fc1 = copy.deepcopy(base.fc1)
        self.fc2 = copy.deepcopy(base.fc2)
        self.dropout_module = copy.deepcopy(base.dropout_module)

    def forward(self, query, key, kay_padding_mask):
        residual = query
        if self.normalize_before:
            query = self.self_attn_layer_norm(x)

        query, att_score = self.self_attn(
            query=query,
            key=key,
            value=key,
            key_padding_mask=kay_padding_mask,
            need_weights=True
        )

        query = self.dropout_module(query)
        # x = self.residual_connection(x, residual)
        if not self.normalize_before:
            query = self.self_attn_layer_norm(query)

        residual = query
        if self.normalize_before:
            query = self.final_layer_norm(query)
        query = self.activation_fn(self.fc1(query))
        query = self.activation_dropout_module(query)
        query = self.fc2(query)
        query = self.dropout_module(query)
        # x = self.residual_connection(x, residual)
        query = query + residual
        if not self.normalize_before:
            query = self.final_layer_norm(query)

        return query, att_score


class Cross_Attention(nn.Module):
    def __init__(self, args, base_layer, relation_num=5):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.attention_heads = args.encoder_attention_heads
        self.relation_num = relation_num
        self.grl = GRL_Layer()

        self.relation_query = nn.Parameter(torch.FloatTensor(self.relation_num, 1, self.embed_dim))
        nn.init.uniform_(self.relation_query)

        self.Blank_token = nn.Parameter(torch.FloatTensor(1, self.embed_dim))
        nn.init.uniform_(self.Blank_token)

        self.T = [nn.Parameter(torch.FloatTensor([1])), nn.Parameter(torch.FloatTensor([0.01])), nn.Parameter(torch.FloatTensor([1]))]

        # self.reduce_dim = nn.Linear(self.embed_dim, self.embed_dim // 4)
        # self.memory_attention = MultiheadAttention(self.embed_dim, self.attention_heads)
        # self.Predictor = nn.Linear(self.embed_dim // 4, relation_num * 2, bias=False)
        self.memory_attention = MultiHeadAttention(base_layer)

    def forward(self, keys, kay_padding_mask):
        T, B, C = keys[0].size()
        x = self.relation_query.repeat(1, B, 1)

        for key in keys:
            # key = self.reduce_dim(self.grl(key))
            x, attn_score = self.memory_attention(query=x, key=key, kay_padding_mask=kay_padding_mask)

        return x
