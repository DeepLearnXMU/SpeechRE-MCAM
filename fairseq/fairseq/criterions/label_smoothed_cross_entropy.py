# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import pdb
from dataclasses import dataclass, field
import torch.nn.functional as F

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import logging
import math
import os
import sys
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("Loss:")

@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

    # if target.dim() == lprobs.dim() - 1:
    #     target = target.unsqueeze(-1)
    # eps_i = epsilon / (lprobs.size(-1) - 1)
    #
    # nll_loss = -lprobs.gather(dim=-1, index=target)
    # smooth_loss = -(eps_i * lprobs).sum(dim=-1, keepdim=True)
    # if ignore_index is not None:
    #     pad_mask = target.eq(ignore_index)
    #     nll_loss.masked_fill_(pad_mask, 0.0)
    #     smooth_loss.masked_fill_(pad_mask, 0.0)
    # else:
    #     nll_loss = nll_loss.squeeze(-1)
    #     smooth_loss = smooth_loss.squeeze(-1)
    # if reduce:
    #     nll_loss = nll_loss.sum()
    #     smooth_loss = smooth_loss.sum()
    # # eps_i = epsilon / (lprobs.size(-1) - 1)
    # # loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    # loss = (1.0 - epsilon - eps_i) * nll_loss + smooth_loss
    #
    # return loss, nll_loss


def Focal_loss(lprobs, target, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    alpha = 1.0
    gamma = 2.0
    p = torch.exp(lprobs)
    lprobs = (alpha * ((1 - p) ** gamma) * lprobs)
    nll_loss = -lprobs.gather(dim=-1, index=target)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()

    return nll_loss



@register_criterion(
    "label_smoothed_cross_entropy", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        # 记录下损失函数参数
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        # self.ctc_loss = torch.nn.CTCLoss(blank=self.padding_idx, reduction='mean')
        self.ctc_loss = torch.nn.CTCLoss(blank=50270, reduction='mean')
        self.Step = 0
        self.N1 =1500.0
        self.N = 300.0

        self.Loss = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def forward(self, model, sample, reduce=True, train=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # net_output, net_output_en = model(**sample["net_input"])
        # net_output, net_output_en, net_output_audio, Loss = model(sample)
        # net_output10, net_output11, net_output20, net_output21, Loss = model(sample)
        # loss10, nll_loss10 = self.compute_loss(model, net_output10, sample, reduce=reduce)
        # loss11, nll_loss11 = self.compute_loss(model, net_output11, sample, reduce=reduce)
        # loss20, nll_loss20 = self.compute_loss(model, net_output20, sample, reduce=reduce)
        # loss21, nll_loss21 = self.compute_loss(model, net_output21, sample, reduce=reduce)

        # log_probs = F.log_softmax(enc_predict, dim=-1)
        # loss_en, nll_loss_en = label_smoothed_nll_loss(
        #     log_probs,
        #     enc_target,
        #     self.eps,
        #     ignore_index=self.padding_idx,
        #     reduce=reduce,
        # )
        # pdb.set_trace()
        # loss_en, nll_loss1_en = self.compute_loss(model, net_output_en, sample, reduce=reduce)
        # loss_au = self.compute_loss_ctc(model, net_output_audio, sample, reduce=reduce)

        # print("\n=====================")
        # print(loss.item(), "\t", loss_en.item(), "\t", loss_au.item(), "\t", Loss.item())
        # print("========================\n")

        # loss = 0.1 * loss10 + loss11 + 0.1 * loss20 + loss21 + Loss
        # nll_loss = 0.1 * nll_loss10 + nll_loss11 + 0.1 * nll_loss20 + nll_loss21
        if model.training:
            self.Step += 1.0
        alph = min(1.0, max(0.1, self.Step / self.N))
        alph1 = min(0.5, (self.Step / self.N))
        # alph1 = 0.1
        # alph = 0.4

        # net_output10, net_output11, enc_out1, enc_out2, Loss, Loss1 = model(sample, self.Step)
        # loss10, nll_loss10 = self.compute_loss(model, net_output10, sample, reduce=reduce)
        # loss11, nll_loss11 = self.compute_loss(model, net_output11, sample, reduce=reduce)
        # CTC_loss = self.compute_loss_ctc(enc_out1, sample, self.Step)

        decoder_out, CTC_inputs, Loss_KD, Loss_CL, CTC_loss = model(sample, self.Step, train=train)
        # RE task loss
        loss_t, nll_loss_t = self.compute_loss(model, decoder_out[0], sample, reduce=reduce)
        loss_m, nll_loss_m = self.compute_loss(model, decoder_out[1], sample, reduce=reduce)
        loss_s, nll_loss_s = self.compute_loss(model, decoder_out[-1], sample, reduce=reduce)

        # ctc loss
        CTC_loss += self.compute_loss_ctc(CTC_inputs, sample, self.Step)

        if model.training and self.Step % 300 == 0:
            self.Loss = [l / 300.0 for l in self.Loss]
            logger.info("CTC Loss: {}, KL loss:{}, CL loss:{}".format(self.Loss[3], self.Loss[4], self.Loss[5]))
            logger.info("Object Loss: {}, {}, {}".format(self.Loss[0], self.Loss[1], self.Loss[2]))
            self.Loss = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            self.Loss[0] += loss_t.item()
            self.Loss[1] += loss_m.item()
            self.Loss[2] += loss_s.item()
            self.Loss[3] += CTC_loss.item()
            self.Loss[4] += Loss_KD.item()
            self.Loss[5] += Loss_CL.item()


        loss = loss_s + loss_t + Loss_CL + CTC_loss + Loss_KD
        nll_loss = nll_loss_s + nll_loss_t + Loss_CL + CTC_loss + Loss_KD

        if(torch.isnan(loss).any() or torch.isinf(loss).any()):
            logger.info("============== Loss is NaN or Inf =============")
            logger.info("CTC Loss: {}, KL loss:{}, CL loss:{}".format(CTC_loss.item(), Loss_KD.item(), Loss_CL.item()))
            logger.info("Object Loss: {}, {}, {}".format(loss_t.item(),loss_m.item(), loss_s.item()))
        # loss = loss11 + alph * (loss10 + Loss) + Loss1
        # nll_loss = nll_loss11 + alph * (nll_loss10 + Loss) + Loss1
        # loss = loss11 + alph * (loss10 + Loss + Loss1)
        # nll_loss = nll_loss11 + alph * (nll_loss10 + Loss + Loss1)
        # breakpoint() 计算关系抽取的指标
        rel_precision, rel_recall, rel_f1, total = self.compute_re(decoder_out[0], sample)
        sample_size = (sample["target"].size(0) if self.sentence_avg else sample["ntokens"])

        logging_output = {
            "rel_precision": rel_precision,
            "rel_recall": rel_recall,
            "rel_f1": rel_f1,
            "loss": loss.item(),
            "nll_loss": nll_loss.item(),
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def compute_loss_en(self, net_output, reduce=True):
        lprobs = F.log_softmax(net_output[0] + 1e-5, dim=-1).view(-1, net_output[0].size(-1))
        target = net_output[1].view(-1)
        torch.isinf(lprobs).any()
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss_ctc(self, net_output, samples, step):
        log_probs = F.log_softmax(net_output[0], dim=-1).transpose(0,1)
        # probs = F.softmax(net_output[0], dim=-1).transpose(0,1)
        # log_probs = torch.log(probs + 1e-8)
        # probs = torch.exp(log_probs)
        # log_probs = ((probs < 0.7).to(probs).detach() + 1e-4) * log_probs

        # probs[1].topk(10, dim=-1)
        # if step>500:
        #     pdb.set_trace()
        input_lengths = net_output[1]
        targets = net_output[2]
        target_lengths = net_output[3]
        blank = log_probs.size(-1) - 1

        # log_probs_list = []
        # Max_len, L = log_probs.size()[1:]
        # for i in range(input_lengths.size(0)):
        #     if input_lengths[i] < target_lengths[i]:
        #         log_probs_list.append(log_probs[i].unsqueeze(-2).expand(-1,2,-1).contiguous().view(-1,L))
        #
        #         input_lengths[i] *= 2
        #         if (Max_len < input_lengths[i]):
        #             Max_len = input_lengths[i]
        #     else:
        #         log_probs_list.append(torch.cat((log_probs[i],log_probs[i]),dim=0))
        # log_probs = (torch.stack(log_probs_list, dim=0)[:, :Max_len]).transpose(0,1)

        # Mask = (target_lengths < input_lengths)
        # if (Mask.float().sum() < 0.5):
        #     return -1.0
        # log_probs = log_probs[:,Mask]
        # targets = targets[Mask]
        # input_lengths = input_lengths[Mask]
        # target_lengths = target_lengths[Mask]

        # temp_len = input_lengths - 5
        # mask = (target_lengths < temp_len).float()
        # target_lengths = (mask * target_lengths + (1 - mask) * temp_len).long()
        loss = torch.tensor(0).to(log_probs)
        Sample_n = torch.tensor(1.0).to(log_probs)
        Flag = True
        for i in range(targets.size(0)):
            # loss_temp = self.ctc_loss(log_probs[:,i:i+1].float(), targets[i:i+1], input_lengths[i:i+1], target_lengths[i:i+1])
            loss_temp = F.ctc_loss(log_probs[:, i:i + 1].float(), targets[i:i + 1], input_lengths[i:i + 1],target_lengths[i:i + 1], blank=blank, reduction="sum")

            if((not torch.isnan(loss_temp).any()) and (not torch.isinf(loss_temp).any())):
                loss += loss_temp
                Sample_n += 1.0
                Flag = False
            else:
                logger.info(samples['id'][i])

        # if Flag:
        #     pdb.set_trace()

        # Mask = (target_lengths > 0)
        # if (Mask.float().sum() < 0.5):
        #     return -1.0
        # log_probs = log_probs[:,Mask]
        # targets = targets[Mask]
        # input_lengths = input_lengths[Mask]
        # target_lengths = target_lengths[Mask]
        #
        # loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        # Sample_n = 2.0 * Sample_n
        if(torch.isnan(loss).any() or torch.isinf(loss).any()):
            pdb.set_trace()
        return loss / Sample_n / 2.0
        # return loss

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        if (len(net_output) > 2):
            target = (net_output[1][:,:-1]).contiguous().view(-1)
            mask_pad = (net_output[2][:,:-1]).contiguous().view(-1)

        # Focalloss
        # nll_loss = Focal_loss(
        #     lprobs,
        #     target,
        #     ignore_index=self.padding_idx,
        #     reduce=reduce,
        # )
        # loss = nll_loss
        # breakpoint()
        # loss：顺滑后的损失，nll_loss：标准交叉熵
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def extract_triplets(self, text):
        del_list = [0, 1, 2]
        text = [None if i in del_list else i for i in text]
        text = str(text).replace(",", "").replace("None", "").strip("[ ]")
        triplets = []
        relation, subject, relation, object_ = '', '', '', ''
        current = 'x'
        for token in text.split():
            if token == "50265":
                current = 't'
                if relation != '':
                    triplets.append({'head': 0, 'type': relation.strip(), 'tail': 0})
                    relation = ''
                subject = ''
            elif token == "50266":
                current = 's'
                if relation != '':
                    triplets.append({'head': 0, 'type': relation.strip(), 'tail': 0})
                object_ = ''
            elif token == "50267":
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
            triplets.append({'head': 0, 'type': relation.strip(), 'tail': 0})
        return triplets

    def compute_re(self, net_output, sample):
        target = sample["target"]
        output = torch.max(net_output[0], dim=-1)[1]
        assert target.shape == output.shape

        p = r = 0
        total_p = total_r = 1e-5
        for j in range(target.shape[0]):
            target_triplets = self.extract_triplets(target[j].tolist())
            output_triplets = self.extract_triplets(output[j].tolist())

            target_relations = []
            output_relations = []
            for i in target_triplets:
                target_relations.append(tuple(i.values()))
            for i in output_triplets:
                output_relations.append(tuple(i.values()))
            for i in target_relations:
                if i in output_relations:
                    r += 1
                total_r += 1
            for i in output_relations:
                if i in target_relations:
                    p += 1
                total_p += 1

        p = p / total_p
        r = r / total_r
        f1 = 2 * p * r / (p + r + 1e-5)
        total = target.shape[0]
        return p, r, f1, total

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        p_sum = sum(log.get("rel_precision", 0) for log in logging_outputs)
        r_sum = sum(log.get("rel_recall", 0) for log in logging_outputs)
        f1_sum = sum(log.get("rel_f1", 0) for log in logging_outputs)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "rel_precision", p_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "rel_recall", r_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "rel_f1", f1_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
