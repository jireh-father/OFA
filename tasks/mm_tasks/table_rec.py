# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
import json
import logging
import os
from typing import Optional
from argparse import Namespace
from itertools import zip_longest
from collections import OrderedDict

import numpy as np
import sacrebleu
import string
from fairseq import metrics, utils
from fairseq.tasks import register_task

from omegaconf import DictConfig

from tasks.ofa_task import OFATask, OFAConfig
from data.mm_data.table_rec_dataset import TableRecDataset
from data.file_dataset import FileDataset
from utils.teds import TEDS, preprocess_tag_str, decode_to_html

EVAL_BLEU_ORDER = 4

logger = logging.getLogger(__name__)


@dataclass
class TableRecConfig(OFAConfig):
    eval: bool = field(
        default=True, metadata={"help": "evaluation"}
    )
    eval_args: Optional[str] = field(
        default='{}',
        metadata={
            "help": 'generation args for BLUE or CIDEr scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )
    max_spans: int = field(
        default=10, metadata={"help": "max numbers of span cells"}
    )


@register_task("table_rec", dataclass=TableRecConfig)
class TableRecTask(OFATask):
    def __init__(self, cfg: TableRecConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.teds = TEDS(n_jobs=4)

    @classmethod
    def setup_task(cls, cfg: DictConfig, **kwargs):
        """Setup the task."""

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(cfg.bpe_dir, "dict.txt")
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(cfg.bpe_dir, "dict.txt")
        )

        src_dict.add_symbol("<tr>")
        src_dict.add_symbol("</tr>")
        src_dict.add_symbol("<td>")
        src_dict.add_symbol("</td>")
        for i in range(cfg.max_spans - 1):
            src_dict.add_symbol('<tdcolspan="{}">'.format(i+2))
            src_dict.add_symbol('<tdrowspan="{}">'.format(i + 2))
        src_dict.add_symbol("<mask>")
        tgt_dict.add_symbol("<mask>")
        for i in range(cfg.code_dict_size):
            src_dict.add_symbol("<code_{}>".format(i))
            tgt_dict.add_symbol("<code_{}>".format(i))
        # quantization
        for i in range(cfg.num_bins):
            src_dict.add_symbol("<bin_{}>".format(i))
            tgt_dict.add_symbol("<bin_{}>".format(i))

        logger.info("source dictionary: {} types".format(len(src_dict)))
        logger.info("target dictionary: {} types".format(len(tgt_dict)))
        return cls(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        if split == 'train':
            file_path = paths[(epoch - 1) % (len(paths) - 1)]
        else:
            file_path = paths[-1]
        dataset = FileDataset(file_path, self.cfg.selected_cols)

        self.datasets[split] = TableRecDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            patch_image_size=self.cfg.patch_image_size,
            imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)
        if self.cfg.eval:
            gen_args = json.loads(self.cfg.eval_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )

        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = criterion(model, sample)

        model.eval()
        if self.cfg.eval:
            hyps, refs = self._inference(self.sequence_generator, sample, model)
            score = self.teds.batch(hyps, refs)
            logging_output["_teds_score_sum"] = sum(score)
            logging_output["_teds_cnt"] = len(score)

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        if self.cfg.eval:
            def compute_teds(meters):
                teds = meters["_teds_score_sum"].sum / meters["_teds_cnt"].sum
                teds = teds if isinstance(teds, float) else teds.item()
                return round(teds, 3)

            metrics.log_scalar("_teds_score_sum", sum_logs("_teds_score_sum"))
            metrics.log_scalar("_teds_cnt", sum_logs("_teds_cnt"))

            metrics.log_derived("teds", compute_teds)

    def _inference(self, generator, sample, model):

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.bpe:
                s = self.bpe.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            decode_tokens = decode(gen_out[i][0]["tokens"])
            ref_decode_tokens = decode(
                utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            )
            hyp_html = decode_to_html(preprocess_tag_str(decode_tokens, True)).strip()
            ref_html = decode_to_html(preprocess_tag_str(ref_decode_tokens, True)).strip()

            hyps.append(hyp_html)
            refs.append(ref_html)
        if self.cfg.eval_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + ' && '.join(refs[0]))

        return hyps, refs
