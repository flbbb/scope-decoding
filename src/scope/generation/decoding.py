import math
from typing import Optional

import torch
from torch import nn
from transformers import (
    LogitsProcessorList,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)


class MixtureLogitsProcessor(UnbatchedClassifierFreeGuidanceLogitsProcessor):
    def __init__(
        self,
        mixture_alpha: float,
        mixture_mode: str,
        unconditional_model: Optional[torch.nn.Module] = None,
        unconditional_ids: Optional[torch.LongTensor] = None,
        unconditional_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        n_untouched_logits: Optional[int] = 0,
        input_seq_length: Optional[int] = 0,
    ):
        super().__init__(
            model=unconditional_model,
            unconditional_ids=unconditional_ids,
            unconditional_attention_mask=unconditional_attention_mask,
            use_cache=use_cache,
            guidance_scale=mixture_alpha,
        )
        mixture_alpha = float(mixture_alpha)
        self.mixture_alpha = mixture_alpha
        self.mixture_mode = mixture_mode
        self.unconditional_model = unconditional_model
        self.n_untouched_logits = n_untouched_logits
        self.input_seq_length = input_seq_length
        if self.mixture_mode == "soft":
            if mixture_alpha != 0.0 and mixture_alpha != 1.0:
                self.log_alpha = math.log(mixture_alpha)
                self.log_minus_alpha = math.log(1.0 - mixture_alpha)
            else:
                self.log_alpha = 0.0
                self.log_minus_alpha = 0.0

    def __call__(self, input_ids, scores):
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        if self.mixture_alpha == 0.0:
            return scores
        step = input_ids.shape[1] - self.input_seq_length

        # needed to update the input
        logits = self.get_unconditional_logits(input_ids)[:, -1, :]
        if step < self.n_untouched_logits:
            return scores

        inf_indices = torch.isinf(scores)
        logits[inf_indices] = float("-Inf")
        unconditional_scores = torch.nn.functional.log_softmax(logits, dim=-1)
        if self.mixture_alpha == 1.0:
            return unconditional_scores

        elif self.mixture_mode == "hard":
            mask_score = torch.rand(scores.shape[0]) < self.mixture_alpha
            scores_processed = scores.clone()
            scores_processed[mask_score] = unconditional_scores[mask_score]

        elif self.mixture_mode == "cad":
            unconditional_indices = torch.isinf(unconditional_scores)
            unconditional_scores[unconditional_indices] = float("Inf")
            scores_processed = (
                1.0 + self.mixture_alpha
            ) * scores - self.mixture_alpha * unconditional_scores

        return scores_processed


class MixtureDecoder(nn.Module):
    def __init__(
        self,
        model,
        unconditional_model,
        mixture_alpha,
        mixture_mode,
        n_untouched_logits=0,
    ):
        super().__init__()
        self.mixture_alpha = mixture_alpha
        self.mixture_mode = mixture_mode
        self.unconditional_model = unconditional_model
        self.model = model
        self.n_untouched_logits = n_untouched_logits

    def generate(self, **kwargs):
        weak_inputs = kwargs.pop("weak_inputs")

        processor = MixtureLogitsProcessor(
            unconditional_model=self.unconditional_model,
            unconditional_ids=weak_inputs["input_ids"],
            unconditional_attention_mask=weak_inputs["attention_mask"],
            mixture_mode=self.mixture_mode,
            mixture_alpha=self.mixture_alpha,
            n_untouched_logits=self.n_untouched_logits,
            input_seq_length=kwargs["input_ids"].shape[1],
        )

        processor = LogitsProcessorList([processor])
        generation_output = self.model.generate(logits_processor=processor, **kwargs)

        return generation_output

    # method as attribute
    @property
    def device(self):
        return self.model.device

    def compute_transition_scores(self, *args, **kwargs):
        return self.model.compute_transition_scores(*args, **kwargs)
