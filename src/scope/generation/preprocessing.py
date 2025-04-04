from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import idr_torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    label_pad_token_id = -100

    def __post_init__(self):
        self.base_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
            label_pad_token_id=self.label_pad_token_id,
        )

    def __call__(self, batch, ids_key="input_ids"):
        formatted_batch = []
        for x in batch:
            # truncate to max_length starting from the end
            if self.max_length is not None:
                formated_prompt = x[ids_key][-self.max_length :]
            else:
                formated_prompt = x[ids_key]
            input_dict = {"input_ids": formated_prompt}

            formatted_batch.append(input_dict)
        features = self.base_collator(formatted_batch)

        return features

    def decode_one(self, x, ids_key="input_ids"):
        return self.tokenizer.decode(x[ids_key][0])


class ExpertGuidedDataCollator:
    def __init__(self, main_data_collator):
        self.data_collator = main_data_collator
        self.tokenizer = main_data_collator.tokenizer

    def __call__(self, batch):
        """Returns main input and noisy input, two dictionnaries with input_ids and labels"""
        main_input = self.data_collator(batch)
        noisy_inputs = self.data_collator(batch, ids_key="noise_input_ids")

        return main_input, noisy_inputs

    def decode_one(self, x):
        main_str = self.data_collator.decode_one(x[0])
        weak_str = self.data_collator.decode_one(x[1])
        full_str = f"Main input:\n{main_str}\n\nNoise input:\n{weak_str}"
        return full_str


def get_scope_dataloader(
    dataset,
    batch_size: int,
    tokenizer: PreTrainedTokenizerBase,
    model_type: str = "standalone",
    max_samples=None,
) -> DataLoader:
    rank = idr_torch.rank
    world_size = idr_torch.world_size

    if max_samples is not None:
        dataset = dataset.select(range(max_samples))

    print("Number of queries in the dataset:", len(dataset))

    data_collator = DataCollator(
        tokenizer=tokenizer,
        max_length=tokenizer.model_max_length,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    if model_type == "mixture":
        data_collator = ExpertGuidedDataCollator(data_collator)

    chunk_dataset = dataset.shard(num_shards=world_size, index=rank, contiguous=True)
    batch_size = min(batch_size, len(chunk_dataset))

    print(f"Batch size: {batch_size}, len(chunk_dataset): {len(chunk_dataset)}")
    print(f"Will truncate to max_length: {tokenizer.model_max_length}")

    data_loader = DataLoader(
        dataset=chunk_dataset,
        batch_size=min(batch_size, len(chunk_dataset)),
        drop_last=False,
        shuffle=False,
        collate_fn=data_collator,
    )

    for x in data_loader:
        print(data_collator.decode_one(x))
        break

    return data_loader
