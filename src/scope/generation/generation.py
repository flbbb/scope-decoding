import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import GenerationConfig


def prepare_generation_inputs(model, mixture_mode, batch):
    if mixture_mode is not None:
        # for contrastive decoding etc

        generation_inputs = {
            "input_ids": batch[0]["input_ids"].to(model.device),
            "attention_mask": batch[0]["attention_mask"].to(model.device),
            "weak_inputs": {
                "input_ids": batch[1]["input_ids"].to(model.device),
                "attention_mask": batch[1]["attention_mask"].to(model.device),
            },
        }

    else:
        generation_inputs = {
            "input_ids": batch["input_ids"].to(model.device),
            "attention_mask": batch["attention_mask"].to(model.device),
        }
    return generation_inputs


def print_generated_sequences(sequences_w_instructions, tokenizer):
    print("Generated sequences:")
    for seq in sequences_w_instructions[:5]:
        print(
            tokenizer.decode(
                seq, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
        )
        print("\n\n")
        print("====================================")
        print("\n\n")


def predict_dataset(
    model,
    tokenizer,
    data_loader,
    mixture_mode=None,
    gen_config=None,
    dtype=torch.bfloat16,
):
    generation_config = GenerationConfig(
        return_dict_in_generate=True, output_scores=True, **gen_config
    )
    sequences_wo_instuctions = []
    print("Running generation")
    for it, batch in enumerate(tqdm(data_loader)):
        generation_inputs = prepare_generation_inputs(
            model, mixture_mode=mixture_mode, batch=batch
        )

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype):
                generation_out = model.generate(
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    **generation_inputs,
                )

        input_len = generation_inputs["input_ids"].shape[1]
        sequences_w_instructions = generation_out.sequences.cpu().tolist()

        sequences_wo_instuctions += generation_out.sequences.cpu()[
            :, input_len:
        ].tolist()

        if it == 2:
            print_generated_sequences(sequences_w_instructions, tokenizer)

    # truncate sequences to remove padding from generation

    for i in range(len(sequences_wo_instuctions)):
        if tokenizer.eos_token_id in sequences_wo_instuctions[i]:
            stop_idx = sequences_wo_instuctions[i].index(tokenizer.eos_token_id)
        else:
            stop_idx = len(sequences_wo_instuctions[i])

        sequences_wo_instuctions[i] = sequences_wo_instuctions[i][: stop_idx + 1]

    gen_dataset = Dataset.from_dict(
        {
            "prediction_ids": sequences_wo_instuctions,
        }
    )

    gen_dataset = gen_dataset.map(
        lambda x: {
            "prediction": tokenizer.batch_decode(
                x["prediction_ids"],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        },
        batched=True,
        batch_size=100,
    )

    return gen_dataset
