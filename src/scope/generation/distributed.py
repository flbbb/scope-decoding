import shutil
from pathlib import Path

import idr_torch
import torch.distributed as dist
from datasets import concatenate_datasets, load_from_disk
from omegaconf import OmegaConf

from generation.generation import predict_dataset


def print_gen_out(dataset, n_print=1):
    max_samples = len(dataset)

    for i in range(min(n_print, max_samples)):
        sample = dataset[i]

        print("====================================")
        try:
            prompt = sample["input"]
        except KeyError:
            prompt = sample["content"]
        print("Prompt:\n", prompt)
        print("Real:\n", sample["reference"])
        print("------------------------------------")
        print("Gen:\n", sample["prediction"])


def initialize_distributed() -> bool:
    init_dist = idr_torch.world_size > 1
    if init_dist:
        dist.init_process_group(
            backend="mpi",
            init_method="env://",
            world_size=idr_torch.world_size,
            rank=idr_torch.rank,
        )
    return init_dist


def save_distributed_results(dataset, tmp_path):
    if idr_torch.rank == 0:
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        dataset.save_to_disk(str(tmp_path / f"predictions_{idr_torch.rank}"))
        print("saved rank 0")
    if idr_torch.world_size > 1:
        dist.barrier()

    dataset.save_to_disk(str(tmp_path / f"predictions_{idr_torch.rank}"))
    print(f"saved rank {idr_torch.rank}")


def gather_distributed_results(tmp_path):
    if idr_torch.world_size > 1:
        dist.barrier()
    if idr_torch.rank == 0:
        list_datasets = [
            load_from_disk(str(tmp_path / f"predictions_{i}"))
            for i in range(idr_torch.world_size)
        ]
        return concatenate_datasets(list_datasets, axis=0)
    return None


def add_generations_to_dataset(dataset_with_results, base_dataset):
    if idr_torch.rank == 0:
        for c in dataset_with_results.column_names:
            base_dataset = base_dataset.add_column(c, dataset_with_results[c])
        return base_dataset


def save_generation_results(dataset, out_path, cfg):
    if idr_torch.rank == 0:
        dataset.save_to_disk(str(out_path / "predictions"))
        print(f"Saved to {out_path / 'predictions'}")
        dataset.to_csv(str(out_path / "predictions.csv"))

        if cfg is not None:
            OmegaConf.save(config=cfg, f=out_path / "config.yaml")
        print_gen_out(dataset)


def distributed_generation(
    model,
    tokenizer,
    data_loader,
    out_path,
    tmp_path,
    mixture_mode=None,
    gen_config=None,
    cfg=None,
):
    initialize_distributed()

    out_path = Path(out_path)

    print("\nSaving to:", out_path, "\n")

    model.eval()

    predicted_dataset = predict_dataset(
        model,
        tokenizer,
        mixture_mode=mixture_mode,
        data_loader=data_loader,
        gen_config=gen_config,
    )
    print(f"finished rank {idr_torch.rank}")
    save_distributed_results(predicted_dataset, tmp_path)

    results = gather_distributed_results(tmp_path)

    save_generation_results(results, out_path, cfg=cfg)
