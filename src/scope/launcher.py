from pathlib import Path

import hydra
from datasets import load_from_disk
from hydra.core.config_store import ConfigStore
from transformers import AutoModelForCausalLM, AutoTokenizer

from scope.config.config import Config
from scope.generation.decoding import MixtureDecoder
from scope.generation.distributed import distributed_generation
from scope.generation.preprocessing import get_scope_dataloader
from scope.utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: Config):
    set_seed(cfg.get("seed", 1234))
    main_model = AutoModelForCausalLM.from_pretrained(cfg.main_model.model_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.main_model.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = cfg.main_model.max_length
    if cfg.mixture_mode is not None:
        print("Loading noise model...")
        gen_config = cfg.generation
        if cfg.noise_model.model_path != cfg.main_model.model_path:
            noise_model = AutoModelForCausalLM.from_pretrained(
                cfg.main_model.model_path
            )
        else:
            noise_model = main_model

        model = MixtureDecoder(
            model=main_model,
            unconditional_model=noise_model,
            mixture_alpha=gen_config.mixture_alpha,
            mixture_mode=cfg.mixture_mode,
            n_untouched_logits=gen_config.mixture_n_untouched,
        )
    else:
        model = main_model
    # dataset should contain 2 columns: "main_text" and "noise_text"
    dataset = load_from_disk(cfg.data.dataset_path)

    def preprocess(x):
        x["main_input_ids"] = tokenizer(x["main_text"], truncation=False)
        x["noise_input_ids"] = tokenizer(x["noise_text"], truncation=False)
        return x

    dataset = dataset.map(preprocess)
    data_loader = get_scope_dataloader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        tokenizer=tokenizer,
        model_type=cfg.mixture_mode,
        max_samples=cfg.data.max_samples,
    )
    if cfg.data.max_samples is not None:
        print(f"Using {cfg.data.max_samples} samples")
        dataset = dataset.select(range(cfg.data.max_samples))
    tmp_path = Path(cfg.out_path / "tmp")

    distributed_generation(
        model=model,
        tokenizer=tokenizer,
        data_loader=data_loader,
        out_path=cfg.out_path,
        tmp_path=tmp_path,
        mixture_mode=cfg.mixture_mode,
        gen_config=cfg.generation,
        cfg=cfg,
    )


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    main()
