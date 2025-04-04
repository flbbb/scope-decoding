from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Model:
    model_path: str
    max_length: int = 2048


@dataclass
class Data:
    dataset_path: str
    max_samples: Optional[int] = None


@dataclass
class Generation:
    min_new_tokens: int = 1
    max_new_tokens: int = 128
    do_sample: bool = True
    num_beams: int = 1
    temperature: float = 0.6
    top_p: float = 0.9
    mixture_alpha: float = 0.5
    mixture_n_untouched: int = 2


@dataclass
class Config:
    main_model: Model = field(Model)
    data: Data = field(Data)
    noise_model: Optional[Model] = None
    mixture_mode: Optional[str] = "hard"
    generation: Generation = field(Generation)

    batch_size: int = 8
    out_path: str
