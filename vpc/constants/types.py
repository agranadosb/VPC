from typing import Tuple, Dict, Union, Any, List, Iterable

import torch


TYPE_OBJECT = Tuple[float, float, float, float, str]
TYPE_OBJECT_YOLO = Tuple[float, float, float, float, str]
TYPE_DATA = Dict[str, Union[str, List[Union[TYPE_OBJECT, TYPE_OBJECT_YOLO]]]]

TYPE_PAIR = Tuple[Iterable, TYPE_DATA]
TYPE_PAIR_YOLO = Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]

TYPE_PAIR_SET = Union[TYPE_PAIR, TYPE_PAIR_YOLO]
