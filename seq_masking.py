from dataclasses import dataclass

import torch
import torch.nn as nn

from typing import Dict, Optional, Any, Iterable


@dataclass
class MaskingInfo:
    schema: torch.Tensor
    targets: torch.Tensor


class MaskSequence(nn.Module):
    """
    Base class for masking inputs
    There are some variants:
        - Causal LM (clm)
        - Masked LM (mlm)
        - Permutation LM (plm)
        - Replacement Token Detection (rtd)
    This class can be extended to add different masking schems
    I use MLM.
    """
    def __init__(self, hidden_size: int,
                 padding_idx: int = 0,
                 eval_on_last_item_only: bool = True,
                 **kwargs):
        super(MaskSequence, self).__init__()
        self.padding_idx = padding_idx
        self.hidden_size = hidden_size
        self.eval_on_last_item_only = eval_on_last_item_only
        self.mask_schema: Optional[torch.Tensor] = None
        self.masked_targets: Optional[torch.Tensor] = None

        self.masked_item_embedding = nn.Parameter(torch.Tensor(self.hidden_size))
        torch.nn.init.normal_(self.masked_item_embedding, mean=0, std=.001)

    def compute_masked_targets(self, item_ids: torch.Tensor, training=False) -> MaskingInfo:
        assert item_ids.ndim == 2
        masking_info = self._compute_masked_targets(item_ids, training=training)
        self.mask_schema, self.masked_targets = masking_info.schema, masking_info.targets
        return masking_info

    def apply_mask_to_inputs(self, inputs: torch.Tensor, schema: torch.Tensor) -> torch.Tensor:
        # выбираем по условию какой элемент маскировать
        inputs = torch.where(schema.unsqueeze(-1).bool(),
                             self.masked_item_embedding.to(inputs.dtype),
                             inputs)
        return inputs

    def predict_all(self, items_ids: torch.Tensor) -> MaskingInfo:
        # уменьшаем размерность
        labels = items_ids[:, 1:]
        # дополняем маскированным элементом
        labels = torch.cat([labels, torch.zeros((labels.shape[0], 1), dtype=labels.type).to(
            items_ids.device)], axis=-1)

        mask_labels = labels != self.padding_idx
        return MaskingInfo(mask_labels, labels)

    def forward(self, inputs: torch.Tensor, item_ids: torch.Tensor, training: bool=False) -> torch.Tensor:
        mask_info = self.compute_masked_targets(item_ids=item_ids, training=training)
        if mask_info.schema is None:
            raise  ValueError("mask_schema must be set.")
        return self.apply_mask_to_inputs(inputs, mask_info.schema)

    def forward_output_size(self, inout_size):
        return inout_size

    def transformer_required_arguments(self) -> Dict[str, Any]:
        return {}

    def transformer_optional_arguments(self) -> Dict[str, Any]:
        return {}

    @property
    def transformer_arguments(self) -> Dict[str, Any]:
        return {**self.transformer_required_arguments(),
                **self.transformer_optional_arguments()}


class MaskedLanguageModeling(MaskSequence):

