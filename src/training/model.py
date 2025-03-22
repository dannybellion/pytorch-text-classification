"""TinyBERT model for text classification."""
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


class TinyBERTClassifier(nn.Module):
    """TinyBERT-based classifier for loan default prediction."""
    
    def __init__(self, model_name: str = "huawei-noah/TinyBERT_General_6L_768D", num_labels: int = 2):
        """Initialize the TinyBERT classifier.
        
        Args:
            model_name: Name of the pre-trained TinyBERT model
            num_labels: Number of output classes
        """
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            labels: Optional labels for computing loss
            
        Returns:
            If labels are provided, returns (loss, logits)
            Otherwise, returns a dictionary with 'logits'
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        if labels is not None:
            return outputs.loss, outputs.logits
        else:
            return {"logits": outputs.logits}