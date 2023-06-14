import torch

from sample_transformer import TransformerTest


def colossal(model, optimizer, criterion):
    pass
    # booster = Booster(plugin=plugin)
    # model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)


def create_model(d_model=1024, num_heads=12, num_layer=12, vocab_size=32000, dtype=torch.float16):
    model = TransformerTest(d_model, num_heads, num_layer, vocab_size, dtype=dtype)
    return model

def train():
    pass
