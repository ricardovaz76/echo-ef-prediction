from .components import (
    RegressionHead,
    ResNetEncoder,
    TemporalConv,
    TemporalTransformer,
    UNet,
    UNetDecoder,
    replace_bn_with_gn,
)
from .echonet import EchoNetModel


FEATURE_DIM = 256


# Builds the full EchoNetModel on device, shared by training, testing and inference
# clip_length sets the transformer's positional embedding size (CLIP_LENGTH in data/clips.py)
def build_model(device, clip_length=32):
    encoder    = ResNetEncoder()
    replace_bn_with_gn(encoder)
    decoder    = UNetDecoder()
    seg_model  = UNet(encoder, decoder)
    ef_head    = RegressionHead(in_channels=2 * FEATURE_DIM + 2)

    temporal_conv = TemporalConv(feature_dim=FEATURE_DIM)
    temporal_transformer = TemporalTransformer(d_model=FEATURE_DIM, nhead=8, num_layers=2, max_len=clip_length)

    model = EchoNetModel(
        seg_model=seg_model,
        ef_head=ef_head,
        temporal_conv=temporal_conv,
        temporal_transformer=temporal_transformer,
    )
    return model.to(device)
