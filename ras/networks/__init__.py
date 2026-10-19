"""Feature extractors f: price pattern P_{t,m} -> embedding h_{t,m}."""
from ras.data import PRICE_FIELDS
from ras.networks.eiie import EIIEConv, EIIELSTM
from ras.networks.hgam import HGAMBackbone
from ras.networks.lsre_caan import LSRECAANExtractor
from ras.networks.rat import make_rat
from ras.networks.transformer import TransformerFeatureExtractor


def build_extractor(cfg, model=None):
    model = model or cfg["model"]
    c, d, layers, window = len(PRICE_FIELDS), cfg["d"], cfg["num_layers"], cfg["window"]
    if model in ("transformer_eiie", "predictor"):
        return TransformerFeatureExtractor(c, d, layers, cfg["num_heads"], cfg["input_kernel"], window, cfg["causal"])
    if model == "cnn_eiie":
        return EIIEConv(c, d, layers, window)
    if model == "brnn_eiie":
        return EIIELSTM(c, d, layers)
    if model == "rat":
        return make_rat(c, d, layers, window)
    if model == "hgam":
        return HGAMBackbone(c, d, layers)
    if model == "lsre_caan":
        return LSRECAANExtractor(c, d, layers, window)
    raise ValueError(f"Unknown model: {model}")
