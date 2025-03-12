from .modeling_phi3 import Phi3RotaryEmbedding
from transformers.models import phi3

phi3.modeling_phi3.Phi3RotaryEmbedding = Phi3RotaryEmbedding
