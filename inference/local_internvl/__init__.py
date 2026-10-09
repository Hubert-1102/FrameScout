# local_internvl/__init__.py
from .modeling_internlm2 import InternLM2ForCausalLM
from .modeling_internvl_chat import InternVLChatModel
from .configuration_internvl_chat import InternVLChatConfig
from .tokenization_internlm2 import InternLM2Tokenizer 
from .conversation import get_conv_template

__all__ = [
    'InternLM2ForCausalLM', 
    'InternVLChatModel', 
    'InternVLChatConfig', 
    'InternLM2Tokenizer',
    'get_conv_template'
]


#pip install transformers==4.45.2