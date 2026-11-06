from transformers import AutoTokenizer, LlamaForCausalLM

from . import register_llm


@register_llm('mobile_llama')
@register_llm('mobile-llama')
@register_llm('mobilellama')
def return_mobilellamaclass():
    def tokenizer_and_post_load(tokenizer):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
        return tokenizer

    return LlamaForCausalLM, (AutoTokenizer, tokenizer_and_post_load)
