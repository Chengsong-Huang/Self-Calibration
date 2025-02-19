SPECIAL_SUFFIXS = {
    "llama": "<|eot_id|><|start_header_id|>user<|end_header_id|>Is the answer correct? (Yes/No)<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    'phi': "<|im_end|>\n<|im_start|>user<|im_sep|>\nIs the answer correct? (Yes/No)<|im_end|>\n<|im_start|>assistant<|im_sep|>",
    'qwen': "<|im_end|>\n<|im_start|>user\nIs the answer correct? (Yes/No)<|im_end|>\n<|im_start|>assistant\n",
    'ds': "<｜end▁of▁sentence｜><｜User｜>Directly answer the question by Yes or No:Is the answer correct?<｜Assistant｜></think>\n\n"
}

def get_suffix(model_name):
    if 'llama' in model_name.lower():
        return SPECIAL_SUFFIXS['llama']
    elif 'phi' in model_name.lower():
        return SPECIAL_SUFFIXS['phi']
    elif 'qwen' in model_name.lower():
        return SPECIAL_SUFFIXS['qwen']
    elif 'ds' or 'deepseek' in model_name.lower():
        return SPECIAL_SUFFIXS['ds']
    else: return SPECIAL_SUFFIXS['llama']
    
def get_split(model_name):
    if 'llama' in model_name.lower():
        return "<|start_header_id|>assistant<|end_header_id|>"
    elif 'phi' in model_name.lower():
        return "<|im_start|>assistant<|im_sep|>"
    elif 'ds' or 'deepseek' in model_name.lower():
        return '<｜Assistant｜>'
    elif 'qwen' in model_name.lower():
        return "<|im_start|>assistant\n"
    else: return "<|start_header_id|>assistant<|end_header_id|>"
    