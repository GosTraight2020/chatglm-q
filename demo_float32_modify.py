import torch
from chatglm_q.decoder import ChatGLMDecoder, chat_template
import time
from chatglm_q.loader import load_model_and_tokenizer
from chatglm_q.model import Embedding
from chatglm_q.decoder import top_p_sampling, process_response
from torch.utils.mobile_optimizer import optimize_for_mobile, MobileOptimizerType
from typing import Set
import time
import copy
import gc

dtype = torch.int8
backend = 'vulkan' 
path = "/home/shanlin/pytorch-vulkan/weight_chatglm-q"
# config, ex_model, tokenizer = load_model_and_tokenizer(path, torch_dtype=torch.float, load_model=True)
decoder = ChatGLMDecoder.from_pretrained(path, torch_dtype=dtype)
ex_model = decoder.model
ex_model.eval()


for name, param in ex_model.named_parameters():
    print("{}: {}".format(name, param.dtype))

script_model = torch.jit.script(ex_model) 


optimizer_set = {MobileOptimizerType.REMOVE_DROPOUT}
print(2123)

vulkan_script_model = optimize_for_mobile(script_model,optimizer_set,  backend=backend)

print(2123)

vulkan_script_model.save('vulkan.pt')
print(2123)



