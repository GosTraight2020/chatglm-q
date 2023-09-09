import torch
from chatglm_q.decoder import ChatGLMDecoder, chat_template
import time
from chatglm_q.decoder import top_p_sampling, process_response
# from torch.utils.mobile_optimizer import optimize_for_mobile
from chatglm_q.loader import ChatGLMLoadConfig, load_model_and_tokenizer, save_model_and_tokenizer
from chatglm_q.int8.quantizer import get_quant_int8_linear, get_quant_embedding
import torch.nn as nn
from pympler import asizeof
from tqdm.auto import tqdm
import sys



def generate(model, tokenizer, input_ids, eos_token_id, max_sequence_length, max_generated_tokens=100, top_k=100, top_p=0.8, temperature=1.0):
    generated_tokens = []
    generate_time = []
    past_key_values = None
    response = []
    len_prefix_ids = input_ids.size()[1]
    while len(generated_tokens) < max_generated_tokens \
        and len(generated_tokens) + len_prefix_ids < max_sequence_length:

        with torch.no_grad():
            start_time = time.perf_counter()
            _, logits, past_key_values = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
            )
            next_token = top_p_sampling(logits[0, -1], top_k, top_p, temperature).item()
            end_time = time.perf_counter()
            generate_time.append(end_time - start_time)

        generated_tokens += [next_token]
        if next_token == eos_token_id:
            break

        response_text = process_response(tokenizer.decode(generated_tokens))
        if response_text and response_text[-1] != "�":
            print(response_text)
            response.append(response_text)

        input_ids = torch.tensor([[next_token]]).long()

    result = process_response(tokenizer.decode(generated_tokens))
    return result, past_key_values



# optionally pass a `torch_dtype=torch.float16` to set the activation dtype
decoder = ChatGLMDecoder.from_pretrained("/Users/liushanlin/.cache/huggingface/hub/models--K024--chatglm2-6b-int8/snapshots/836cd93aa971aa78afd106ff337f4536c5180a0d/", torch_dtype=torch.float)
# decoder = ChatGLMDecoder.from_pretrained("K024/chatglm2-6b-int8", torch_dtype=torch.float)
model = decoder.model
size_with_sys = sys.getsizeof(model)
print(size_with_sys)
tokenizer = decoder.tokenizer

model.eval()
for name, param in model.named_parameters():
    print(f"Parameter name: {name}, Data type: {param.dtype}")
script_model = torch.jit.script(model)
# script_model = optimize_for_mobile(script_model, backend='vulkan')
# print(script_model.graph)
# script_model.save("chatglm.pt")
history = []


while True:    
    query = input("请输入：")
    prompt = chat_template(history, query)
    prefix_ids = decoder.tokenizer.encode(prompt)
    input_ids = torch.LongTensor([prefix_ids])
    response, past_key_value = generate(script_model, tokenizer, input_ids, decoder.eos_token_id, decoder.max_sequence_length)
    print(response)
    history = history + [(query, response)]
    print(history)
    prompt = chat_template(history, query)
    prefix_ids = decoder.tokenizer.encode(prompt)
    input_ids = torch.LongTensor([prefix_ids])
