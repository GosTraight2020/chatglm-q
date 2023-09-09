import torch
from chatglm_q.decoder import ChatGLMDecoder, chat_template
import time
from chatglm_q.decoder import top_p_sampling, process_response
from torch.utils.mobile_optimizer import optimize_for_mobile
import time


def generate(model, tokenizer, input_ids, eos_token_id, max_sequence_length, max_generated_tokens=100, top_k=100, top_p=0.8, temperature=1.0):

    generated_tokens = []
    generate_time = []
    past_key_values = None
    response = []
    len_prefix_ids = input_ids.size()[1]
    while len(generated_tokens) < max_generated_tokens \
        and len(generated_tokens) + len_prefix_ids < max_sequence_length:

        with torch.inference_mode():
            start_time = time.perf_counter()
            print(444)
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

        input_ids = torch.tensor([[next_token]]).int()

    result = process_response(tokenizer.decode(generated_tokens))
    return result, past_key_values


decoder = ChatGLMDecoder.from_pretrained("/home/shanlin/pytorch-vulkan/weight_chatglm-q", torch_dtype=torch.float)
ex_model = decoder.model
torch.save(ex_model.state_dict, 'fp32.pt');
tokenizer = decoder.tokenizer

ex_model.eval()
script_model = torch.jit.script(ex_model) 
history = []


# while True:    
#     # query = input("请输入：")
#     prompt = chat_template(history, query)
#     prefix_ids = decoder.tokenizer.encode(prompt)
#     input_ids = torch.LongTensor([prefix_ids])
#     response, past_key_value = generate(script_model, tokenizer, input_ids, decoder.eos_token_id, decoder.max_sequence_length)
#     print(response)
#     history = history + [(query, response)]
#     print(history)
#     prompt = chat_template(history, query)
#     prefix_ids = decoder.tokenizer.encode(prompt)
#     input_ids = torch.LongTensor([prefix_ids])