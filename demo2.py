import torch
from chatglm_q.decoder import ChatGLMDecoder, chat_template
import time
from chatglm_q.decoder import top_p_sampling, process_response


def generate(model, tokenizer, input_ids, eos_token_id, max_sequence_length, max_generated_tokens=10, top_k=100, top_p=0.8, temperature=1.0):
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
            # print(response_text)
            response.append(response_text)

        input_ids = torch.tensor([[next_token]]).long()

    result = process_response(tokenizer.decode(generated_tokens))
    return result, past_key_values



# optionally pass a `torch_dtype=torch.float16` to set the activation dtype
decoder = ChatGLMDecoder.from_pretrained("/Users/liushanlin/.cache/huggingface/hub/models--K024--chatglm2-6b-int4g32/snapshots/22ff2dc454baf70db47308a93cd137afda8e68fe/", torch_dtype=torch.float)
# decoder = ChatGLMDecoder.from_pretrained("K024/chatglm2-6b-int4g32", torch_dtype=torch.float)
model = decoder.model
tokenizer = decoder.tokenizer
model.eval()
script_model = torch.jit.script(model)
# script_model.save("chatglm.pt")
history = []
prompt = chat_template(history, "鸡你太美？")
prefix_ids = decoder.tokenizer.encode(prompt)

input_ids = torch.LongTensor([prefix_ids])
response, past_key_value = generate(script_model, tokenizer, input_ids, decoder.eos_token_id, decoder.max_sequence_length)
print(response)
# while True:
    
#     response, past_key_value = generate(script_model, tokenizer, input_ids, decoder.eos_token_id, decoder.max_sequence_length)
#     print(response)
#     intput_word = input("请输入：")
#     prompt = chat_template(history, intput_word)
#     prefix_ids = decoder.tokenizer.encode(prompt)
#     input_ids = torch.LongTensor([prefix_ids])
