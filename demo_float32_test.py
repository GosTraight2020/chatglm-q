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


def generate(model, tokenizer, input_ids, input_embedding, eos_token_id, max_sequence_length, max_generated_tokens=100, top_k=100, top_p=0.8, temperature=1.0):
    total_time = 0
    generated_tokens = []
    generate_time = []
    past_key_values = None
    eos_token = "</s>",
    eos_token_id = tokenizer(eos_token) # TODO
    response = []
    len_prefix_ids = input_ids.size()[1]
    while len(generated_tokens) < max_generated_tokens \
        and len(generated_tokens) + len_prefix_ids < max_sequence_length:

        with torch.inference_mode():
            start_time = time.perf_counter()
            _, logits, past_key_values = model(
                input_embeddings=input_embedding,
                past_key_values=past_key_values,
            )
        logits = logits.to(device='cpu')
        next_token = top_p_sampling(logits[0, -1], top_k, top_p, temperature).item()
        end_time = time.perf_counter()
        current_time = end_time - start_time
        generate_time.append(current_time)
        print(current_time)


        generated_tokens += [next_token]
        if next_token == eos_token_id:
            break

        total_time+= current_time
        mean_time = total_time / len(generated_tokens)
        print('mean time : {}'.format(mean_time))

        response_text = process_response(tokenizer.decode(generated_tokens))
        if response_text and response_text[-1] != "�":
            print(response_text)
            response.append(response_text)

        input_ids = torch.tensor([[next_token]]).int()
        input_embedding = embedding_model(input_ids)
        input_embedding = input_embedding.to(torch.device('vulkan'))

    mean_time = total_time / len(generated_tokens)
    print('mean time : {}'.format(mean_time))
    result = process_response(tokenizer.decode(generated_tokens))
    return result, past_key_values

debug = True
dtype = torch.int8
backend = 'vulkan'
path = "/home/shanlin/pytorch-vulkan/weight_chatglm-q"
config, model, tokenizer = load_model_and_tokenizer(path, torch_dtype=dtype, load_model=True)
decoder = ChatGLMDecoder.from_pretrained(path, torch_dtype=dtype)
model_config = config.model_config

# embedding_model = QEmbedding(
#     num_embeddings=model_config.vocab_size, embedding_dim=model_config.hidden_size, dtype=dtype
# )

# embedding_model.load_state_dict(torch.load('./embedding.pt'))
embedding_model = copy.deepcopy(model.word_embedding)
torch.save(embedding_model.state_dict(), 'embedding.pt')

history = []

# query = "你好呀,很高兴认识你！"
# prompt = chat_template(history, query)
# prefix_ids = tokenizer.encode(prompt)
# input_ids = torch.tensor([prefix_ids], dtype=torch.long)
embedding_model.eval()
# input_embedding = embedding_model(input_ids)
# print(input_embedding.dtype)

# input_embedding = input_embedding.to(torch.device('vulkan'))

vulkan_script_model = torch.jit.load('vulkan.pt')


with open('./chatglm_vulkan_graph.txt', 'w') as f:
    f.write(str(vulkan_script_model.graph))



# with torch.inference_mode():
#     vulkan_response, _ = generate(vulkan_script_model, tokenizer, input_ids, input_embedding, None, config.model_config.max_sequence_length)

with torch.inference_mode():
    while True:    

        query = input("请输入：")
        prompt = chat_template(history, query)
        prefix_ids = decoder.tokenizer.encode(prompt)
        input_ids = torch.LongTensor([prefix_ids])
        input_embedding = embedding_model(input_ids)
        input_embedding = input_embedding.to(torch.device('vulkan'))
        response, past_key_value = generate(vulkan_script_model, tokenizer, input_ids, input_embedding, None, config.model_config.max_sequence_length)
        print(response)
        history = history + [(query, response)]


