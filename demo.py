import torch
from chatglm_q.decoder import ChatGLMDecoder, chat_template
import time
from chatglm_q.decoder import top_p_sampling, process_response
# device = torch.device("cuda")
# optionally pass a `torch_dtype=torch.float16` to set the activation dtype
# decoder = ChatGLMDecoder.from_pretrained("K024/chatglm2-6b-int4g32", torch_dtype=torch.float)

# prompt = chat_template([], "鸡你太美？")
# for text in decoder.generate(prompt):
#     print(text)
    
class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder =ChatGLMDecoder.from_pretrained("K024/chatglm2-6b-int4g32", torch_dtype=torch.float)
        self.decoder.model.eval()
        # self.device = torch.device('mps')

    def generate(self, input_ids, max_generated_tokens=4, top_k=100, top_p=0.8, temperature=1.0):
        eos_token_id = self.decoder.eos_token_id
        
        generated_tokens = []
        generate_time = []
        past_key_values = None
        response = []
        len_prefix_ids = input_ids.size()[1].item()
        print(type(len_prefix_ids))
        print(len_prefix_ids)
        while len(generated_tokens) < max_generated_tokens \
            and len(generated_tokens) + len_prefix_ids < self.decoder.max_sequence_length:

            with torch.no_grad():
                start_time = time.perf_counter()
                _, logits, past_key_values = self.decoder.model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                )
                next_token = top_p_sampling(logits[0, -1], top_k, top_p, temperature).item()
                end_time = time.perf_counter()
                generate_time.append(end_time - start_time)

            generated_tokens += [next_token]
            if next_token == eos_token_id:
                break

            response_text = process_response(self.decoder.tokenizer.decode(generated_tokens))
            if response_text and response_text[-1] != "�":
                # print(response_text)
                response.append(response_text)

            input_ids = torch.tensor([[next_token]]).long()

        result = process_response(self.decoder.tokenizer.decode(generated_tokens))
        return result
    
    
prompt = chat_template([], "鸡你太美？")

warp = Wrapper()

model, tokenizer = warp.decoder.model, warp.decoder.tokenizer

prefix_ids = tokenizer.encode(prompt)

input_ids = torch.LongTensor([prefix_ids])
input_ids.requires_grad_(False)
print(input_ids.size())

# response = warp(input_ids)
warp.decoder.model.eval()
warp.eval()
trace_model = torch.jit.trace(warp, (input_ids,))
trace_model.save('./chatglm.pt')
response = trace_model(input_ids)

print(response)