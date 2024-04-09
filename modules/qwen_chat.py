import torch
from typing import List, Union, Optional, Any
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen1.5-0.5B"
max_new_tokens = 500
temperature = 0.1
device = 'cuda'

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map=device
).to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    # llama不支持fast
    use_fast=False if model.config.model_type == 'llama' else True
)

def qwen_response(model, tokenizer, max_new_tokens, temperature, device, text):
    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    #bos_token_id = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(device)
    # eos_token_id = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long).to(device)
    #input_ids = torch.concat([bos_token_id, input_ids, eos_token_id], dim=1)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature
        )
    outputs = outputs.tolist()[0][len(input_ids[0]):]
    response = tokenizer.decode(outputs)
    response = response.strip().replace("<|im_end|>", "").replace("<|im_start|>", "").replace("<|endoftext|>", "").strip()
    
    return response

# print(qwen_response(model, tokenizer, text))

class QwenChatModel(BaseChatModel):
    """
    Simple Qwen Chat Model.
    这里有个比较矛盾的地方，记录一下。首先，BaseMessage 是一个 Union，类型为 str 或者 List(Union(str, dict))。但是在此处使用的时候，后文定义的正则处理函数指定了 model 的 output 类型强制为 str，所以这里把 output_str 写成 list 的形式就会报错。改成单条 string 了。
    """
    def messages_to_str(
        self,
        messages: List[BaseMessage],    
    ) -> List[str]:
        str_messages = []
        for msg in messages:
            if type(msg.content) == list:
                for m in msg.content:
                    str_messages.append(str(m))
            else:
                str_messages.append(str(msg.content))
        
        return str_messages

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # output_str = []
        # for msg in self.messages_to_str(messages):
        #     response =  qwen_response(model, tokenizer, msg)# self._call(messages, stop=stop, run_manager=run_manager, **kwargs)
        #     output_str.append(response)
        # print(f"type:{type(output_str)}, content:{output_str}")
        # message = AIMessage(content=output_str)
        output_str = qwen_response(
            model, 
            tokenizer, 
            max_new_tokens, 
            temperature, 
            device, 
            self.messages_to_str(messages)[0]
            )
        message = AIMessage(content=output_str)
        generation = ChatGeneration(message=message)

        return ChatResult(generations=[generation])
    
    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "qwen-chat-model"