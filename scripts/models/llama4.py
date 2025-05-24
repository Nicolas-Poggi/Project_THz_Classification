import torch
import cv2
import os

"""
REQUIRED PACKAGES:
pip install transformers[hf_xet]
"""

class Llama4Scores:
    def __init__(self, model=None, processor=None):
        self.model = model
        self.processor = processor

    def setup_llama4(self, prompt, max_tokens):
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        inputs = self.processor.apply_chat_template(
                                                messages, 
                                                add_generation_prompt=True, 
                                                tokenize=True, 
                                                return_dict=True, 
                                                return_tensors="pt"
                                               ).to(self.model.device, 
                                                    dtype=torch.bfloat16)
        
        generate_ids = self.model.generate(**inputs, max_new_tokens=max_tokens, eos_token_id=None, pad_token_id=self.model.config.eos_token_id)
        decoded_output = self.processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return decoded_output

    def get_answer(self, image_path, prompt, max_tokens):     
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(messages, 
                                                  tokenize=False, 
                                                  add_generation_prompt=True,
                                                  tokenize=True,
                                                  return_dict=True,
                                                  return_tensors="pt"
                                                  ).to(model.device)
        
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, eos_token_id=None, pad_token_id=self.model.config.eos_token_id)
        response = self.processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
        return response