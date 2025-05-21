import torch
import cv2
import os
from PIL import Image


class MistralScores:
    def __init__(self, model=None, processor=None):
        self.model = model
        self.processor = processor
        
    def setup_mistral(self, prompt=None):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        inputs = self.processor.apply_chat_template(messages, 
                                               add_generation_prompt=True, 
                                               tokenize=True, return_dict=True, 
                                               return_tensors="pt"
                                               ).to(self.model.device, 
                                                    dtype=torch.bfloat16)
        #max new tokens is set to 512 - Before
        generate_ids = self.model.generate(**inputs, max_new_tokens=1024, eos_token_id=None, pad_token_id=self.model.config.eos_token_id)
        decoded_output = self.processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return decoded_output
    
    def get_answer(self, image_path=None, prompt=None):     
        image = Image.open(image_path).convert("RGB")
        if image is None:
            raise ValueError(f"Image at path {image_path} could not be read.")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        
        inputs = self.processor.apply_chat_template(messages, 
                                               add_generation_prompt=True, 
                                               tokenize=True, return_dict=True, 
                                               return_tensors="pt"
                                               ).to(self.model.device, 
                                                    dtype=torch.bfloat16)
        
        generate_ids = self.model.generate(**inputs, max_new_tokens=1024, eos_token_id=None, pad_token_id=self.model.config.eos_token_id)
        decoded_output = self.processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        return decoded_output