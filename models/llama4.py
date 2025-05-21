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

    def get_answer(self, image_path=None, prompt=None):     
        
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
        text = self.processor.apply_chat_template(messages, 
                                                  tokenize=False, 
                                                  add_generation_prompt=True,
                                                  tokenize=True,
                                                  return_dict=True,
                                                  return_tensors="pt"
                                                  ).to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
        )

        response = processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
        return response