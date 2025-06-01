import torch
from qwen_vl_utils import process_vision_info
import cv2
import os

"""
REQUIRED PACKAGES:
pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
pip install git+https://github.com/huggingface/transformers accelerate
pip install qwen-vl-utils[decord]==0.0.8
"""

class QwenScores:
    def __init__(self, model=None, processor=None):
        self.model = model
        self.processor = processor

    def setup_qwen(self, prompt, max_tokens):
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

        inputs = self.processor.apply_chat_template(messages,add_generation_prompt=True, tokenize=False)

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[inputs],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        print("\n\n",generated_ids,"\n\n")
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        generated_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        generated_text = generated_text[0]
        return generated_text