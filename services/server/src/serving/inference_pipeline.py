# from fastapi import FastAPI, Depends
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
#
# app = FastAPI()
#
# class ModelLoader:
#     def __init__(self):
#         self.model = None
#         self.tokenizer = None
#
#     def get_model(self):
#         if self.model is None or self.tokenizer is None:
#             print("Loading model for the first time...")
#             self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
#             self.model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda" if torch.cuda.is_available() else "cpu")
#         return self.model, self.tokenizer
#
# # Global singleton instance
# model_loader = ModelLoader()
#
# @app.post("/predict/")
# def predict(input_text: str, model_and_tokenizer=Depends(model_loader.get_model)):
#     model, tokenizer = model_and_tokenizer
#     inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
#     outputs = model.generate(**inputs, max_length=50)
#     return {"generated_text": tokenizer.decode(outputs[0], skip_special_tokens=True)}
#
#
# class InferencePipeline(object):
#     def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
#
#         self.device = device
#         print(f"Loading model '{model_name}' on {self.device}...")
#
#         # Load tokenizer and model
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(model_name)
#         self.model.to(self.device)
#         print("Model and tokenizer loaded successfully.")
#
#     def preprocess(self, text: str) -> torch.Tensor:
#
#         inputs = self.tokenizer(text, return_tensors="pt")
#         return inputs.to(self.device)
#
#     def generate(self, inputs: torch.Tensor, max_length: int = 50, **kwargs) -> str:
#
#         print("Generating text...")
#         outputs = self.model.generate(
#             **inputs, max_length=max_length, **kwargs
#         )
#         return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#
#     def postprocess(self, generated_text: str) -> str:
#
#         # Example post-processing: Strip unwanted whitespaces or fix formatting
#         return generated_text.strip()
#
#     def run(self, prompt: str, max_length: int = 50, **kwargs) -> str:
#
#         print("Starting inference pipeline...")
#         inputs = self.preprocess(prompt)
#         raw_output = self.generate(inputs, max_length=max_length, **kwargs)
#         final_output = self.postprocess(raw_output)
#         print("Inference pipeline completed.")
#         return final_output
#
