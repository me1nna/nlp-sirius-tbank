import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

class GeneratorModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = 'hivaze/AAQG-QA-QG-FRED-T5-1.7B'
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(self.device)

    def generate_text(self, context, question, n=1, temperature=0.8, num_beams=3):
        prompt = f"Сгенерируй ответ на вопрос по тексту. Текст: '{context}'. Вопрос: '{question}'."
        encoded_input = self.tokenizer.encode_plus(prompt, return_tensors='pt')
        encoded_input = {k: v.to(self.model.device) for k, v in encoded_input.items()}
        resulted_tokens = self.model.generate(**encoded_input,
                                              max_new_tokens=64,
                                              do_sample=True,
                                              num_beams=num_beams,
                                              num_return_sequences=n,
                                              temperature=temperature,
                                              top_p=0.9,
                                              top_k=50)
        resulted_texts = self.tokenizer.batch_decode(resulted_tokens, skip_special_tokens=True)
        return resulted_texts[0]
