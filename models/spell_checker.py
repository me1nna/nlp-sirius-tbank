import torch
from transformers import T5TokenizerFast, AutoModelForSeq2SeqLM

class SpellCheckerModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = 'UrukHan/t5-russian-spell'
        self.tokenizer = T5TokenizerFast.from_pretrained(checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(self.device)

    def correct_spelling(self, input_text):
        task_prefix = "Spell correct: "
        encoded = self.tokenizer([task_prefix + input_text], return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            predicted = self.model.generate(**encoded)
        corrected_text = self.tokenizer.decode(predicted[0], skip_special_tokens=True)
        return corrected_text
