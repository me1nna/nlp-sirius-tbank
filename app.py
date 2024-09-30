from flask import Flask, request, jsonify
from models.retriever import RetrieverModel
from models.generator import GeneratorModel
from models.toxicity import ToxicityModel
from models.spell_checker import SpellCheckerModel

app = Flask(__name__)

# Инициализация моделей
retriever_model = RetrieverModel()
generator_model = GeneratorModel()
toxicity_model = ToxicityModel()
spell_checker_model = SpellCheckerModel()

@app.route('/get_answer', methods=['POST'])
def get_answer():
    data = request.json
    question = data.get('question')

    if not question:
        return jsonify({"error": "Question is required"}), 400

    # Проверка на токсичность
    if toxicity_model.detect_toxicity(question):
        return jsonify({"message": "Please ask your question politely."}), 400

    # Исправление орфографии
    question_corrected = spell_checker_model.correct_spelling(question)

    # Поиск релевантного контекста
    context = retriever_model.retrieve_context(question_corrected)

    # Генерация ответа
    answer = generator_model.generate_text(context, question_corrected)

    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
