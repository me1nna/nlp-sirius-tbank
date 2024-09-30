import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.neighbors import NearestNeighbors

class RetrieverModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer("BAAI/bge-m3", device=self.device)
        self.model.max_seq_length = 512
        self.dataset = load_dataset("kuznetsoffandrey/sberquad")
        self.contexts = list(set(self.dataset['train']['context']))
        self.chunked_contexts = self.chunk_text(self.contexts)
        self.embeddings = self.model.encode(self.chunked_contexts, convert_to_tensor=True).detach().cpu().numpy()
        self.knn = NearestNeighbors(metric='cosine')
        self.knn.fit(self.embeddings)

    def chunk_text(self, texts, chunk_size=512):
        chunked_texts = []
        for text in texts:
            chunks = textwrap.wrap(text, chunk_size)
            chunked_texts.extend(chunks)
        return chunked_texts

    def retrieve_context(self, question):
        question_embedding = self.model.encode([question], convert_to_tensor=True).detach().cpu().numpy()
        distances, indices = self.knn.kneighbors(question_embedding, n_neighbors=1)
        relevant_context = self.chunked_contexts[indices[0][0]]
        return relevant_context
