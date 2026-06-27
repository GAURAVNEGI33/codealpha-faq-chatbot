"""
CodeAlpha Internship - Task 2: NLP-based FAQ Chatbot
------------------------------------------------------
Tech used: Python, NLTK (preprocessing), scikit-learn (TF-IDF + Cosine Similarity)

How it works:
1. Load FAQ question-answer pairs from faqs.json
2. Preprocess all questions (lowercase, remove punctuation, remove stopwords, lemmatize)
3. Convert all FAQ questions into TF-IDF vectors
4. When user asks a question, preprocess it the same way and convert to a vector
5. Compute cosine similarity between user question and all FAQ questions
6. Return the answer of the FAQ with the highest similarity score (if above threshold)
"""

import json
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- One-time NLTK downloads ----------
def setup_nltk():
    required = ['punkt', 'punkt_tab', 'stopwords', 'wordnet']
    for item in required:
        try:
            nltk.data.find(item)
        except LookupError:
            nltk.download(item, quiet=True)

setup_nltk()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess(text):
    """Lowercase, tokenize, remove punctuation & stopwords, lemmatize."""
    text = text.lower()
    tokens = word_tokenize(text)
    cleaned = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in string.punctuation and token not in stop_words
    ]
    return " ".join(cleaned)


class FAQChatbot:
    def __init__(self, faq_path="faqs.json", threshold=0.2):
        with open(faq_path, "r", encoding="utf-8") as f:
            self.faqs = json.load(f)

        self.questions = [item["question"] for item in self.faqs]
        self.answers = [item["answer"] for item in self.faqs]
        self.threshold = threshold

        # Preprocess all FAQ questions
        self.processed_questions = [preprocess(q) for q in self.questions]

        # Build TF-IDF vectorizer on FAQ questions
        self.vectorizer = TfidfVectorizer()
        self.faq_vectors = self.vectorizer.fit_transform(self.processed_questions)

    def get_response(self, user_query):
        processed_query = preprocess(user_query)
        user_vector = self.vectorizer.transform([processed_query])

        similarities = cosine_similarity(user_vector, self.faq_vectors)[0]
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]

        if best_score < self.threshold:
            return ("Sorry, I couldn't find a relevant answer to that. "
                    "Could you please rephrase your question?")

        return self.answers[best_idx]


def main():
    print("=" * 55)
    print(" FAQ CHATBOT (CodeAlpha Task 2) ")
    print(" Type 'exit' or 'quit' to end the chat ")
    print("=" * 55)

    bot = FAQChatbot(faq_path="faqs.json")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Bot: Goodbye! 👋")
            break
        if not user_input:
            continue
        response = bot.get_response(user_input)
        print(f"Bot: {response}")


if __name__ == "__main__":
    main()