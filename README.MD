FAQ Chatbot (NLP-based)

CodeAlpha Internship – Task 2

A simple command-line chatbot that answers user questions about college B.Tech admissions (eligibility, fees, hostel, scholarships, placements, etc.) by matching them against predefined FAQs using NLP preprocessing and cosine similarity.

Tech Stack


Python
NLTK (tokenization, stopword removal, lemmatization)
scikit-learn (TF-IDF vectorization + Cosine Similarity)


How It Works


FAQ question-answer pairs are loaded from faqs.json
Each FAQ question is preprocessed: lowercased, tokenized, punctuation & stopwords removed, lemmatized
All FAQ questions are converted into TF-IDF vectors
When the user types a question, it goes through the same preprocessing and vectorization
Cosine similarity is calculated between the user's question and every FAQ question
The FAQ with the highest similarity score is returned as the answer (if above a similarity threshold)


How to Run

bashpip install nltk scikit-learn
python chatbot.py

On first run, NLTK will automatically download required data (punkt, stopwords, wordnet).

Example

You: What is the eligibility for B.Tech admission?
Bot: Candidates must have passed 12th grade with Physics, Chemistry, and Mathematics...

You: exit
Bot: Goodbye! 👋

Files


chatbot.py — main chatbot logic
faqs.json — FAQ dataset (question-answer pairs)


Future Improvements


Add a web-based UI using Flask
Use word embeddings (Word2Vec/BERT) instead of TF-IDF for better semantic matching