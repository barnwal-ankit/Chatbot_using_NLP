Chatbot using NLP
📌 Description
This project is a Natural Language Processing (NLP) chatbot developed as part of a 4-week AICTE internship under Edunet Foundation. The chatbot is trained to understand user queries and provide relevant responses. It utilizes TF-IDF vectorization and Logistic Regression for intent classification.

🚀 Features
✔️ Handles multiple intents like greetings, finance, investing, freelancing, negotiation, etc.
✔️ Uses TF-IDF to transform text data into numerical form.
✔️ Implements Logistic Regression for intent classification.
✔️ Tokenizes input using NLTK’s word_tokenize.
✔️ Built with Streamlit for an interactive web interface.

🛠️ Technologies & Libraries Used
📌 Python – Core programming language
📌 NLTK – Tokenization & text processing
📌 Scikit-learn – TF-IDF vectorization & Logistic Regression
📌 NumPy – Data manipulation
📌 Streamlit – Web app framework for chatbot UI

💻 Installation Guide
1️⃣ Clone the Repository

bash
Copy
Edit
git clone https://github.com/RGS-AI/Chatbot_using_NLP_AICTE_Cycle4.git
cd Chatbot_using_NLP_AICTE_Cycle4
2️⃣ Install Required Dependencies

bash
Copy
Edit
pip install -r requirements.txt
📜 requirements.txt
If you need to create a requirements.txt, add:

nginx
Copy
Edit
nltk
numpy
scikit-learn
streamlit
3️⃣ Download NLTK Data (if not already installed)

python
Copy
Edit
import nltk
nltk.download('punkt')
4️⃣ Run the Chatbot

bash
Copy
Edit
streamlit run app.py
📝 How It Works
User Input → Tokenized using word_tokenize from NLTK
Feature Extraction → TfidfVectorizer converts text to numerical format
Model Prediction → LogisticRegression classifies intent
Response Generation → Chatbot selects a relevant response from predefined intents
🤝 Contribution
Feel free to fork, raise issues, or submit pull requests to enhance this project!

📜 License
This project is licensed under the MIT License.

