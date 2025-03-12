# 🤖 Chatbot using NLP  

## 📌 Description  
This project is a **Natural Language Processing (NLP) chatbot** developed as part of a **4-week AICTE internship** under **Edunet Foundation**. The chatbot is trained to understand user queries and provide relevant responses. It utilizes **TF-IDF vectorization** and **Logistic Regression** for intent classification.  

---

## 🚀 Features  
✔️ Handles multiple intents like **greetings, finance, investing, freelancing, negotiation**, etc.  
✔️ Uses **TF-IDF** to transform text data into numerical form.  
✔️ Implements **Logistic Regression** for intent classification.  
✔️ Tokenizes input using **NLTK’s `word_tokenize`**.  
✔️ Built with **Streamlit** for an interactive web interface.  

---

## 🛠️ Technologies & Libraries Used  
📌 **Python** – Core programming language  
📌 **NLTK** – Tokenization & text processing  
📌 **Scikit-learn** – TF-IDF vectorization & Logistic Regression  
📌 **NumPy** – Data manipulation  
📌 **Streamlit** – Web app framework for chatbot UI  

---
## 💻 Installation Guide  

### 1️⃣ Clone the Repository  
Run the following command in your terminal:  

```bash
git clone https://github.com/barnwal-ankit/Chatbot_using_NLP.git
cd Chatbot_using_NLP
```

### 2️⃣ Install Required Dependencies  
Ensure you have Python 3.7+ installed. Then, install the required dependencies:  

```bash
pip install -r requirements.txt
```

If the `requirements.txt` file is missing, manually install the necessary packages:  

```bash
pip install nltk numpy scikit-learn streamlit
```

### 3️⃣ Run the Chatbot  
Launch the chatbot using Streamlit:  

```bash
streamlit run chatbot.py
```

