import nltk
import random
import streamlit as st
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download NLTK data
nltk.download('punkt')

# Define chatbot intents
intents = [ 
    {
        'tag': 'greeting',
        'patterns': ['Hi', 'Hello', 'Hey', 'Whats up', 'How are you'],
        'responses': ['Hi there', 'Hello', 'Hey', 'Nothing much', 'I’m fine, thank you']
    },
    {
        'tag': 'goodbye',
        'patterns': ['Bye', 'See you later', 'Goodbye', 'Take care'],
        'responses': ['Goodbye', 'See you later', 'Take care']
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you", "What's your age"],
        "responses": ["I don't have an age. I'm a chatbot.", "I was just born in the digital world.", "Age is just a number for me."]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like", "How's the weather today"],
        "responses": ["I'm sorry, I cannot provide real-time weather information.", "You can check the weather on a weather app or website."]
    },
    {
        "tag": "budget",
        "patterns": ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"],
        "responses": [
            "To make a budget, start by tracking your income and expenses. Then, allocate your income towards essential expenses like rent, food, and bills. "
            "Next, allocate some of your income towards savings and debt repayment. Finally, allocate the remainder of your income towards discretionary expenses like entertainment and hobbies.",
            "A good budgeting strategy is to use the 50/30/20 rule. This means allocating 50% of your income towards essential expenses, 30% towards discretionary expenses, and 20% towards savings and debt repayment.",
            "To create a budget, start by setting financial goals for yourself. Then, track your income and expenses for a few months to get a sense of where your money is going. "
            "Next, create a budget by allocating your income towards essential expenses, savings and debt repayment, and discretionary expenses."
        ]
    },
    {
        "tag": "credit_score",
        "patterns": ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"],
        "responses": [
            "A credit score is a number that represents your creditworthiness. It is based on your credit history and is used by lenders to determine whether or not to lend you money. The higher your credit score, the more likely you are to be approved for credit.",
            "You can check your credit score for free on several websites such as Credit Karma and Credit Sesame."
        ]
    },
    {
        "tag": "investing",
        "patterns": ["How do I start investing", "What are good investments", "How to invest money", "Investment advice"],
        "responses": [
            "To start investing, first build an emergency fund, then consider retirement accounts like 401(k) or IRA, and finally explore other investment options like index funds or ETFs.",
            "Good investments for beginners include index funds, ETFs, and mutual funds as they provide diversification and typically have lower fees.",
            "When investing, consider your goals, risk tolerance, and time horizon. Start with low-cost index funds for long-term growth."
        ]
    },
    {
        "tag": "side_hustle",
        "patterns": ["How can I make extra money?", "Side hustle ideas", "Ways to earn extra income", "Best side hustles"],
        "responses": [
            "Popular side hustles include freelancing, selling products online, tutoring, and investing in stocks or real estate.",
            "Consider gig economy jobs like Uber, DoorDash, or freelance work on platforms like Upwork and Fiverr.",
            "If you have a skill, turn it into a side hustle! Graphic design, writing, coding, and consulting are in demand."
        ]
    },
    {
        "tag": "cryptocurrency",
        "patterns": ["What is cryptocurrency?", "How to invest in crypto?", "Is Bitcoin a good investment?", "Best cryptocurrencies to buy"],
        "responses": [
            "Cryptocurrency is a digital or virtual currency that uses cryptography for security and operates on decentralized blockchain technology.",
            "Investing in crypto can be risky but also rewarding. Always do thorough research and consider stable coins like Bitcoin and Ethereum.",
            "You can buy cryptocurrency through exchanges like Coinbase, Binance, or Kraken. Always store your assets in a secure wallet."
        ]
    },
    {
        "tag": "freelancing",
        "patterns": ["How to start freelancing?", "Best freelance websites", "Freelance jobs for beginners", "Freelancing tips"],
        "responses": [
            "To start freelancing, choose a skill, build a portfolio, and find clients on platforms like Upwork, Fiverr, and Freelancer.",
            "Good freelance jobs for beginners include content writing, graphic design, virtual assistance, and web development.",
            "Be sure to set clear rates, deliver quality work on time, and maintain good communication with clients."
        ]
    },
    {
        "tag": "entrepreneurship",
        "patterns": ["How to start a business?", "Business ideas", "Steps to become an entrepreneur", "Startup advice"],
        "responses": [
            "To start a business, begin with market research, create a business plan, and secure funding or investment.",
            "Some profitable business ideas include dropshipping, content creation, e-commerce, and consulting.",
            "Entrepreneurship requires persistence, adaptability, and financial planning. Start small and scale as you grow."
        ]
    },
    {
        "tag": "inflation",
        "patterns": ["What is inflation?", "How does inflation affect me?", "How to protect money from inflation?", "Inflation rate"],
        "responses": [
            "Inflation is the rate at which the general level of prices for goods and services rises, reducing purchasing power.",
            "You can protect your money from inflation by investing in assets like real estate, stocks, or inflation-protected securities.",
            "Governments and central banks try to control inflation through monetary policies like adjusting interest rates."
        ]
    },
    {
        "tag": "passive_income",
        "patterns": ["How to make passive income?", "Best passive income ideas", "Ways to earn money while sleeping"],
        "responses": [
            "Passive income ideas include dividend stocks, real estate rentals, affiliate marketing, and creating online courses.",
            "Investing in REITs (Real Estate Investment Trusts) is a great way to earn passive income without owning property.",
            "Selling digital products like ebooks, templates, and stock photos can provide a steady stream of passive income."
        ]
    },
    {
        "tag": "negotiation",
        "patterns": ["How to negotiate a salary?", "Salary negotiation tips", "How to get a raise?", "Best negotiation strategies"],
        "responses": [
            "When negotiating a salary, research industry rates, highlight your achievements, and confidently ask for what you're worth.",
            "Timing is key! Ask for a raise after a successful project or performance review.",
            "If your employer can't offer a higher salary, negotiate for perks like remote work, bonuses, or additional vacation days."
        ]
    },
    {
        "tag": "resume_tips",
        "patterns": ["How to write a resume?", "Resume tips", "Best resume format"],
        "responses": [
            "A strong resume should be concise, highlight achievements, and use action words. Keep it to 1-2 pages.",
            "Use a professional template and tailor your resume for each job application."
        ]
    },
    {
        "tag": "unknown",
        "patterns": [],
        "responses": ["I'm not sure about that. Can you rephrase?", "Sorry, I don't have an answer for that."]
    }
]
