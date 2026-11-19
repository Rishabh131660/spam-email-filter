import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Spam Email Detector", page_icon="📧")
st.title("📧 Spam Email Filtration Web App")
st.write("Paste any email message below to check whether it is *Spam* or *Safe (Ham)*.")


@st.cache_resource
def get_model():
    data = [
        ("spam", "Congratulations! You have won a $1000 Walmart gift card. Click here to claim now!"),
        ("ham", "Hi team, please find attached the meeting notes from today's project sync."),
        ("spam", "URGENT: Your bank account has been suspended. Verify your password immediately."),
        ("ham", "Are we still meeting for lunch at 1 PM tomorrow near the office?"),
        ("spam", "FREE entry into our weekly cash draw! Text WIN to 80085 to claim your prize."),
        ("ham", "Can you review the Q3 budget spreadsheet before Friday afternoon?"),
        ("spam", "Get cheap meds online without prescription! Limited time 80% discount offer."),
        ("ham", "Thanks for sending over the invoice. I have forwarded it to accounting."),
        ("spam", "You have been selected for a free iPhone 15! Reply YES to confirm shipping."),
        ("ham", "Let me know when you arrive at the station so I can pick you up."),
        ("spam", "Act now! Exclusive crypto investment doubles your money in 24 hours."),
        ("ham", "Reminder: The quarterly town hall starts in 15 minutes in Conference Room B."),
        ("spam", "You have an unclaimed tax refund waiting. Click the link to submit details."),
        ("ham", "Could you share the updated slides from yesterday's client presentation?"),
    ]
    X = [item[1] for item in data]
    y = [item[0] for item in data]
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
        ("clf", MultinomialNB()),
    ])
    model.fit(X, y)
    return model


model = get_model()

email_input = st.text_area("Enter or paste email text here:", height=150)

if st.button("Check Email"):
    if not email_input.strip():
        st.warning("Please enter some email text first.")
    else:
        pred = model.predict([email_input])[0]
        probs = model.predict_proba([email_input])[0]
        classes = list(model.classes_)
        spam_prob = probs[classes.index("spam")] * 100
        ham_prob = probs[classes.index("ham")] * 100

        if pred == "spam":
            st.error(f"🚨 *SPAM EMAIL* (Spam probability: *{spam_prob:.1f}%*)")
        else:
            st.success(f"✅ *SAFE / HAM EMAIL* (Safe probability: *{ham_prob:.1f}%*)")
