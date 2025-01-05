import streamlit as st
import pandas as pd
import numpy as np
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from transformers import AutoTokenizer

import nltk

# Download necessary NLTK resources
nltk.download('vader_lexicon')
nltk.download('sentiwordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Initialize VADER and BERT sentiment analyzer
sia = SentimentIntensityAnalyzer()
sentiment_analyzer = pipeline("sentiment-analysis")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Constants for Sentiment Labels
POSITIVE = 'positive'
NEGATIVE = 'negative'
NEUTRAL = 'neutral'

# Function to get WordNet POS tag for SentiWordNet
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    return None

# SentiWordNet-based Sentiment Scoring
def get_sentiwordnet_score(word, pos_tag):
    synsets = list(swn.senti_synsets(word, pos_tag))
    if not synsets:
        return 0  # Neutral if no matching synset
    pos_score = sum([syn.pos_score() for syn in synsets]) / len(synsets)
    neg_score = sum([syn.neg_score() for syn in synsets]) / len(synsets)
    return pos_score - neg_score  # Positive - Negative score

# Function to identify the aspect terms
def get_aspect_terms(review, aspect_keywords):
    if not review:
        return []

    tokens = word_tokenize(review.lower())  # Tokenize review
    aspect_terms = []

    # Clean the tokens and check for keyword matches
    for aspect, keywords in aspect_keywords.items():
        for keyword in keywords:
            keyword = keyword.strip().lower()
            for token in tokens:
                token_cleaned = token.strip(".,!?")  # Clean punctuation around the token
                if keyword in token_cleaned:  # Match against cleaned tokens
                    aspect_terms.append(aspect)

    return aspect_terms

# Combined Aspect Review Sentiment Analysis
def combined_aspect_review_sentiment(review, rating, aspect_keywords):
    if not review:
        return NEUTRAL

    max_length = 1012
    tokens = tokenizer.tokenize(review, truncation=True, max_length=512)
    truncated_review = tokenizer.convert_tokens_to_string(tokens)

    aspect_scores = []
    for aspect, keywords in aspect_keywords.items():
        tokens = word_tokenize(truncated_review)
        tagged_tokens = pos_tag(tokens)

        for keyword in keywords:
            if keyword.lower() in truncated_review.lower():
                for word, tag in tagged_tokens:
                    if keyword.lower() in word.lower():
                        wn_tag = get_wordnet_pos(tag)
                        if wn_tag:
                            score = get_sentiwordnet_score(word, wn_tag)
                            aspect_scores.append(score)

    avg_sentiwordnet_score = np.mean(aspect_scores) if aspect_scores else 0
    vader_score = sia.polarity_scores(truncated_review)['compound']
    bert_result = sentiment_analyzer(truncated_review[:max_length])[0]
    bert_score = 1 if bert_result['label'] == 'POSITIVE' else -1

    combined_score = (avg_sentiwordnet_score * 0.4) + (vader_score * 0.35) + (bert_score * 0.25)

    if combined_score > 0.25 or (vader_score > 0.5 and bert_score == 1):
        return POSITIVE
    elif combined_score < -0.25 or (vader_score < -0.5 and bert_score == -1):
        return NEGATIVE
    else:
        return NEUTRAL

# Streamlit Web App
st.title("Aspect and Sentiment Analysis Web App")

review_text = st.text_area("Enter Review Text:", "")
rating = st.selectbox("Select Rating (Optional):", ["", 1, 2, 3, 4, 5], index=0)
st.write("### Define Aspect Keywords")
# Define Aspect Keywords
aspect_keywords = {
    "price": st.text_input("Price Keywords (comma-separated):", "cost, price, inexpensive, investment, money, penny, pay, cheap, spent, pricy, priced, expensive, cheaper, costs, cheapest, free, paid, dollar, overpriced, bucks, over priced, pricing, budget, tax, money, 0, expense, costly, fee, affordable, expensive, cheap, budget-friendly, budget friendly, cost-effective, cost effective, overpriced, reasonable, inexpensive, value for money, premium-priced, premium priced, worth the price, high-priced, high priced, economical, competitive pricing, fair price, low-cost, low cost, steep price, bargain, mid-range pricing, exorbitant"),
    "quality": st.text_input("Quality Keywords (comma-separated):", "high quality, poor quality, well-made, durable, cheap material, quality, broken, tore, lasts, inferior, solid, brass, scraped, smells, delicate, plastic, stiff, tolerate, textured, chinsy, blunt, sharp edges, sharp, waterproof, soft, smell, smooths, broke, poor, textureline, fabric, scratched, metal, smooth, damage, poorly, flimsy, weak, blur, stainless steel, rubbery, rubber, material, sturdy, repair, defective, wrinkles, smelled, described, last longer, mark, uneffected, undamaged, strong, durable, thick, poorest, damaged, break, thicker, reliable, low-grade, tarnish, breaks, lasted, leather, conductive, steady, latex, sleek, weaker, melted, steel, came off, quallity, cotton, overheating, lather, rusted, durability, poor quality, materials, rubberized, lasting, flimsy, sturdy, weak, cheap, solid, fragile, heavy-duty, high-quality, substandard, premium, inferior, top-notch, poor-quality, reliable, cheap materials, luxurious, wear-resistant, wear resistent, reliable, faulty, efficient, unreliable, smooth operation, problematic, consistent, malfunctioning, high-performance, high performance, high-quality, high quality, well-finished, well finished, low-quality performance, low-quality, low quality, rough, sleek, poor craftsmanship, polished, scratched, elegant, shoddy, clean, breaks easily, wears out quickly, enduring, short lifespan, resilient, prone to damage, retains quality, fades quickly, maintains durability, works perfectly, defective, smooth functioning, prone to malfunction, glitchy, operational issues, performs as expected, faulty mechanism, seamless performance, unreliable performance"),
    "service": st.text_input("Service Keywords (comma-separated):", "customer service, support, help, assistance, manual, instructions, contact, seller, shipping, return, arrived, cardboard box, box, packaging, packaged, date, contacted, response, refund, apologized, trust, duplicate, delivered, advertisement, instructional, description, policy, unprofessional, advertised, pollicies, replied, fake, company, missing, production, consumers, waiting, warranty, email, advertising, shipped, misleading, packed, customer, service, replacing, returns, sent back, advertized, exchange, package, ship, advertises, comply, contacting, respond, delivery, dellivery, warn, described, details, miswire, manufacturer, tech staff, mentioned, lack, faulty, arrive, repairable, emails, calls, companies, reply, inquiries, customer service, receipt, customers, manufacture, manufacturers, apology, receive, mention, condition, unboxed, misrepresented, timely, misunderstood, specify, packing, reliable, sealed, refunded, serviced, refurbished, reimbursement, reported, emailed, shipment, explained, specs, miss-leading, miss leading, responsive, unhelpful, friendly, rude, knowledgeable, incompetent, polite, prompt, slow, efficient, unresponsive, fast, delayed, on-time, late, efficient, well-packaged, damaged in transit, safe, smooth, poor handling, excellent follow-up, excellent followup, delayed response, poor after-care, supportive, neglectful, warranty fulfillment, difficult return process, great replacement service, transparent, lack of updates, frequent follow-ups, no contact"),
    "usability": st.text_input("Usability Keywords (comma-separated):", "difficult, complicated, intuitive, useful, functions, wear, pulled, workout, roll down, rolled up down, flexibility, using, useless, work, making, function, holding, playing, comfy, fun, uncomfortable, taste, tastes, performance, use, moves, works, learning tool, stacking, knocking, playtime, spin, twirl, educational, worked, rotate, used, played, boring, portable, comfortable, worn, play, assemble, drag, absorbs, pull, wearing, pushed, flexible, roll up, wore, squeezed, hold, feel, holds, bruising, pushing, working, push, put it on, hurts, lifting, put on, does the job, felt, burned, eating, attention, user friendly, feeding, warming, disassemble, functional, effective, assembly, easy to put, assembled, operated, scrubbed, roll over, flickering, stopped, stayed, usage, plugged, stops working, job, functionality, install, uses, turning, feels, programmed, riding, unstable, design, installation, installed, assembling, installing, designed, usable, user-friendly, intuitive, user friendly, easy to use, straightforward, simple interface, accessible, convenient, seamless experience, effortless, responsive, smooth navigation, learning curve, customizable, ergonomic, clear instructions, interactive, efficient, time-saving, functional, cluttered"),
    "size": st.text_input("Size Keywords (comma-separated):", "size, fits, heavy, sizes, chart, smaller, compact design, snug, oversized, mini, clunky, too small, too large, perfectly, slim, large, feet, big, fit, longer, small, tiny, width, thin, taller, tight, small inch, skinny, hefty, long, xl, length, ft, inches, measurement, streched, medium, xlarge, sized, smaller size, gigantic, pound, tall, tightness, bulky, sizing, measure, shorter, short, tighter, inch, size chart, xs, high, measured, stouter, wider, x-large, mediums, bigger, foot, lower, height, lowering, fitted, higher, lowered, cumbersome, larger")
}

aspect_keywords = {k: v.split(",") for k, v in aspect_keywords.items()}

if st.button("Analyze Review"):
    aspect_terms = get_aspect_terms(review_text, aspect_keywords)
    unique_aspect_terms = list(set(aspect_terms))  # Remove duplicates
    sentiment_label = combined_aspect_review_sentiment(review_text, rating, aspect_keywords)
    st.write(f"### Aspect Categories Identified: {', '.join(unique_aspect_terms)}")
    st.write(f"### Sentiment Analysis Result: {sentiment_label.capitalize()}")
