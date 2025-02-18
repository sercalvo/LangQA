# -*- coding: utf-8 -*-
"""
Created on Wed September 11 13:49:10 2024

@author: Sergio
"""

import streamlit as st
from spellchecker import SpellChecker
import spacy
from nltk.stem.snowball import SnowballStemmer
import pandas as pd

# Load spaCy model for NLP
nlp = spacy.load("en_core_web_sm")

# Initialize spell checker
spell = SpellChecker()

# Sample text with errors
sample_text = """
The cat was plaing with the ball. They is very happy becuse it's a sunny day.
Tomorrw, we will go to park, however we didn't bring any food.
"""

# Streamlit UI
st.title("AI-driven Localization QA & Terminology Assistant")

# Text input for user-provided or sample text
text_input = st.text_area("Input Text", sample_text, height=200)

# Error detection functions
def check_spelling(text):
    words = text.split()
    misspelled = spell.unknown(words)
    return misspelled

def extract_terms(text):
    doc = nlp(text)
    terms = [token.text for token in doc if token.pos_ == "NOUN" or token.pos_ == "PROPN"]
    return terms

def check_term_consistency(source_terms, target_terms):
    stemmer = SnowballStemmer("english")
    source_stems = {stemmer.stem(term) for term in source_terms}
    target_stems = {stemmer.stem(term) for term in target_terms}

    inconsistent_terms = source_stems.symmetric_difference(target_stems)
    return inconsistent_terms

# Function to highlight errors in different colors
def highlight_errors(text, misspelled_words, grammar_errors=None, term_inconsistencies=None):
    for word in misspelled_words:
        text = text.replace(word, f'<span style="color:red;">{word}</span>')
    if grammar_errors:
        for word in grammar_errors:
            text = text.replace(word, f'<span style="color:orange;">{word}</span>')
    if term_inconsistencies:
        for word in term_inconsistencies:
            text = text.replace(word, f'<span style="color:yellow;">{word}</span>')
    return text

# Error types (e.g., grammar) - placeholder
grammar_errors = ["is"]  # A basic placeholder to show grammar errors

# Run QA Checks
if text_input:
    # Spell checker
    misspelled_words = check_spelling(text_input)

    # Extract terms and check terminology consistency (using same text as source/target for now)
    source_terms = extract_terms(text_input)
    target_terms = extract_terms(text_input)
    term_inconsistencies = check_term_consistency(source_terms, target_terms)

    # Highlight errors in the text
    highlighted_text = highlight_errors(text_input, misspelled_words, grammar_errors, term_inconsistencies)
    
    # Display the highlighted text
    st.markdown(f"### Text with Highlighted Errors:")
    st.markdown(f'<p>{highlighted_text}</p>', unsafe_allow_html=True)

    # Prepare table for error summary
    error_summary = []
    
    # Add spelling errors to table
    for word in misspelled_words:
        error_summary.append({
            "Error Type": "Spelling",
            "Word": word,
            "Suggestions": ", ".join(spell.candidates(word))
        })
    
    # Add grammar errors to table (simplified)
    for word in grammar_errors:
        error_summary.append({
            "Error Type": "Grammar",
            "Word": word,
            "Suggestions": "Check subject-verb agreement"
        })
    
    # Add terminology inconsistencies to table
    for word in term_inconsistencies:
        error_summary.append({
            "Error Type": "Terminology Consistency",
            "Word": word,
            "Suggestions": "Ensure consistent terminology usage"
        })

    # Convert error summary to a DataFrame and display it as a table
    if error_summary:
        df = pd.DataFrame(error_summary)
        st.markdown("### Error Summary Table")
        st.table(df)
