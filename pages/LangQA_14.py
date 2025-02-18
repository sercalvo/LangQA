# -*- coding: utf-8 -*-
"""
Created on Wed September 11 13:49:10 2024

@author: Sergio
"""

import streamlit as st
import pandas as pd
import spacy
from spellchecker import SpellChecker
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import language_tool_python

# Initialize language tools
tool_es = language_tool_python.LanguageTool('es')
tool_en = language_tool_python.LanguageTool('en')

# Load Spanish and English models
@st.cache_resource
def load_models():
    nlp_es = spacy.load("es_core_news_md")
    nlp_en = spacy.load("en_core_web_md")
    return nlp_es, nlp_en

nlp_es, nlp_en = load_models()

# Load spell checkers for Spanish and English
@st.cache_resource
def load_spell_checkers():
    spell_es = SpellChecker(language='es')
    spell_en = SpellChecker(language='en')
    return spell_es, spell_en

spell_es, spell_en = load_spell_checkers()

# Initialize the stemmer
stemmer = PorterStemmer()

# Function to stem text
def stem_text(text):
    words = word_tokenize(text)
    return " ".join([stemmer.stem(word) for word in words])

# Function to detect foreign words or typos
def check_foreign_words(token, custom_dict):
    if token.text.lower() in custom_dict.keys():
        return f"**Foreign word or typo**", f"'{token.text}' should be '{custom_dict[token.text.lower()]}'"
    return None, None

# Function to check spelling
def check_spelling(token, spell_checker):
    misspelled = spell_checker.unknown([token.text])
    if token.text in misspelled:
        candidates = spell_checker.candidates(token.text)
        suggestion = next(iter(candidates), token.text) if candidates else token.text
        return f"**Spelling**", f"Potential spelling error: '{token.text}' (suggested: '{suggestion}')"
    return None, None

# Function to check punctuation and capitalization
def check_punctuation_and_capitalization(sent, language):
    issues = []

    # Check if sentence ends with proper punctuation
    if sent[-1].text not in [".", "!", "?", "¡", "¿"]:
        issues.append(("**Punctuation**", "Sentence missing proper end punctuation"))

    # Detect missing starting interrogation mark for Spanish
    if language == "es" and sent[0].text not in ["¡", "¿"]:
        issues.append(("**Punctuation**", "Sentence missing starting interrogation mark"))
    
    # Detect capitalization issues after missing punctuation
    words = [token.text for token in sent]
    for i in range(1, len(words)):
        if words[i].isalpha() and words[i][0].isupper() and words[i-1] not in [".", "!", "?", "¡", "¿"]:
            issues.append(("**Capitalization issue**", f"'{words[i]}' is capitalized after '{words[i-1]}' without proper punctuation."))

    return issues

# Function to check repeated words within a sentence
def check_repeated_words(sent):
    issues = []
    previous_word = ""
    for token in sent:
        if token.text.lower() == previous_word.lower():
            issues.append(("**Repeated word**", f"'{token.text}'"))
        previous_word = token.text
    return issues

# Main proofreading function
def proofreading(text, language, custom_dict, spell_checker):
    doc = nlp_es(text) if language == "es" else nlp_en(text)
    errors = []

    for sent in doc.sents:
        sent_errors = []

        # Check each token for foreign words, spelling, and grammar
        for token in sent:
            foreign_word_issue, foreign_word_suggestion = check_foreign_words(token, custom_dict)
            spelling_issue, spelling_suggestion = check_spelling(token, spell_checker)

            if foreign_word_issue:
                sent_errors.append((foreign_word_issue, foreign_word_suggestion))
            if spelling_issue:
                sent_errors.append((spelling_issue, spelling_suggestion))

        # Add punctuation and capitalization checks
        punctuation_issues = check_punctuation_and_capitalization(sent, language)
        if punctuation_issues:
            sent_errors.extend(punctuation_issues)

        # Add repeated words check
        repeated_word_issues = check_repeated_words(sent)
        if repeated_word_issues:
            sent_errors.extend(repeated_word_issues)

        # Add any found issues to the errors list
        if sent_errors:
            errors.append({"sentence": sent.text, "issues": sent_errors})

    return errors

# Function for terminology consistency check
def check_terminology(df, source_col, target_col, source_terms, target_terms):
    terminology_issues = []
    stemmed_source_terms = [stem_text(term) for term in source_terms]
    stemmed_target_terms = [stem_text(term) for term in target_terms]

    for index, row in df.iterrows():
        source = row[source_col] if pd.notna(row[source_col]) else ""
        target = row[target_col] if pd.notna(row[target_col]) else ""
        
        stemmed_source = stem_text(source)
        
        for term in stemmed_source_terms:
            if term in stemmed_source:
                stemmed_target = stem_text(target)
                if not any(target_term in stemmed_target for target_term in stemmed_target_terms):
                    terminology_issues.append((index, term, target))
    
    return pd.DataFrame(terminology_issues, columns=['Row Index', 'Source Term', 'Target Text'])

# Streamlit app
st.title("Advanced Translation QA Tool with Proofreading")

# Language choice and text area on the same page
language_choice = st.selectbox("Choose the language", ("Spanish", "English"))

if language_choice == "Spanish":
    nlp = nlp_es
    spell = spell_es
    custom_dictionary = custom_dictionary_es
elif language_choice == "English":
    nlp = nlp_en
    spell = spell_en
    custom_dictionary = custom_dictionary_en

# Input method choice
upload_option = st.radio("Choose input method", ("Upload Excel file", "Paste text"))

if upload_option == "Upload Excel file":
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.write("Uploaded DataFrame:", df.head())

        source_col = st.selectbox('Select the Source column', df.columns)
        target_col = st.selectbox('Select the Target column', df.columns)

        source_terms = st.text_input('Enter source terms list (comma separated)', 'cat,play').split(',')
        target_terms = st.text_input('Enter target terms list (comma separated)', 'gato,jugar').split(',')

        if st.button('Run QA'):
            terminology_issues = check_terminology(df, source_col, target_col, source_terms, target_terms)
            st.subheader("Terminology Issues")
            st.dataframe(terminology_issues)

            st.subheader("Target Text Proofreading Issues")
            for index, row in df.iterrows():
                target_text = row[target_col] if pd.notna(row[target_col]) else ""
                
                errors_target = proofreading(target_text, language_choice[:2], custom_dictionary, spell)
                
                if errors_target:
                    for error in errors_target:
                        st.write(f"**Sentence**: {error['sentence']}")
                        st.write("**Issues**:")
                        df_errors = pd.DataFrame(error['issues'], columns=["Error Category", "Suggestion"])
                        st.table(df_errors)
                else:
                    st.write("No issues detected!")

elif upload_option == "Paste text":
    text = st.text_area("Enter text", 
        "Este es un ejemplo de texto para proofread. Puede patecer una tontría pero es lo que que es Y la necedad esta en que chatgpt no sabe arremangar este texto ni analizar si esto tiene sentido." if language_choice == "Spanish" 
        else "This is an example of text for proofread. It can seems silly but it's what it is And the nonsense lies in that chatgpt can't really proofread this text properly.")

    if st.button("Proofread"):
        errors = proofreading(text, language_choice[:2], custom_dictionary, spell)
        if errors:
            for error in errors:
                st.write(f"**Sentence**: {error['sentence']}")
                st.write("**Issues**:")
                df_errors = pd.DataFrame(error['issues'], columns=["Error Category", "Suggestion"])
                st.table(df_errors)
        else:
            st.write("No issues detected!")
