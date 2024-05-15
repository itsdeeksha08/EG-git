import streamlit as st
import re
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Load your dataset
df = pd.read_csv('cobol_java_pairs_preprocessed.csv')

# Load the pre-trained model
vectorizer = TfidfVectorizer()
random_forest_classifier = RandomForestClassifier()
X_cobol_dense = vectorizer.fit_transform(df['COBOL Code']).toarray()
random_forest_classifier.fit(X_cobol_dense, df['Preprocessed Java Translation'])

# Load your .h5 model
model = tf.keras.models.load_model('EGDK/cobol_java_translator.h5')

def generate_java_code_from_cobol(cobol_code):
    cobol_tokens = tokenize_code(cobol_code)
    cobol_normalized_tokens = normalize_code(cobol_tokens)
    cobol_token_string = ' '.join(cobol_normalized_tokens)

    X_cobol = vectorizer.transform([cobol_token_string])
    X_cobol_dense = X_cobol.toarray()

    java_code_tokens = random_forest_classifier.predict(X_cobol_dense)

    # Use your loaded model here
    # For example, replace this line with your model's prediction logic
    java_code_tokens = random_forest_classifier.predict(X_cobol_dense.reshape(1, -1))
    java_code = convert_tokens_to_syntactic(java_code_tokens)

    return format_java_code(java_code)

def convert_tokens_to_syntactic(java_code_tokens):
    syntactic_java_code = ""
    for token in java_code_tokens:
        if token.lower() == "identification":
            continue  # Skip COBOL identification division
        elif token.lower() == "division" or token == ".":
            syntactic_java_code += ";\n"  # End of statement in Java
        elif token.lower() == "program-id":
            syntactic_java_code += "public class "  # Start of Java class
        elif token.lower() == "procedure":
            syntactic_java_code += "public static void main(String[] args) {\n"  # Start of main method
        elif token.lower() == "display":
            syntactic_java_code += "System.out.println"  # Equivalent Java print statement
        elif token == '"':
            syntactic_java_code += "("  # Start of string in Java print statement
        elif token == '.':
            syntactic_java_code += ");"  # End of string in Java print statement
        elif token.isdigit():
            syntactic_java_code += token  # Numbers remain unchanged
        else:
            syntactic_java_code += token.lower()  # Convert to lowercase
    syntactic_java_code += "}\n"  # End of class
    return syntactic_java_code

def format_java_code(java_code):
    # Add indentation
    formatted_java_code = ""
    indentation_level = 0
    for line in java_code.split('\n'):
        if "{" in line:
            formatted_java_code += "\t" * indentation_level + line + "\n"
            indentation_level += 1
        elif "}" in line:
            indentation_level -= 1
            formatted_java_code += "\t" * indentation_level + line + "\n"
        else:
            formatted_java_code += "\t" * indentation_level + line + "\n"
    return formatted_java_code

def tokenize_code(code):
    # Tokenize the code based on whitespace and special characters
    tokens = re.findall(r'\w+|[^\w\s]', code)
    return tokens

def normalize_code(tokens):
    # Normalize the tokens by converting them to lowercase
    normalized_tokens = [token.lower() for token in tokens]
    return normalized_tokens

st.title("COBOL to Java Code Converter")

cobol_code = st.text_area("Enter your COBOL code here:")

if st.button("Convert"):
    java_code = generate_java_code_from_cobol(cobol_code)
    st.code(java_code, language='java')
