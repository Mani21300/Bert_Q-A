import streamlit as st
from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import pipeline

# Load the pre-trained model and tokenizer
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Set up the question-answering pipeline with additional parameters
question_answering = pipeline(
    "question-answering",
    model=model,
    tokenizer=tokenizer,
    handle_impossible_answer=True,  # Handle cases where no answer can be found
)

# Streamlit app layout
st.title("Question Answering with BERT")

st.write("""
### Enter the passage and your question, and BERT will try to answer it!
""")

# Text input for passage and question
passage = st.text_area("Passage", "Enter a passage here...")
question = st.text_input("Question", "Enter your question...")

if st.button("Get Answer"):
    if passage and question:
        # Perform question answering
        result = question_answering(
            question=question,
            context=passage,
            max_answer_len=50,  # Limit the length of the answer
        )
        
        # Display the result and confidence score
        st.write(f"**Answer**: {result['answer']}")
        st.write(f"**Confidence Score**: {result['score']:.2f}")
    else:
        st.write("Please provide both a passage and a question.")
 