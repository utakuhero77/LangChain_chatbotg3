# Install required libraries:
# pip install langchain sentence-transformers faiss-cpu langchain-huggingface

from langchain.chains import LLMChain  # Updated import for LLMChain
from langchain_core.prompts import PromptTemplate  # Updated import for PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import for HuggingFaceEmbeddings
from sentence_transformers import util

# Load a pre-trained Sentence Transformer model
model_name = 'all-MiniLM-L6-v2'
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Define some sentences
sentences = ["Hello, world!", "This is a test sentence."]

# Generate embeddings for the sentences
sentence_embeddings = embeddings.embed_documents(sentences)

# Print the embeddings
print("Sentence embeddings:")
for sentence, embedding in zip(sentences, sentence_embeddings):
    print(f"Sentence: {sentence}")
    print(f"Embedding shape: {len(embedding)}")
    print(f"Embedding: {embedding}")
    print()

# Compute cosine similarity between two sentences
similarity = util.cos_sim(sentence_embeddings[0], sentence_embeddings[1])
print(f"Cosine similarity between '{sentences[0]}' and '{sentences[1]}': {similarity.item()}")

# Create a simple LangChain "Hello World" script
prompt_template = PromptTemplate(
    input_variables=["input_text"],
    template="You said: {input_text}"
)

# Initialize the LLMChain with a simple prompt
chain = LLMChain(llm=embeddings, prompt=prompt_template)

# Run the chain with some input text
input_text = "Hello, LangChain!"
output = chain.run(input_text)

print(output)