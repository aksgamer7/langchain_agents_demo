import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
#from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM


# Load environment variables
load_dotenv()
pdf_path = os.getenv("PDF_PATH")
model_name = os.getenv("MODEL")

# Step 1: Load PDF
loader = PyPDFLoader(r"C:\Users\Admin\Projects\Langchain_agents_demo\pdfs\meteor350usermanual.pdf")
documents = loader.load()

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(documents)

# Step 3: Load local LLM (Ollama)
#llm = Ollama(model=model_name)
llm = OllamaLLM(model=model_name)


# Step 4: Define Prompt Template
prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in 5 bullet points:\n\n{text}"
)

# Step 5: LCEL pipeline (prompt | llm)
chain = prompt | llm

# Step 6: Run summarization
for doc in docs[:2]:  # demo: summarize first 2 chunks
    summary = chain.invoke({"text": doc.page_content})
    print("\n--- Summary ---\n", summary)