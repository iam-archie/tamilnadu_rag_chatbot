
import os

from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==========================================
# STEP 1: Load Environment Variables
# ==========================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is missing in .env file")


# ==========================================
# STEP 2: Document Loader
# ==========================================

url = "https://www.tn.gov.in/scheme_details.php?id=ODAz"

print("\nLoading Tamil Nadu Government webpage...")

loader = WebBaseLoader(web_paths=(url,))

documents = loader.load()

print(f"Documents loaded: {len(documents)}")


# ==========================================
# STEP 3: Text Splitter
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")


# ==========================================
# STEP 4: Embeddings
# ==========================================

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

print("\nCreating embeddings...")


# ==========================================
# STEP 5: Vector Store
# ==========================================

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("Documents stored in FAISS vector database!")


# ==========================================
# STEP 6: Retriever
# ==========================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# ==========================================
# STEP 7: LLM
# ==========================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# ==========================================
# STEP 8: Prompt Template
# ==========================================

prompt = ChatPromptTemplate.from_template("""
You are a helpful Tamil Nadu Government Scheme Assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not make up information.
2. If the answer is not available in the context,
   say "The information is not available in the source document."
3. Give simple and clear answers.
4. Do not assume missing eligibility conditions.

Context:
{context}

User Question:
{question}

Answer:
""")


# ==========================================
# STEP 9: RAG Chain
# ==========================================

rag_chain = prompt | llm | StrOutputParser()


# ==========================================
# STEP 10: Chatbot
# ==========================================

print("\n====================================")
print(" Tamil Nadu Government Scheme Chatbot")
print("====================================")
print("Ask questions about the government scheme.")
print("Type 'exit' to stop.\n")


while True:

    question = input("You: ").strip()

    if question.lower() in ["exit", "quit", "bye"]:
        print("Chatbot: Thank you! Goodbye.")
        break

    if not question:
        continue

    # Retrieve relevant document chunks
    retrieved_docs = retriever.invoke(question)

    # Prepare context
    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    # Generate answer
    response = rag_chain.invoke({
        "context": context,
        "question": question
    })

    print("\nChatbot:", response)

    # Display sources
    print("\nSources:")

    for doc in retrieved_docs:
        print("-", doc.metadata.get("source", url))

    print("\n" + "=" * 50 + "\n")
