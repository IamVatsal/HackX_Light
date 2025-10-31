from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from Utils.Gemini_Api import call_gemini_api

# 1. Initialize the same embedding model we used for ingestion
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Connect to the existing Qdrant collection
client = QdrantClient(url="http://localhost:6333")
qdrant = QdrantVectorStore(
    client=client, 
    collection_name="aarogya_sahayak", 
    embedding=embeddings
)

def query_documents(query_text: str, k: int = 3):
    """
    Query the Qdrant vector database for the most relevant documents.

    Args:
        query_text (str): The user's query.
        k (int): Number of top relevant documents to retrieve.

    Returns:
        list: A list of the most relevant documents.
    """
    rewritten_query = call_gemini_api(f"""
    You are an expert Clinical Query Rewriter for a medical Retrieval-Augmented Generation (RAG) system.
    Your primary function is to transform a user's question, which may be simple or colloquial, into a highly specific, clinically detailed, and technical query. The goal is to maximize the retrieval of relevant medical literature, clinical trial data, or detailed pathological descriptions from the vector database.
    Instructions:
        1. Analyze the user's original question: {query_text}
        2. The hypothetical answer must utilize precise medical terminology (e.g., specific drug names, anatomical structures, diagnostic criteria, ICD-10 codes if relevant, etc.) and establish the clinical context.
        3. The output must be only the generated hypothetical answer, with no preamble, explanations, or conversational text.
    Original Question: {query_text}
    Example Transformations:
        - Original Question: "what are the symptoms of influenza?"
          Rewritten Query (Hypothetical Answer): "The clinical presentation of seasonal influenza (flu), caused by Orthomyxoviridae viruses, typically involves the acute onset of systemic symptoms such as high-grade fever, myalgia, severe fatigue, and respiratory manifestations including non-productive cough and pharyngitis, often requiring differentiation from common cold and COVID-19 in the differential diagnosis."
        - Original Question: "How can I tell if my kid has a fever or something worse?"
          Rewritten Query (Hypothetical Answer): "To differentiate between a simple fever and a more serious condition in children, clinicians often use a combination of history-taking (e.g., duration of fever, associated symptoms) and physical examination findings (e.g., signs of respiratory distress, dehydration). Laboratory tests such as complete blood count (CBC) or inflammatory markers (e.g., CRP) may also be employed to assess the severity of the illness."
    """)
    found_docs = qdrant.similarity_search(rewritten_query, k=k)
    output = ""
    for i, doc in enumerate(found_docs):
        output += f"\n--- Document {i+1} ---"
        output += f"\nSource: {doc.metadata.get('source', 'Unknown')}"
        output += f"\nContent: {doc.page_content}"
    return output

if __name__ == '__main__':
    # 3. Define our query
    query = "what are the symptoms of influenza?"
    print(f"Searching for: '{query}'")

    # 4. Perform the similarity search
    # This retrieves the 3 most relevant chunks from the database.
    found_docs = qdrant.similarity_search(query, k=10)

    # 5. Print the results
    print(f"\n--- Found {len(found_docs)} relevant documents ---")
    output = ""
    for i, doc in enumerate(found_docs):
        output += f"\n--- Document {i+1} ---"
        output += f"\nSource: {doc.metadata.get('source', 'Unknown')}"
        output += f"\nContent: {doc.page_content}"
    print(output)