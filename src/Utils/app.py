import os
import sys
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from query import query_documents

# --- INITIALIZATION ---
# Load environment variables from a .env file
load_dotenv()

# Create the Flask application instance
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) to allow frontend access
CORS(app)

# --- GEMINI API CONFIGURATION ---
try:
    # Configure the Gemini API with the key from environment variables
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables. Please create a .env file and add it.")
    genai.configure(api_key=gemini_api_key)
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    sys.exit(1)
    # You might want to exit or handle this more gracefully
    # For this example, we'll let it proceed and fail on the API call

VALID_ROLES = {"user", "model", "system"}

def normalize_history(history):
    normalized = []
    for item in history:
        role = item.get("role", "").lower()

        # Fix OpenAI-style "assistant" → Gemini's "model"
        if role == "assistant":
            role = "model"

        # Skip anything invalid
        if role not in VALID_ROLES:
            continue

        normalized.append({
            "role": role,
            "parts": item.get("parts") or item.get("content", "")
        })
    return normalized

# --- API ENDPOINT ---
@app.route('/chat', methods=['POST'])
def chat_handler():
    """
    Handles chat requests to the Gemini API.
    Expects a JSON payload with 'user_query' and 'history'.
    """
    # 1. Validate incoming data
    if not request.json:
        return jsonify({"error": "Invalid request: No JSON payload received."}), 400

    user_query = request.json.get('user_query')
    history = request.json.get('history', []) 

    if not user_query:
        return jsonify({"error": "Missing 'user_query' in the request."}), 400
    
    context_doc = query_documents(user_query, k=10)
    
    # 2. Construct the prompt with a strong system instruction
    system_prompt_parts = """
        You are a specialized health assistant AI. Your primary role is to answer health-related questions based mainly on the provided context.
        You must respond in the same language as the user's query. For example, if the query is in English, reply in English; if in Odia, reply in Odia; and if in a Devanagari script, reply in that same script.
        Your response must be plain text, without any formatting like bold, italics, or lists, for SMS and WhatsApp compatibility.
        If the user says 'talk to a person', 'human', or expresses severe distress, your only response must be: 'If you are experiencing a medical emergency, please call your local emergency number. To speak with a healthcare professional, please call XXX-XXX-XXXX.'
        When a user mentions a symptom (like 'headache' or 'cough'), always ask for other accompanying symptoms at least once to better understand their condition and find the most relevant information. For example, if they say 'I have a headache', respond with something like 'I can help with that. Are you experiencing any other symptoms like fever, nausea, or sensitivity to light?'
        After answering the user's question, suggest a logical next question they might have. For example, if they ask about symptoms, you could ask if they want to know about treatments.
        Refer to yourself only as a 'health assistant AI'. Be transparent that your information can be wrong and is not a substitute for professional medical advice.
        Do not reveal any other details about your origin, such as being developed by Google.
        If the user asks a question unrelated to health (e.g., about politics, sports, etc.), you must politely decline by saying 'I can only answer questions related to health.'
        However, if the user offers a simple greeting like 'hi' or 'hello', you should respond with a brief, friendly greeting in return.
        provide the most accurate and concise answer possible.
    """.strip()

    # Normalize the history to ensure correct roles and structure
    history = normalize_history(history)
    full_query = f"""
    Here is context data you must rely on:
    --- CONTEXT START ---
    {context_doc}
    --- CONTEXT END ---

    User question: {user_query}
    """

    # 4. Call the Gemini API
    try:
        # Initialize the specific Gemini model we want to use
        model = genai.GenerativeModel('gemini-2.0-flash',system_instruction=system_prompt_parts)

        # Start a chat session, loading it with the previous history
        chat_session = model.start_chat(history=history)
        
        # Generate content based on the constructed prompt
        response = chat_session.send_message(full_query)

        # 3. Return the response to the client
        return jsonify({
            "response": response.text,
        })

    except Exception as e:
        # Handle potential errors during the API call
        print(f"An error occurred: {e}")
        return jsonify({"error": "Failed to get a response from the Gemini API."}), 500

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # Runs the Flask development server
    # Use Gunicorn for production deployment
    app.run(host='0.0.0.0', port=5000, debug=True)
