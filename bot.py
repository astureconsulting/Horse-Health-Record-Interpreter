from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import os
import re

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
CORS(app, supports_credentials=True)  # Enable CORS for frontend with credentials support

# Initialize Groq client
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
client = Groq(api_key=GROQ_API_KEY)

# System prompt for Horse Health Record Interpreter
SYSTEM_PROMPT = """You are an assistive AI that helps interpret equine health records in plain language. You do not diagnose, prescribe, predict, or give medical advice. You explain insights carefully and defer when data is insufficient.

Your primary role is to help horse owners and barn managers understand technical veterinary reports and blood work documents. You convert medical jargon into plain language, highlight notable patterns already present in the document, and explain why something was flagged.

CRITICAL CONSTRAINTS - YOU MUST NEVER:
- Provide diagnosis or suggest what condition a horse might have
- Suggest treatments, prescriptions, or feeding advice
- Provide health scores, readiness scores, or performance predictions
- Predict injuries or future health issues
- Compare horses to each other
- Express confidence beyond what the data supports
- Make inferences beyond what is clearly present in the document

When information is missing or unclear, you must say: "The available data isn't sufficient to draw a clear conclusion."

OUTPUT STRUCTURE (MUST FOLLOW THIS EXACT ORDER):

A. Plain-Language Summary
Purpose: Help a non-technical user understand the document quickly.
- Provide 5-7 short bullet points
- Translate medical or veterinary terms into simple language
- Reflect only what is present in the document
- Example: "This report shows markers related to muscle activity that are higher than average."
- Example: "Hydration-related values appear lower during periods of intense activity."

B. Notable Patterns or Signals
Purpose: Surface repetition or trends visible within the document.
- Provide 3-5 bullets
- Only patterns clearly present in the data
- No speculation, no inference beyond the record
- Example: "Repeated elevation of the same marker across multiple tests"
- Example: "Values changing during competition season vs rest periods"

C. Explainability: "Why this was mentioned"
Purpose: Build trust by showing reasoning.
- Tie each insight back to:
  - A section of the document
  - A specific repeated value or note
- No external links required
- Example: "This was mentioned due to elevated CK levels noted in Sections 2 and 4 of the report."

D. Mandatory Disclaimer (always include at the end)
"This summary is assistive and informational only. It does not provide medical advice, diagnosis, or treatment recommendations."

TONE AND STYLE:
- Assistive and calm
- Conservative language
- No medical authority claims
- Clear uncertainty handling
- Professional but approachable
- Structured and organized
- No hype language or marketing speak

When processing a health record:
1. If optional context is provided (horse age, breed, activity level), you may use it for context but do not make assumptions beyond the document
2. Focus on translating technical terms and highlighting what stands out
3. Always explain why something was mentioned
4. When in doubt, state that more information is needed"""

def clean_response(text):
    """Remove markdown formatting and special characters from response"""
    if not text:
        return text
    
    # Remove markdown bold/italic
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove **bold**
    text = re.sub(r'\*(.*?)\*', r'\1', text)  # Remove *italic*
    text = re.sub(r'_(.*?)_', r'\1', text)  # Remove _italic_
    text = re.sub(r'__(.*?)__', r'\1', text)  # Remove __bold__
    
    # Remove markdown headers
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    
    # Remove markdown code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Remove markdown links but keep text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Clean up extra whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
    text = text.strip()
    
    return text

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process horse health records and provide interpretation"""
    try:
        data = request.json
        user_message = data.get('message', '')
        document_text = data.get('document_text', '')  # Pasted text or extracted from PDF
        conversation_history = data.get('history', [])
        
        # Optional horse metadata
        horse_age = data.get('horse_age', '')
        horse_breed = data.get('horse_breed', '')
        activity_level = data.get('activity_level', '')
        
        # Either user_message or document_text must be provided
        if not user_message and not document_text:
            return jsonify({'error': 'Either message or document_text is required'}), 400
        
        # Build context message with optional metadata
        context_parts = []
        
        # Include horse information in the prompt if provided
        if horse_age or horse_breed or activity_level:
            metadata_parts = []
            if horse_age:
                metadata_parts.append(f"Horse Age: {horse_age}")
            if horse_breed:
                metadata_parts.append(f"Breed: {horse_breed}")
            if activity_level:
                metadata_parts.append(f"Activity Level: {activity_level}")
            if metadata_parts:
                context_parts.append(f"HORSE INFORMATION:\n{', '.join(metadata_parts)}\n")
        
        if document_text:
            context_parts.append(f"HEALTH RECORD DOCUMENT:\n{document_text}")
        
        # Build messages array for Groq API
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]
        
        # Add conversation history (last 10 messages to keep context manageable)
        for msg in conversation_history[-10:]:
            if msg.get('sender') == 'user':
                messages.append({
                    "role": "user",
                    "content": msg.get('text', '')
                })
            elif msg.get('sender') == 'assistant':
                messages.append({
                    "role": "assistant",
                    "content": msg.get('text', '')
                })
        
        # Build the user message
        # Always include document context if present, even for follow-up questions
        user_content = ''
        if context_parts:
            user_content = '\n\n'.join(context_parts)
            if user_message:
                # For follow-up questions, make it clear this is a question about the document
                if document_text and user_message:
                    user_content += f"\n\nUSER QUESTION (about the above document): {user_message}"
                else:
                    user_content += f"\n\nUSER QUESTION: {user_message}"
        else:
            user_content = user_message
        
        # If document exists but wasn't in context_parts (shouldn't happen, but safety check)
        if document_text and document_text not in user_content:
            user_content = f"HEALTH RECORD DOCUMENT:\n{document_text}\n\n{user_content}"
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        # Call Groq API with Llama model
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",  # Using Llama model
            temperature=0.7,
            max_tokens=2000,  # Increased for structured output
        )
        
        # Extract response
        response_text = chat_completion.choices[0].message.content
        
        # Clean up any markdown formatting that might have slipped through
        response_text = clean_response(response_text)
        
        return jsonify({
            'response': response_text,
            'success': True
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'error': 'An error occurred while processing your request',
            'details': str(e)
        }), 500

@app.route('/', methods=['GET'])
def index():
    """Simple root route to confirm the service is running."""
    return jsonify({
        'message': 'Horse Health Record Interpreter API is running.',
        'endpoints': {
            'health': '/health',
            'chat': '/api/chat'
        }
    }), 200

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_ENV') == 'development')
