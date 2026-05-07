import os
import json
import logging
import re
import time
import html
import uuid
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from upstash_redis import Redis

load_dotenv()

app = Flask(__name__)

UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL")
UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")

redis = Redis(
    url=UPSTASH_REDIS_REST_URL,
    token=UPSTASH_REDIS_REST_TOKEN
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IBM_API_KEY = os.getenv("IBM_API_KEY")
IBM_PROJECT_ID = os.getenv("IBM_PROJECT_ID")
IBM_URL = os.getenv("IBM_URL")

MODEL_ID = "mistralai/mistral-small-3-1-24b-instruct-2503"

credentials = {
    "url": IBM_URL,
    "apikey": IBM_API_KEY
}


PLANNER_PARAMS = {
    GenParams.MAX_NEW_TOKENS: 200,
    GenParams.TEMPERATURE: 0.1,
}

WRITER_PARAMS = {
    GenParams.MAX_NEW_TOKENS: 500,
    GenParams.TEMPERATURE: 0.7,
}

RESEARCHER_PARAMS = {
    GenParams.MAX_NEW_TOKENS: 200,
    GenParams.TEMPERATURE: 0.3,
}

def validate_input(user_input: str) -> tuple[bool, str]:
    """Basic checks + tries to catch prompt injection."""
    if not user_input or not user_input.strip():
        return False, "Empty input not allowed"

    #Arbitrary length limit to prevent abuse
    if len(user_input) > 1000:
        return False, "Input exceeds maximum length of 1000 characters"
   
    # Protection aginst some common prompt injection attacks
    injection_patterns = [
        r'(?i)(ignore|forget|disregard)\s+(previous|all|your)\s+(instructions|rules|prompts)',
        r'(?i)(you\s+are|act\s+as|pretend\s+to\s+be)\s+(not|no\s+longer)',
        r'(?i)(system|admin|root|god)\s+(mode|access|privileges)',
        r'(?i)(execute|run|eval)\s+(this|the)\s+(code|command|script)',
        r'(?i)(<\?php|<script|javascript:|data:)',
        r'(?i)(__import__|eval|exec|open|file)\s*\(',
    ]
   
    for pattern in injection_patterns:
        if re.search(pattern, user_input):
            logger.warning(f"Prompt injection attempt detected: {pattern}")
            return False, "Invalid input"

    return True, "Input validated"

def rate_limit_check(client_ip: str) -> tuple[bool, str, bool]:
    '''
    Returns (is_allowed, message, is_redis_issue)
    retuns is_redis_issue to differentiate if rate limit issue or redis issue
    '''

    now = int(time.time())
    minute_key = f"{client_ip}:{now // 60}"

    try:
        current_count = redis.get(minute_key)
        if current_count is None:
            redis.setex(minute_key, 60, 1)
            return True, "Rate Limit Check Passed", False
        new_count = int(current_count) + 1
       
        if new_count > 10:
            return False, "Rate Limit Check Failed: Exceeded 10 requests per minute", False
        redis.setex(minute_key, 60, new_count)
        return True, "Rate Limit Check Passed", False

    except Exception as e:
        logger.error(f"Redis rate limit error: {str(e)}")
        # Security: if Redis is down, block any attempts to use agent
        return False, "Rate limit unavailable", True


def sanitize_response(response: str) -> str:
    text = str(response)

    # Remove null bytes
    text = text.replace("\x00", "")

    max_len = 2100

    if len(text) > max_len:
        text = text[:max_len] + "\n\n[Response truncated]"

    return text.strip()

def call_llm(system_prompt: str, user_content: str, params:dict=None, sanitize=True) -> str:
    try:
        model = ModelInference(
            model_id=MODEL_ID,
            credentials=credentials,
            project_id=IBM_PROJECT_ID,
            params=params
        )

        full_prompt = system_prompt + "\n\nUser request: " + user_content + "\n\nResponse:"

        logger.info(f"LLM call initiated - Prompt length: {len(full_prompt)}")

        response = model.generate_text(prompt=full_prompt)

        response = response.strip()

        if sanitize:
            response = sanitize_response(response)

        logger.info(f"LLM call completed - Response length: {len(response)}")

        return response

    except Exception as e:
        logger.error(f"LLM call failed: {str(e)}")
        raise
def is_safe_query(user_query: str) -> tuple[bool, str]:
    """LLM decides if query is safe or violtes any policy/guidelines"""

    safety_prompt = """
    You are a content safety classifier. Determine if the user's query asks for:
    - Illegal activities (hacking, theft, fraud, violence, weapons, drugs, etc.)
    - Hate speech or harassment
    - Self-harm or harm to others
    - Any other NSFW or inappropriate content

    If the query is safe for a general research assistant, output the word: 'SAFE', 
    otherwise output the word 'UNSAFE'

    Do not provide any other output.
    """
   
    response = call_llm(safety_prompt, user_query)
    if "UNSAFE" in response.strip().upper():
        return False, "Query blocked: does not comply with content policy"
    return True, "Query is safe"

def planner_agent(user_query: str) -> list:
    """
    Break down the user's request into a JSON list of subtasks
    """
    system_prompt = '''
    You are a planner agent.
    You are a Planner Agent. Break the user's request into a JSON list of subtasks.
    Output ONLY valid JSON, like ["task1", "task2"]. Do not include any other text or explanation.

    Minimum of 2 subtasks required. If it is a simple query then can still still breakdown into
    fundamental subtasks like "What is X?", "Why is X important?", "Who created X?", etc.
    '''

    response = call_llm(system_prompt, user_query, PLANNER_PARAMS, sanitize=False)
    logging.info(f"Planner response: {response}")
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        #Fallback to treat the whole thing as one task
        logging.warning(f"Planner output invalid JSON, using fallback. Output was: {response[:200]}")
        return [user_query]

def researcher_agent(subtasks: list) -> list:
    results = []
    # Wanted a real fact-checking tool but couldn't find a free one, so just having
    # the LLM include sources for transparency and so can be fact checked by the user
    for task in subtasks:
        system_prompt = """
        You are a Researcher Agent. Provide a concise, factual answer (1-2 sentences) to the subtask.

        When possible please include a credible source (website, newspaper, publication, etc.) for each claim, where a user could verify the information.

        Output JSON with this exact structure:
        {"answer": "The answer to the subtask", "source": "The source of the information"}
        """
        user_content = "Subtask: " + task
        response = call_llm(system_prompt, user_content, RESEARCHER_PARAMS, sanitize=False)
        try:
            item = json.loads(response)
        except:
            # Fallback if JSON parsing fails
            item = {"answer": response, "source": "LLM (no external source)"}
        results.append(item)
    return results

def writer_agent(user_query: str, research: list) -> str:
    # Fomulates all of the research into a coherent final answer
    research_text = "\n".join([item.get('answer', 'No answer') + " (Source: " + item.get('source', 'Unknown') + ")" for item in research])

    system_prompt = """
    You are a Report Writer. Synthesize the research into a clear and coherent final answer (2-3 paragraphs).
    Include in-line citations for each claim using the source information provided.
    """

    user_content = "Original query: " + user_query + "\n\nResearch with sources:\n" + research_text
    return call_llm(system_prompt, user_content, WRITER_PARAMS, sanitize=True)

def orchestrator(user_query: str, request_id: str) -> dict:
    try:
        logger.info(f"[{request_id}] Processing query: {user_query[:100]}...")
       
        subtasks = planner_agent(user_query)
        logger.info(f"[{request_id}] Planner generated {len(subtasks)} subtasks")
       
        research = researcher_agent(subtasks)
        logger.info(f"[{request_id}] Researcher completed {len(research)} items")
           
        final_report = writer_agent(user_query, research)
        logger.info(f"[{request_id}] Writer generated final report")
               
        return {
            "status": "success",
            "final_report": final_report
        }
               
    except Exception as e:
        logger.error(f"[{request_id}] Orchestration failed: {str(e)}")
        return {
            "status": "failed",
            "error": "Something went wrong"
        }

@app.route('/research', methods=['POST'])
def research():
    try:

        request_id = str(uuid.uuid4())

        # Get the caller's ip and possible proxies for checking rate limits
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        
        logger.info(f"Request {request_id} from IP: {client_ip}")

        is_allowed, rate_message, is_redis_error = rate_limit_check(client_ip)
        if not is_allowed:
            if is_redis_error:
                logger.warning(f"[{request_id}] Rate limit check failed due to Redis issue")
                return jsonify({"error": rate_message}), 503
            logger.warning(f"[{request_id}] Rate limit exceeded for IP: {client_ip}")
            return jsonify({"error": rate_message}), 429
       
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' field or JSON body"}), 400
       
        query = data['query']
       
        is_valid, validation_message = validate_input(query)
        if not is_valid:
            logger.warning(f"[{request_id}] Input validation failed: {validation_message}")
            return jsonify({"error": validation_message}), 400

        is_safe, safety_message = is_safe_query(query)
        if not is_safe:
            logger.warning(f"[{request_id}] Blocking unsafe query: {query[:100]}")
            return jsonify({"error": safety_message}), 400
              
        result = orchestrator(query, request_id)
       
        logger.info(f"Request {request_id} completed with status: {result.get('status', 'unknown')}")
       
        return jsonify(result), 200
       
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error in research endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
