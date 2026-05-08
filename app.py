import os
import json
import logging
import re
import time
import uuid
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from upstash_redis import Redis

load_dotenv()

app = FastAPI(title="Multi-Agent Research Report Generator")

UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL")
UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")
redis = Redis(url=UPSTASH_REDIS_REST_URL, token=UPSTASH_REDIS_REST_TOKEN)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IBM_API_KEY = os.getenv("IBM_API_KEY")
IBM_PROJECT_ID = os.getenv("IBM_PROJECT_ID")
IBM_URL = os.getenv("IBM_URL")
MODEL_ID = "mistralai/mistral-small-3-1-24b-instruct-2503"

credentials = {"url": IBM_URL, "apikey": IBM_API_KEY}

PLANNER_PARAMS = {GenParams.MAX_NEW_TOKENS: 200, GenParams.TEMPERATURE: 0.1}
WRITER_PARAMS = {GenParams.MAX_NEW_TOKENS: 500, GenParams.TEMPERATURE: 0.7}
RESEARCHER_PARAMS = {GenParams.MAX_NEW_TOKENS: 200, GenParams.TEMPERATURE: 0.3}

class ResearchQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)

class SubtaskList(BaseModel):
    subtasks: list[str] = Field(..., min_items=2)

class ResearchItem(BaseModel):
    answer: str = Field(...)
    source: str = Field(...)

class ResearchResult(BaseModel):
    items: list[ResearchItem] = Field(..., min_items=1)

class FinalReport(BaseModel):
    report: str


def validate_input(user_input: str) -> tuple[bool, str]:
    if not user_input or not user_input.strip():
        return False, "Empty input not allowed"
    if len(user_input) > 1000:
        return False, "Input exceeds maximum length of 1000 characters"
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
    now = int(time.time())
    minute_key = f"{client_ip}:{now // 60}"
    try:
        current_count = redis.get(minute_key)
        if current_count is None:
            redis.setex(minute_key, 60, 1)
            return True, "OK", False
        new_count = int(current_count) + 1
        if new_count > 10:
            return False, "Rate limit exceeded (10/min)", False
        redis.setex(minute_key, 60, new_count)
        return True, "OK", False
    except Exception as e:
        logger.error(f"Redis error: {e}")
        return False, "Rate limit service unavailable", True

def sanitize_response(response: str) -> str:
    text = str(response)
    text = text.replace("\x00", "")
    text = text.replace('\\t', '\t').replace('\\r', '\r').replace('\\n', '\n')
    max_len = 2100
    if len(text) > max_len:
        text = text[:max_len] + "\n\n[Response truncated]"
    return text.strip()

def call_llm(system_prompt: str, user_content: str, params: dict = None, sanitize: bool = True) -> str:
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
    safety_prompt = """
    You are a content safety classifier. Determine if the user's query asks for:
    - Illegal activities (hacking, theft, fraud, violence, weapons, drugs, etc.)
    - Hate speech or harassment
    - Self-harm or harm to others
    - Any other NSFW or inappropriate content

    If safe, output 'SAFE'; if unsafe, output 'UNSAFE'. No other output.
    """
    response = call_llm(safety_prompt, user_query)
    if "UNSAFE" in response.strip().upper():
        return False, "Query blocked: does not comply with content policy"
    return True, "Query is safe"

def planner_agent(user_query: str) -> SubtaskList:
    system_prompt = """
    You are a Planner Agent. Break the user's request into a JSON list of subtasks.
    Output ONLY valid JSON, like ["task1", "task2"]. Minimum 2 subtasks.
    """
    response = call_llm(system_prompt, user_query, PLANNER_PARAMS, sanitize=False)
    logger.info(f"Planner response: {response}")
    try:
        data = json.loads(response)
        return SubtaskList(subtasks=data)
    except json.JSONDecodeError:
        logger.warning(f"Planner invalid JSON, using fallback: {response[:200]}")
        return SubtaskList(subtasks=[user_query])

def researcher_agent(subtasks: list) -> ResearchResult:
    results = []
    for task in subtasks:
        system_prompt = """
        You are a Researcher Agent. Provide a concise, factual answer (1-2 sentences) to the subtask.
        Include a credible source. Output JSON: {"answer": "...", "source": "..."}
        """
        user_content = "Subtask: " + task
        response = call_llm(system_prompt, user_content, RESEARCHER_PARAMS, sanitize=False)
        try:
            item = ResearchItem.model_validate_json(response)
        except:
            item = ResearchItem(answer=response, source="LLM (no external source)")
        results.append(item)
    return ResearchResult(items=results)

def writer_agent(user_query: str, research: ResearchResult) -> str:
    research_text = "\n".join([f"{item.answer} (Source: {item.source})" for item in research.items])
    system_prompt = """
    You are a Report Writer. Synthesize the research into 2-3 paragraphs with inline citations.
    """
    user_content = f"Original query: {user_query}\n\nResearch with sources:\n{research_text}"
    return call_llm(system_prompt, user_content, WRITER_PARAMS, sanitize=True)

def orchestrator(user_query: str, request_id: str) -> dict:
    try:
        logger.info(f"[{request_id}] Processing query: {user_query[:20]}...")
        subtasks = planner_agent(user_query)
        logger.info(f"[{request_id}] Planner generated {len(subtasks.subtasks)} subtasks")
        research = researcher_agent(subtasks.subtasks)
        logger.info(f"[{request_id}] Researcher completed {len(research.items)} items")
        final_report = writer_agent(user_query, research)
        logger.info(f"[{request_id}] Writer generated report")
        return {"status": "success", "research": research.model_dump(), "final_report": final_report}
    except Exception as e:
        logger.error(f"[{request_id}] Orchestration failed: {e}")
        return {"status": "failed", "error": "Something went wrong"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/research")
async def research(request: Request, query_data: ResearchQuery):
    request_id = str(uuid.uuid4())
    client_ip = request.client.host
    logger.info(f"Request {request_id} from IP: {client_ip}")

    is_allowed, rate_message, is_redis_error = rate_limit_check(client_ip)
    if not is_allowed:
        status_code = 503 if is_redis_error else 429
        raise HTTPException(status_code=status_code, detail=rate_message)

    query = query_data.query

    is_valid, val_msg = validate_input(query)
    if not is_valid:
        raise HTTPException(status_code=400, detail=val_msg)

    is_safe, safe_msg = is_safe_query(query)
    if not is_safe:
        logger.warning(f"[{request_id}] Blocked unsafe query: {query[:20]}")
        raise HTTPException(status_code=400, detail=safe_msg)

    result = orchestrator(query, request_id)
    logger.info(f"Request {request_id} completed: {result.get('status')}")
    return result

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
