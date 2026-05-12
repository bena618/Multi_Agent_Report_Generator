import os
import json
import logging
import re
import time
import uuid
import asyncio
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Dict, Any
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

PLANNER_PARAMS = {GenParams.MAX_NEW_TOKENS: 400, GenParams.TEMPERATURE: 0.1}
WRITER_PARAMS = {GenParams.MAX_NEW_TOKENS: 500, GenParams.TEMPERATURE: 0.7}
RESEARCHER_PARAMS = {GenParams.MAX_NEW_TOKENS: 200, GenParams.TEMPERATURE: 0.3}

class ResearchQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)

class SubtaskDependency(BaseModel):
    id: int = Field(...)
    task: str = Field(...)
    depends_on: List[int] = Field(default_factory=list)

class PlannerOutput(BaseModel):
    subtasks: List[SubtaskDependency] = Field(...)

class ResearchItem(BaseModel):
    answer: str = Field(...)
    source: str = Field(...)

class PlannerError(Exception):
    pass

class AgentState(TypedDict):
    query: str
    request_id: str
    planner_output: Optional[Dict[str, Any]]
    research_results: Optional[Dict[int, ResearchItem]]
    final_report: Optional[str]

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
            return False, "Prompt injection attack detected: Request denied"
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

def planner_agent(user_query: str) -> PlannerOutput:
    system_prompt = """
    You are a Planner Agent. Break the user's request into subtasks with dependencies.

    Output ONLY valid JSON with this exact structure:
    {
    "subtasks": [
        {"id": 1, "task": "First subtask question", "depends_on": []},
        {"id": 2, "task": "Second subtask question", "depends_on": []}
    ]
    }

    CRITICAL RULES:
    - Each subtask MUST have a unique numeric id (1, 2, 3, ...)
    - "depends_on" is a list of ids that this subtask needs BEFORE it can run
    - If a subtask has no dependencies, use an empty list []
    - Base dependencies on logical flow and information hierarchy/need, not any specific pattern
    - Do not force comparison structure or anything that may force dependencies if the user's query doesn't require it, such as a simple fact query
    - Do not include any other text or explanation outside the JSON

    Now please analyse the user's query and break it down into appropriate subtasks with dependencies:
    """
    response = call_llm(system_prompt, user_query, PLANNER_PARAMS, sanitize=False)
    logger.info(f"Planner response: {response[:500]}")
    try:
        # Clean response if it has markdown
        cleaned = response.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.startswith('```'):
            cleaned = cleaned[3:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        data = json.loads(cleaned.strip())
        return PlannerOutput(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.warning(f"Planner failed: {e}")
        raise HTTPException(
            status_code=400,
            detail="Planner failed to break down task into subqueries. Please try again with a more detailed/contextualized prompt"
        )

async def run_single_researcher(
    task: str, 
    request_id: str, 
    previous_results: List[ResearchItem] = None
) -> ResearchItem:
    """Run a single research task with optional structured context from previous subtasks"""
    logger.info(f"[{request_id}] Researching: {task[:50]}...")
    
    system_prompt = """
    You are a Researcher Agent. Provide a concise, factual answer (1-2 sentences) to the subtask.
    Include a credible source. Output JSON: {"answer": "...", "source": "..."}
    """
    
    # Build context string from structured previous results
    if previous_results:
        context_parts = []
        for item in previous_results:
            context_parts.append(f"- {item.answer} (Source: {item.source})")
        context = "Relevant information from previous research:\n" + "\n".join(context_parts)
        user_content = f"{context}\n\nNow answer this specific subtask: {task}"
    else:
        user_content = "Subtask: " + task
    
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: call_llm(system_prompt, user_content, RESEARCHER_PARAMS, sanitize=False)
    )
    try:
        return ResearchItem.model_validate_json(response)
    except:
        return ResearchItem(answer=response, source="LLM (no external source)")

async def execute_dependent_subtasks(planner_output: PlannerOutput, request_id: str) -> Dict[int, ResearchItem]:
    """Execute subtasks respecting dependencies, passing structured context from completed subtasks"""
    results: Dict[int, ResearchItem] = {}
    completed = set()
   
    logger.info(f"[{request_id}] Dependency graph: {[(s.id, s.depends_on) for s in planner_output.subtasks]}")
   
    while len(completed) < len(planner_output.subtasks):
        # Find subtasks whose dependencies are satisfied
        ready = [
            sub for sub in planner_output.subtasks
            if sub.id not in completed
            and all(dep in completed for dep in sub.depends_on)
        ]
       
        if not ready:
            remaining = [sub.id for sub in planner_output.subtasks if sub.id not in completed]
            dep_strings = [f"{sub.id}: depends on {sub.depends_on}" for sub in planner_output.subtasks if sub.id in remaining]
            raise Exception(f"Circular dependency detected among {remaining}. Dependencies: {dep_strings}")
       
        # For each ready subtask, collect structured previous results
        for sub in ready:
            previous_results = []
            if sub.depends_on:
                for dep_id in sub.depends_on:
                    if dep_id in results:
                        previous_results.append(results[dep_id])
                logger.info(f"[{request_id}] Subtask {sub.id} depends on {sub.depends_on}, passing {len(previous_results)} previous result(s)")
            
            # Run researcher with structured previous results
            results[sub.id] = await run_single_researcher(sub.task, request_id, previous_results if previous_results else None)
            completed.add(sub.id)
   
    return results

def writer_agent(user_query: str, research_results: Dict[int, ResearchItem], subtasks: List[SubtaskDependency]) -> str:
    """Generate final report from research results preserving all sources"""
    # Order by original subtask order
    ordered_items = []
    for sub in subtasks:
        if sub.id in research_results:
            item = research_results[sub.id]
            ordered_items.append(f"{item.answer} (Source: {item.source})")
   
    research_text = "\n".join(ordered_items)
    system_prompt = """
    You are a Report Writer. Synthesize the research into 2-3 paragraphs with inline citations.
    Preserve the original sources when citing information.

    If the answer to the query is a simple fact or other straightforward information, 
    do not overcomplicate the response to get to paragraph length, just answer the question directly.
    """
    user_content = f"Original query: {user_query}\n\nResearch with sources:\n{research_text}"
    return call_llm(system_prompt, user_content, WRITER_PARAMS, sanitize=True)

def planner_node(state: AgentState) -> AgentState:
    logger.info(f"[{state['request_id']}] LangGraph planner node")
    planner_output = planner_agent(state["query"])
    return {**state, "planner_output": planner_output.model_dump()}

async def researcher_node(state: AgentState) -> AgentState:
    logger.info(f"[{state['request_id']}] LangGraph researcher node ")
    planner_output = PlannerOutput(**state["planner_output"])
    research_results = await execute_dependent_subtasks(planner_output, state["request_id"])

    # Convert to serializable dict
    results_dict = {k: v.model_dump() for k, v in research_results.items()}
    return {**state, "research_results": results_dict}

async def writer_node(state: AgentState) -> AgentState:
    logger.info(f"[{state['request_id']}] LangGraph writer node")
    planner_output = PlannerOutput(**state["planner_output"])
    research_results = {int(k): ResearchItem(**v) for k, v in state["research_results"].items()}
    report = writer_agent(state["query"], research_results, planner_output.subtasks)
    return {**state, "final_report": report}

builder = StateGraph(AgentState)
builder.add_node("planner", planner_node)
builder.add_node("researcher", researcher_node)
builder.add_node("writer", writer_node)

builder.set_entry_point("planner")
builder.add_edge("planner", "researcher")
builder.add_edge("researcher", "writer")
builder.add_edge("writer", END)

graph = builder.compile()

async def run_agent_graph(query: str, request_id: str) -> dict:
    initial_state = {
        "query": query,
        "request_id": request_id,
        "planner_output": None,
        "research_results": None,
        "final_report": None
    }
    final_state = await graph.ainvoke(initial_state)
   
    research_list = []
    if final_state.get("research_results"):
        planner_output = PlannerOutput(**final_state["planner_output"])
        for sub in planner_output.subtasks:
            if sub.id in final_state["research_results"]:
                result = final_state["research_results"][sub.id]
                research_list.append({"answer": result["answer"], "source": result["source"]})
   
    return {
        "status": "success",
        "research": research_list,
        "final_report": final_state.get("final_report")
    }

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

    result = await run_agent_graph(query, request_id)
    logger.info(f"Request {request_id} completed: {result.get('status')}")
    return result

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port) 
