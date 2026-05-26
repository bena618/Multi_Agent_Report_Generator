# Multi-Agent Research Report Generator

This is a multi-agent system that takes a research question, breaks it into smaller tasks, executes parallel research with dependency tracking, verifies claims via Google Search, and synthesizes cited reseach to output a report for the end user. Built with FastAPI, LangGraph, IBM watsonx.ai for LLM calls, and Redis for basic rate limiting.

---

## Live API

**Base URL**: https://multiagentreportgenerator-production.up.railway.app

**Health Check**: GET /health

**Research endpoint**: POST /research

---

## How it works

The system uses graph-based agent orchestration (LangGraph) with 4 specificialized agents:

When you send a query to the '/research' endpoint, the system runs it through three steps:

1. **Planner** – splits the users query into smaller research tasks with logical dependencies
2. **Researcher** – answers each task in parallel returning answers with sources
3. **Fact Checker** – verifies information from the researcher with Google Custom Search
4. **Writer** – combines everything into a final report to deliver back to the user with in-line citations

## Security Features:
- **Two-layer prompt injection defense**(regex + LLM classifier)
- **Rate limiting via Redis** (10 requests/minute, fail-closed)
- **LLM-based content safety classification**
- **Response sanitization**(XSS,Javascript,CSS protection, etc.)
- **Request ID logging** for traceability without exposing sensitive data/PII

## Performance Optimizations:
- **Parallel execution** of independant research tasks (asyncio.gather)
- **Model caching** LLMs cached by paremeter configuration to avoid redundant LLM initializations
- **Rate limiting compliance** using `aiolimiter` to follow IBM 2 request/second limit
- **Dependency-aware scheduling** Tasks are scheduled based on their dependencies to ensure that tasks are only executed when their dependencies are met

## Other Features:
- **Google Custom Search integration** for fact checking
- **Built in guardrails** to prevent going over rate limits since this is built on free tiers


## Example request

```bash
curl -X POST "https://multiagentreportgenerator-production.up.railway.app/research" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"Do the benefits of solar energy outweigh the drawbacks?\"}"
```
