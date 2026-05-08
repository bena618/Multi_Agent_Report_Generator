# Multi-Agent Research Report Generator

This is a small multi-agent system that takes a research question, breaks it into smaller tasks, gathers information, and then produces a final written report.

It's built with Flask and uses IBM watsonx.ai for the LLM calls, along with Redis for basic rate limiting.

---

## Live API

https://multiagentreportgenerator-production.up.railway.app

---

## How it works

When you send a query to the '/research' endpoint, the system runs it through three steps:

1. **Planner** – splits the question into smaller research tasks  
2. **Researcher** – answers each task and tries to include sources  
3. **Writer** – combines everything into a final report to deliver back to the user

Possible upgrade in the future would be a critic that acts kinda like 2FA to verify the informtion the agent retrieved is accurate(even though it does use in text citation) and possibly ensure that the research does match the question and the answer(ie. agent says yes but reasoning supports a no answer)


## Example request

```bash
curl -X POST "https://multiagentreportgenerator-production.up.railway.app/research" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"Do the benefits of solar energy outweigh the drawbacks?\"}"
```
