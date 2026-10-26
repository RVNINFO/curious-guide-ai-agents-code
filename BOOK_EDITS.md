# Edits to make in the manuscript (resolves R-006)

Replace `<ORG>` once the repo is published, e.g. `https://github.com/<ORG>/curious-guide-ai-agents-code`.
Pin the book to a release tag (e.g. `v1.0`) rather than `main`, so printed instructions never drift.

## 1. Preface - new section "Code that goes with this book"

> Every code listing in this book is available, runnable, at **github.com/<ORG>/curious-guide-ai-agents-code**
> (release `v1.0`). Each listing is one file, and each has a test. You do not need an API key to run the tests:
> they replace the model with a scripted stand-in, so they check the parts you write (tools, limits, approvals,
> evals) rather than a model's judgment. To run a listing against a real model, follow the README.
> Agent frameworks change quickly; the repository pins the versions the book was written against.

## 2. Per-listing callout (one line under each code block)

Format: `Listing 3.7.1 - code/ch03/l3_7_1_human_in_the_loop.py  |  github.com/<ORG>/curious-guide-ai-agents-code`

Add a visible caption to every listing so readers can cite and find it. Suggested captions (numbers are the
section they sit in; extend with `-a`, `-b` where a section has two):

| Section | Caption | File |
|---|---|---|
| 3.1.5 | Listing 3.1.5 - The agent loop | ch03/l3_1_5_agent_loop.py |
| 3.2.3 | Listing 3.2.3 - A well-described tool | ch03/l3_2_3_order_status_tool.py |
| 3.3.1 | Listing 3.3.1 - Tool definition (JSON, Pydantic) | ch03/l3_3_1_tool_schema.py |
| 3.3.2 | Listing 3.3.2 - Structured output | ch03/l3_3_2_structured_output.py |
| 3.3.4 | Listing 3.3.4 - Tool-execution pipeline | ch03/l3_3_4_execute_tool_call.py |
| 3.4.1 | Listing 3.4.1 - Summarising old messages | ch03/l3_4_1_summarization_agent.py |
| 3.5.1/.4/.5 | Listings 3.5.1, 3.5.4, 3.5.5 - Embeddings, filtering, reranking | ch03/l3_5_rag.py |
| 3.6.1 | Listing 3.6.1 - Planning with a todo list | ch03/l3_6_1_todo_planning.py |
| 3.6.3 | Listing 3.6.3 - Retry with the validation error | ch03/l3_6_3_generate_valid.py |
| 3.7.1 | Listing 3.7.1 - Human approval | ch03/l3_7_1_human_in_the_loop.py |
| 3.8.1 | Listing 3.8.1 - Router | ch03/l3_8_1_router.py |
| 3.8.3 | Listing 3.8.3 - Reflection | ch03/l3_8_3_reflection.py |
| 3.8.5 | Listing 3.8.5 - Reviewer | ch03/l3_8_5_reviewer.py |
| 3.11.1 | Listing 3.11.1 - A minimal MCP server | ch03/l3_11_1_mcp_server.py |
| 3.12.3 | Listing 3.12.3 - A minimal A2A client | ch03/l3_12_3_a2a_client.py |
| 4.2.3 | Listing 4.2.3 - LLM-as-judge | ch04/l4_2_3_llm_judge.py |
| 5.x | Listing 5.x - Tenant-scoped refund tool | ch05/l5_tenant_scoped_refund.py |
| 5.11.1 | Listing 5.11.1 - The complete agent | ch05/l5_11_1_capstone.py |

Also: the existing header comments ("# Framework: LangChain 1.x") can stay; the callout adds the location.
In 5.11.1 the listing is printed in four pieces (tools / agent / running / evals); put one callout after the last.

## 3. Defects in the printed code found while building the repo (fix in the manuscript)

1. **Listing 3.3.1 JSON is not valid JSON.** The `"description"` string is broken across lines with literal
   newlines. Use `\n` escapes (see `data/search_orders.tool.json`) or collapse it to one line.
2. **Listing 3.1.5 vs Ex. 3 starter disagree on the response type.** The loop reads `response.message.content`
   and appends `response.message`, but the exercise's `Response` has only `.content` and `.tool_calls`, so a
   reader copying the loop into the exercise gets `AttributeError`. Either add a `message` property to the
   exercise class or use `response.content` in the listing.
3. **Listing 3.5.4 filter syntax is store-specific.** The dict `filter={...}` works for stores such as Chroma but
   not for `InMemoryVectorStore` (needs a callable). Say which store the example assumes.
4. **Listing 3.12.3 (A2A) is not runnable as printed**: `get_task` is defined before `BASE`, step comment "1."
   appears twice, `token` is a placeholder, and `supplier.example.com` does not exist. The repo version wraps it
   in functions with an injectable HTTP client; consider printing that shape.
5. **Versions.** The book says "FastMCP 2.x"; the current release is 4.x. The listing works unchanged on 4.1.4,
   but the printed label is stale. State the tested versions (see `requirements.txt`).
6. **Model IDs** (`claude-sonnet-4-6`, `claude-haiku-4-5`) were not verified against a live account. Confirm
   them before release, or print a placeholder constant (`MODEL = ...`).
7. **Listing 4.x `score_case` starter** has an over-indented docstring (cosmetic, but readers copy it).
8. **Cross-tenant test (4.x)** is pseudocode that calls an undefined `run_agent(request=..., test_identity=...,
   fixtures=...)`. The repo provides it in `ch05/harness.py`; mention that in the text.
9. **Listing 3.2.3 / 5.x** use `db` / `payments` that do not exist in the text. The repo supplies in-memory stand-ins;
   the note "application-owned adapters" is there for 5.x but 3.2.3 only says "your own database client".

## 4. Publish checklist (needs you - I can't do these from here)

```bash
cd curious-guide-agents-code
# choose and add a LICENSE (MIT/Apache-2.0 are typical for companion code) - your decision
git init && git add . && git commit -m "Companion code for A Curious Guide to AI Agents"
gh repo create <ORG>/curious-guide-ai-agents-code --public --source . --push
git tag v1.0 && git push origin v1.0
```
Then confirm the Actions run is green, paste the URL into the three places above, and set R-006 to Done.
