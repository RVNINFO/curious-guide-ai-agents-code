# Listing index

Book section → file → test. "Offline" means it runs with no API key (scripted fake model);
"Live" means the same code also runs against a real provider (see README).

| Book § | Listing | File | Test | Mode |
|---|---|---|---|---|
| 3.1.5 | Agent loop (framework-free) | `ch03/l3_1_5_agent_loop.py` | `test_agent_loop_*` | Offline |
| 3.2.3 | `get_order_status` tool | `ch03/l3_2_3_order_status_tool.py` | `test_get_order_status` | Offline |
| 3.3.1 | Tool schema (JSON + Pydantic) | `ch03/l3_3_1_tool_schema.py`, `data/search_orders.tool.json` | `test_search_orders_schema_*` | Offline |
| 3.3.2 | Structured output / `TicketTriage` | `ch03/l3_3_2_structured_output.py` | `test_triage_*` | Offline + Live |
| 3.3.4 | Tool-execution pipeline | `ch03/l3_3_4_execute_tool_call.py` | `test_pipeline_*` | Offline |
| 3.4.1 | `SummarizationMiddleware` agent | `ch03/l3_4_1_summarization_agent.py` | `test_summarization_*` | Offline + Live |
| 3.5.1 / 3.5.4 / 3.5.5 | Embeddings, filtered search, reranking | `ch03/l3_5_rag.py` | `test_rag_*` | Offline (fake embeddings) + Live (OpenAI, Cohere) |
| 3.6.1 | `TodoListMiddleware` planning | `ch03/l3_6_1_todo_planning.py` | `test_todo_planning_agent_*` | Offline + Live |
| 3.6.3 | `generate_valid` self-correction | `ch03/l3_6_3_generate_valid.py` | `test_generate_valid_*` | Offline |
| 3.7.1 | Human-in-the-loop approval | `ch03/l3_7_1_human_in_the_loop.py` | `test_hitl_*` | Offline + Live |
| 3.8.1 | Router pattern | `ch03/l3_8_1_router.py` | `test_router_*` | Offline |
| 3.8.3 | Reflection pattern | `ch03/l3_8_3_reflection.py` | `test_reflection_*` | Offline |
| 3.8.5 | Reviewer pattern | `ch03/l3_8_5_reviewer.py` | `test_reviewer_paths` | Offline |
| 3.11.1 | MCP server (FastMCP) | `ch03/l3_11_1_mcp_server.py`, `data/mcp_tools_*.jsonrpc.json` | `test_mcp_server_*` | Offline (in-memory client) |
| 3.12.2 | Agent Card JSON | `data/agent_card.json` | `test_agent_card_*` | Data |
| 3.12.3 | A2A client | `ch03/l3_12_3_a2a_client.py` | `test_a2a_client_*` | Offline (mock transport) |
| 3.4.3 | Memory record schema | `data/memory_record.json` | `test_agent_card_and_memory_record_shapes` | Data |
| 4.2.2 / 4.2.3 | Eval case + LLM-judge prompt | `ch04/l4_2_3_llm_judge.py`, `data/eval_case_*.json` | `test_judge_*`, `test_process_scoring_*` | Offline + Live |
| 4.x | Cross-tenant eval case + test | `data/eval_case_cross_tenant.json`, `ch05/harness.py` | `test_cross_tenant_refund_is_denied` | Offline |
| 5.x | Tenant-scoped refund tool | `ch05/l5_tenant_scoped_refund.py`, `ch05/adapters.py` | `test_refund_rules`, `test_same_tenant_*` | Offline + Live |
| 5.x | Typed tool results | `ch05/l5_tool_result_types.py` | `test_tool_result_types_*` | Offline |
| 5.3 | Step-3 tool result JSON | `data/tool_result_lookup_order.json` | `test_data_files_are_valid_json` | Data |
| 5.11.1 | Complete agent (tools, approval, evals) | `ch05/l5_11_1_capstone.py`, `ch05/scripted.py` | `test_capstone_*` | Offline + Live (`--live`) |
| Ex. 3 | Agent-loop starter / solution | `exercises/ex_3_1_5_*`, `solutions/ex_3_1_5_*` | `test_exercise_agent_loop_*` | Offline |
| Ex. 4.2 | Eval scoring starter / solution | `exercises/ex_4_*`, `solutions/ex_4_*` | `test_exercise_score_case_*` | Offline |
| Ex. 5.x | Refund-window starter / solution | `exercises/ex_5_*`, `solutions/ex_5_*` | `test_exercise_refund_window_*` | Offline |

Section numbers for 3.4.3, 3.12.2, 4.x and 5.x items are approximate; confirm them against the final
manuscript when you add the "Listing" captions (see `BOOK_EDITS.md`).
