
# sales_agent_new.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langmem.short_term import SummarizationNode
from langchain_core.messages.utils import count_tokens_approximately
import uuid, json
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from tools.product_tools import tools  # your existing product tools

# Load key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize model
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")
model_with_tools = llm.bind_tools(tools)


# -------------------------------------------------------------------
# SYSTEM message — updated to require explicit confirmation and to limit memory usage
# -------------------------------------------------------------------
SYSTEM = SystemMessage(content="""
You are a concise, professional AI sales agent for an online product catalog. Only use the provided product tools for any product data or actions — do not invent products, details, or prices. Never ask the user for the CSV (it already exists).

Tool contract (use these EXACT tool names/signatures):
- filter_products_json(product_type: Optional[str], min_rating: Optional[float], min_price: Optional[float], max_price: Optional[float], top_n: int) -> JSON
  - Returns a JSON object with keys: success, count, recommendations (list of product dicts with product_id, product_name, price, rating, inventory_count, type, product_description).
  - Use this to produce the Top-5 recommendations.

- check_inventory_json(product_id: str) -> JSON
  - Returns JSON with success and in_stock (boolean) and inventory_count.

- checkout_product_json(product_id: str, quantity: int = 1) -> JSON
  - Returns JSON with success and order info (order_id, product_id, qty, total_price).

Hard behavior rules (follow exactly):
1. When user asks to buy or browse, call `filter_products_json(...)` to fetch recommendations (top_n <= 5). Infer parameters from the user's request (product type, min_rating, price range). If any required info is missing, ask exactly one concise clarifying question for that specific missing field. Repeat probing only as needed until required info is collected.
2. After receiving `filter_products_json`, present the Top-N recommendations (N <= 5) as a numbered list showing `product_id`, `product_name`, `price`, and `rating`. Ask the user to select by `product_id` or number.
3. **Do NOT call** `checkout_product_json` automatically. When a user selects a product (by id or number), you MUST call `check_inventory_json(product_id)` first and then **ask the user** for explicit confirmation to checkout. The explicit confirmation must be an affirmative token such as "yes", "confirm", or "checkout now". Only after the user explicitly confirms should you call `checkout_product_json(product_id, quantity)`. If out of stock, inform the user and offer an alternative from the last recommendations.
4. Never return raw JSON to the user. Use JSON only for internal logic; format human-readable replies.
5. If the user gives a vague purchase command (e.g., "Buy one of these"), resolve it using the most recent recommendations and if needed ask a single clarifying question: “Which product_id or number would you like?”.
6. Keep replies short, actionable, and tool-driven. Always follow tool outputs — do not hallucinate inventory, price, or product details.
7. STATE & MEMORY: When composing responses, prefer using only the **last 10** user/assistant exchanges to decide actions and prompts. Do not rely on older exchanges unless the user explicitly references them.

If you understand, proceed: if the user's initial request lacks required filters (type, rating, price range), ask a single concise clarifying question. Otherwise call `filter_products_json` with inferred or default values (default: top_n=5).
""")


# -------------------------------------------------------------------
# MEMORY (from LangGraph docs)
# -------------------------------------------------------------------
summ_model = llm.bind(max_tokens=128)
summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=summ_model,
    max_tokens=256,
    max_tokens_before_summary=256,
    max_summary_tokens=128,
)

memory = MemorySaver()


# -------------------------------------------------------------------
# LLM node — limit messages used to the last 10 exchanges
# -------------------------------------------------------------------
def llm_call(state: dict):
    """Model decides next step — use only last 10 exchanges when calling the LLM"""
    # state["messages"] is a list of Message objects; keep only last 10 entries
    msgs = state.get("summarized_messages") or state.get("messages") or []
    # limit to last 10 message objects for context (preserves ordering)
    msgs = msgs[-10:]
    return {
        "messages": [
            llm.invoke([SYSTEM] + msgs)
        ]
    }


# -------------------------------------------------------------------
# ROUTE logic — check if model called a tool
# -------------------------------------------------------------------
def route(state):
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


# -------------------------------------------------------------------
# Build simple graph (same as docs)
# -------------------------------------------------------------------
def build_agent():
    g = StateGraph(dict)
    g.add_node("summarize", summarization_node)
    g.add_node("llm_call", llm_call)
    g.add_node("tools", ToolNode(tools))

    g.add_edge(START, "summarize")
    g.add_edge("summarize", "llm_call")
    g.add_conditional_edges("llm_call", route, {"tools": "tools", END: END})
    g.add_edge("tools", "llm_call")

    return g.compile(checkpointer=memory)


agent = build_agent()


# -------------------------------------------------------------------
# Simple answer function (straight from docs)
# -------------------------------------------------------------------

def answer(thread_id: str, text: str) -> str:
    cfg = {"configurable": {"thread_id": thread_id}}
    final = ""

    hm = HumanMessage(id=str(uuid.uuid4()), content=text)

    # For debugging: collect streamed contents
    parts = []

    for event in agent.stream({"messages": [hm]}, config=cfg):
        for _, payload in event.items():
            msgs = payload.get("messages", [])
            if not msgs:
                continue
            last = msgs[-1]

            # If last is an AIMessage (final reply without tool_calls), accumulate
            if isinstance(last, AIMessage) and not getattr(last, "tool_calls", None):
                parts.append(last.content)
                final = "".join(parts).strip()

            # If last contains tool calls (tool call request), you may see ToolMessage responses
            # If ToolMessage content is JSON string, parse it for nicer display or follow-up
            if isinstance(last, ToolMessage):
                try:
                    parsed = json.loads(last.content)
                    # optionally convert parsed to human readable string
                    parts.append(f"[Tool response] {json.dumps(parsed, indent=2)}")
                except Exception:
                    # not JSON — just append raw content
                    parts.append(last.content)

    return final


# -------------------------------------------------------------------
# TEST (unchanged)
# -------------------------------------------------------------------
if __name__ == "__main__":
    tid = "test-thread"
    print(answer(tid, "I want a smartphone"))
    print(answer(tid, "4"))
    print(answer(tid, "under 1000"))
