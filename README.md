# 🚀 AI Sales Assistant

An intelligent product-discovery and recommendation engine powered by LLMs and structured tools.
The AI agent understands natural language queries, filters products based on specifications, checks inventory, and guides users through product decisions.
FastAPI serves as the main backend for production use, while a Streamlit UI is available as an optional lightweight interface.

---

## 1. Overview

This project implements an AI-powered **Sales Assistant** that helps users:

* Search for products by name
* Filter items by attributes (brand, RAM, storage, price, category, etc.)
* Receive intelligent recommendations
* Check inventory availability
* Simulate checkout actions

Under the hood, the system combines:

* **FastAPI backend** for serving the agent
* **LLM reasoning** for natural-language understanding
* **Tool-based structured functions** for deterministic filtering and searching
* **Optional Streamlit frontend** for interactive chat

---

## 2. Architecture

### Core Components

**1. `product_tools.py`**
Defines structured tools that the LLM can call:

* `search_by_name(name)`
* `filter_products(criteria)`
* `check_inventory(product_id)`
* `checkout_product(product_id)`

These operate on an internal product catalog.

**2. `sales_agent_new.py`**
Implements the intelligent agent:

* Parses user messages
* Uses LLM function calling
* Routes requests to tools
* Returns final natural-language answers
* Maintains minimal conversation context

The main function `answer(thread_id, user_input)` is the primary entrypoint for FastAPI.

**3. `streamlit_app_sales.py` (optional)**
A simple Streamlit chat interface that sends queries to the agent and displays results.

---

## 3. Features

### Natural-language Product Querying

Users can ask questions such as:

* “Show me Samsung phones under 15000.”
* “Anything with 8GB RAM and 128GB storage?”
* “Do you have laptops below 50k?”

The agent analyzes intent and triggers the correct tool.

### Intelligent Tool Routing

The LLM selects the appropriate function:

* Filtering products by specs
* Searching by approximate name
* Checking inventory
* Recommending alternatives

Tool responses are converted to friendly answers.

### Extensible Product Catalog

The catalog in `product_tools.py` is easily replaceable with:

* Database connections
* External APIs
* CSV imports

### Multi-Interface Support

The core agent is backend-agnostic and can be used through:

* **FastAPI (recommended)**
* **Streamlit UI (optional)**
* CLI or other UIs (custom)

---

## 4. Getting Started

### Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure you have an OpenAI API key set:

```bash
export OPENAI_API_KEY="your_key_here"
```

---

## 5. Running the FastAPI Server (Primary Interface)

Create a simple FastAPI app:

```python
from fastapi import FastAPI
from sales_agent_new import answer

app = FastAPI()

@app.post("/query")
async def run_agent(thread_id: str, user_input: str):
    return answer(thread_id, user_input)
```

Start the server:

```bash
uvicorn api:app --reload
```

Send a test request:

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"thread_id": "123", "user_input": "phones under 15000"}'
```

---

## 6. Running the Optional Streamlit UI

```bash
streamlit run streamlit_app_sales.py
```

This provides:

* Chat message history
* Live responses from the agent
* Visible tool actions

---

## 7. Example Queries

The agent handles free-form queries such as:

* “Find me a phone with good battery under 12k.”
* “Do you have Redmi models?”
* “Check inventory for product id 4.”
* “Show laptops with i5, 8GB RAM.”

If criteria are unclear, the agent asks follow-up questions.

---

## 8. Project Structure

```
project/
│
tools/
  ├──product_tools.py       # Structured tools for the LLM
├── sales_agent_new.py     # Agent logic & FastAPI-facing utility
├── streamlit_app_sales.py # Optional UI
└── README.md
```

---

## 9. Customization

You may extend:

* Product attributes
* Filter logic
* Additional tools (compare, sort, detailed specs)
* Database integration
* Agent memory system

The agent is structured for modular growth.

---

## 10. License

MIT License.
Use freely, modify openly.

---

## 11. Author

Project developed by **Ramsha Akhter**.
Feel free to extend, integrate, or build upon this system.

