# streamlit_app.py
import streamlit as st
import traceback
import uuid

st.set_page_config(page_title="Sales Agent (LangGraph)", layout="wide")
st.title("🛍️ AI Sales Agent")
st.caption("Chat-based product recommender with checkout simulation")

# ----------------- import backend chat or answer -----------------
chat_fn = None
answer_fn = None
try:
    # prefer a chat(memory, user_input) function if present
    from sales_agent_new import chat as imported_chat
    chat_fn = imported_chat
    st.success("✅ sales_agent_new.chat loaded")
except Exception:
    try:
        # fallback to answer(...) if chat isn't present
        from sales_agent_new import answer as imported_answer
        answer_fn = imported_answer
        st.success("✅ sales_agent_new.answer loaded (will wrap memory -> prompt)")
    except Exception:
        st.error("❌ Failed to import sales_agent_new.chat or sales_agent_new.answer")
        st.code(traceback.format_exc())

# ----------------- Session state -----------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    # structured for display: {"role":"user"/"assistant","content": "..."}
    st.session_state.messages = []
if "memory" not in st.session_state:
    # memory format expected by your chat() implementation: list of "User: ...\nAgent: ..."
    st.session_state.memory = []

# Clear button
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.memory = []
        st.experimental_rerun()

# render history in chat UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# collect user input
user_input = st.chat_input("Type something like 'I want a smartwatch'...")

if user_input:
    # append and show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call backend via chat_fn if present (preferred)
    try:
        if chat_fn is not None:
            # expected signature: chat(memory=None, user_input=None) -> (response, updated_memory)
            response, updated_memory = chat_fn(memory=st.session_state.memory, user_input=user_input)
        elif answer_fn is not None:
            # build a prompt from memory and user_input and call answer()
            # memory entries are strings like "User: ...\nAgent: ..."
            previous = "\n".join(st.session_state.memory)
            # create the same template your chat() used earlier:
            prompt = f"Previous conversation: {previous}\nlatest query: {user_input}"
            # call answer. Try both common signatures: answer(user_input) or answer(thread_id, user_input)
            try:
                # try simple call first
                answer_text = answer_fn(prompt)
            except TypeError:
                # try thread-aware signature
                try:
                    answer_text = answer_fn(st.session_state.thread_id, prompt)
                except TypeError:
                    # final attempt: pass only user_input
                    answer_text = answer_fn(user_input)
            response = answer_text
            # update memory the same way chat() would
            updated_memory = st.session_state.memory + [f"User: {user_input}\nAgent: {response}"]
        else:
            response = "Error: no backend chat/answer function available."
            updated_memory = st.session_state.memory

    except Exception as e:
        response = f"Error while calling backend: {e}"
        updated_memory = st.session_state.memory

    # ensure memory is a list of strings and trim to last 5 exchanges
    if not isinstance(updated_memory, list):
        updated_memory = st.session_state.memory
    # if memory contains exchanges, ensure they are strings like "User:...\nAgent:..."
    try:
        updated_memory = [str(x) for x in updated_memory][-5:]
    except Exception:
        updated_memory = st.session_state.memory

    # store results
    st.session_state.memory = updated_memory
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

    # ---- DEBUG: show what was sent to backend ----
    with st.expander("Debug — prompt/history sent to backend (for troubleshooting)"):
        st.write("thread_id:", st.session_state.thread_id)
        st.write("memory (last entries):", st.session_state.memory)
        # Build the exact prompt string we sent (if using answer_fn fallback)
        if answer_fn is not None and chat_fn is None:
            previous = "\n".join(st.session_state.memory)
            st.write("prompt string sent to answer():")
            st.code(f"Previous conversation: {previous}\nlatest query: {user_input}")
