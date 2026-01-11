import streamlit as st
from rag_engine import answer_question

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Class 11 Sets – AI Tutor",
    layout="centered"
)

# --------------------------------------------------
# Session State Initialization
# --------------------------------------------------
if "explanation" not in st.session_state:
    st.session_state.explanation = ""
    st.session_state.mcqs = []
    st.session_state.confidence_level = ""
    st.session_state.confidence_score = 0

# --------------------------------------------------
# Robust MCQ Parser
# --------------------------------------------------
def parse_mcqs(mcq_text):
    if not mcq_text or not isinstance(mcq_text, str):
        return []

    lines = [l.strip() for l in mcq_text.split("\n") if l.strip()]
    mcqs = []
    current = {"question": "", "options": [], "correct": ""}

    for line in lines:
        if line.lower().startswith("q"):
            if current["question"]:
                mcqs.append(current)
                current = {"question": "", "options": [], "correct": ""}
            current["question"] = line
        elif line.startswith(("A.", "B.", "C.", "D.")):
            current["options"].append(line)
        elif "answer" in line.lower():
            current["correct"] = line.split()[-1].replace(".", "").upper()

    if current["question"]:
        mcqs.append(current)

    return [m for m in mcqs if len(m["options"]) == 4 and m["correct"]]

# --------------------------------------------------
# UI Header
# --------------------------------------------------
st.title("📘 Class 11 Mathematics – Sets AI Tutor")
st.caption("AI Tutor built using RAG + Local LLM (Phi-3)")
st.markdown("---")

# --------------------------------------------------
# Question Input Section
# --------------------------------------------------
with st.container():
    st.subheader("Ask Your Question")
    question = st.text_input(
        "Type your question from the Sets chapter:",
        placeholder="Example: Write set-builder form of {2,3,5,7}"
    )
    ask = st.button("Get Answer", use_container_width=True)

# --------------------------------------------------
# Get Answer Logic
# --------------------------------------------------
if ask:
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            explanation, mcqs_text, confidence_level, confidence_score = answer_question(question)

        st.session_state.explanation = explanation or "No explanation found."
        st.session_state.mcqs = parse_mcqs(mcqs_text)
        st.session_state.confidence_level = confidence_level
        st.session_state.confidence_score = confidence_score

# --------------------------------------------------
# Output Section
# --------------------------------------------------
if st.session_state.explanation:

    st.markdown("---")
    st.subheader("📘 Explanation")
    st.info(st.session_state.explanation)

    st.subheader("🔍 Confidence Level")
    st.progress(st.session_state.confidence_score / 100)
    st.write(f"Confidence: {st.session_state.confidence_level} ({st.session_state.confidence_score}%)")

    st.markdown("---")
    st.subheader("📝 Practice MCQs")

    if not st.session_state.mcqs:
        st.info("No MCQs generated for this question.")
    else:
        for idx, mcq in enumerate(st.session_state.mcqs):
            with st.expander(f"Question {idx+1}", expanded=True):
                st.markdown(f"**{mcq['question']}**")

                selected = st.radio(
                    "Choose an option:",
                    mcq["options"],
                    key=f"mcq_option_{idx}"
                )

                if st.button(f"Submit Answer {idx+1}", key=f"submit_{idx}"):
                    chosen = selected.split(".")[0].upper()

                    if chosen == mcq["correct"]:
                        st.success("✅ Correct answer!")
                    else:
                        st.error(f"❌ Incorrect. Correct answer is **{mcq['correct']}**")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Built by Rohan & Amit using RAG, FAISS, and Phi-3 | Educational AI Project")
