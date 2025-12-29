import streamlit as st
from huggingface_hub import InferenceClient
from fpdf import FPDF
import io
import re
import graphviz #

# --- 1. SETUP ---
HF_TOKEN = st.secrets["HF_TOKEN"]
client = InferenceClient(token=HF_TOKEN)
TEXT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

st.set_page_config(page_title="Advanced DS Logic Partner", layout="centered")
st.title("🤖📊 Specialized Data Science Assistant")

# --- 2. HELPER FUNCTIONS ---

def create_pdf(text, question):
    """Generates PDF with the technical notes."""
    def clean_text(t):
        return t.encode('latin-1', 'replace').decode('latin-1')
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, clean_text("Data Science Technical Lesson"), ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Helvetica", 'B', 12)
    pdf.multi_cell(0, 10, txt=clean_text(f"Topic: {question}"))
    pdf.ln(5)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 10, txt=clean_text(text))
    return pdf.output()

# --- 3. MAIN APP LOGIC ---

user_question = st.text_input("Ask any Data Science, ML, or DL question:")

if user_question:
    # 🕵️ COMPREHENSIVE KEYWORD GUARDRAIL
    ds_keywords = [
        "data", "model", "learning", "neural", "network", "statistics", "encoding", 
        "pca", "gradient", "regression", "nlp", "language", "text", "lemmatization", 
        "stemming", "token", "vector", "cleaning", "cnn", "rnn", "transformer", 
        "weights", "bias", "variance", "overfitting", "underfitting", "cluster",
        "forest", "tree", "svm", "knn", "bayes", "probability", "epoch", "activation"
    ]
    
    if not any(word in user_question.lower() for word in ds_keywords):
        st.warning("👋 I am a specialized Data Science agent. Please ask about ML, DL, NLP, or Statistics!")
    else:
        # PERSONA LOGIC: Default to Kid, upgrade for M.Tech
        is_mtech = any(word in user_question.lower() for word in ["m.tech", "complex", "advanced", "technical"])
        
        # Color Theme Logic
        if is_mtech:
            node_color = "slategray"
            bg_color = "white"
            font_color = "black"
            style_desc = "professional, architectural, and clean"
        else:
            node_color = "lightblue"
            bg_color = "lightyellow"
            font_color = "darkblue"
            style_desc = "colorful, fun, and simple"

        with st.spinner('Generating logical map...'):
            system_msg = f"""You are a Data Science Expert. 
            Explain to a {'Senior M.Tech student' if is_mtech else '10-year-old child'}.
            {'Provide technical formulas and deep theory.' if is_mtech else 'Use simple stories and no jargon.'}
            
            End your response with a Graphviz DOT code block starting with '---DOT---' and ending with '---END---'.
            Use a {style_desc} style. 
            Set node [style=filled, fillcolor={node_color}, fontcolor={font_color}].
            Set graph [bgcolor={bg_color}]."""

            try:
                response = client.chat_completion(
                    model=TEXT_MODEL,
                    messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_question}],
                    max_tokens=1000
                )
                full_content = response.choices[0].message.content
                parts = full_content.split('---DOT---')
                explanation = parts[0]
                dot_code = parts[1].split('---END---')[0].strip() if len(parts) > 1 else None

                st.markdown("### 💡 Explanation")
                st.write(explanation)

                if dot_code:
                    st.markdown("### 📐 Logical Flow")
                    st.graphviz_chart(dot_code) #

                # PDF Export
                safe_name = re.sub(r'[^\w\s-]', '', user_question).strip().replace(' ', '_')[:30]
                pdf_data = create_pdf(explanation, user_question)
                st.download_button("📥 Download PDF", data=bytes(pdf_data), file_name=f"{safe_name}.pdf")

            except Exception as e:
                st.error(f"Logic Error: {e}")