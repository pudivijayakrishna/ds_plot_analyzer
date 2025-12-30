import streamlit as st
import asyncio
import json
import re
import html
import os
import atexit
import logging
import markdown
import uuid
from typing import Optional
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
import nest_asyncio
import graphviz
from playwright.async_api import async_playwright, Browser, Playwright

# --- 0. PRODUCTION SETUP ---
PROMPT_VERSION = "22.0.0-Platinum"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

nest_asyncio.apply()

st.set_page_config(page_title="Intelligent DS Tutor", layout="centered", page_icon="🎓")

# --- 1. ASYNC CORE ---
_PLAYWRIGHT: Optional[Playwright] = None
_BROWSER: Optional[Browser] = None

def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

@st.cache_resource
def get_browser_instance():
    async def init_browser():
        global _PLAYWRIGHT, _BROWSER
        if _BROWSER is None:
            _PLAYWRIGHT = await async_playwright().start()
            _BROWSER = await _PLAYWRIGHT.chromium.launch()
        return _BROWSER
    return run_async(init_browser())

@atexit.register
def cleanup_resources():
    if _BROWSER: run_async(_BROWSER.close())
    if _PLAYWRIGHT: run_async(_PLAYWRIGHT.stop())
    logger.info("♻️ Resources cleaned up.")

# --- 2. CONFIGURATION ---
class AppConfig:
    HF_TOKEN = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

    # Minimal fallback to prevent crash, but we prefer showing error messages now
    DEFAULT_DOT = """
    digraph G {
        rankdir=LR; bgcolor="white";
        node [shape=box, style=filled, fillcolor="white", fontname="Helvetica"];
        Error [label="System Offline", color="red"];
    }
    """

    SYSTEM_PROMPT = f"""
    You are an academic teaching assistant STRICTLY LIMITED to:
    Data Science, Machine Learning, Deep Learning, NLP, and AI.

    --------------------------------------------------
    EDUCATION LEVEL RULE (MANDATORY)
    --------------------------------------------------
    The system will specify EDUCATION_LEVEL as one of:
    - KID (default)
    - BTECH
    - MTECH

    You MUST follow it exactly.

    --------------------
    MTECH MODE (ADVANCED)
    --------------------
    You MUST include ALL of the following sections in order:
    1. **Problem Motivation**: Real-world trigger.
    2. **Mathematical Formulation**: Full derivations.
    3. **Explanation of Symbols**: Define every variable.
    4. **Algorithm / Working Steps**: Step-by-step logic.
    5. **Real-World Applications**: At least 2 concrete cases.
    6. **Why This Method Is Used**: Comparison & Trade-offs.
    7. **When NOT to Use**: Explicit failure cases.

    --------------------------------------------------
    MATH RULE (MANDATORY)
    --------------------------------------------------
    - **Block Equations:** Must be wrapped in DOUBLE dollar signs: $$ y = mx + b $$
    - **Inline Variables:** Must be wrapped in SINGLE dollar signs: "where $m$ is slope"
    - **FORBIDDEN:** Do NOT use parentheses like ( t ) for math. Use $t$.

    --------------------------------------------------
    DIAGRAM RULE (MANDATORY)
    --------------------------------------------------
    You MUST ALWAYS include a diagram.
    - If the concept is an architecture, use standard Graphviz.
    - If abstract, generate a CENTRAL-NODE RADIAL MIND MAP.
    - dot_code must NEVER be null.

    --------------------------------------------------
    OUTPUT FORMAT (STRICT JSON)
    --------------------------------------------------
    Return ONLY valid JSON:
    {{
      "explanation": "Markdown text...",
      "dot_code": "Valid Graphviz DOT code"
    }}
    """

if not AppConfig.HF_TOKEN:
    st.error("🚨 Critical: `HF_TOKEN` missing.")
    st.stop()

# --- 3. POLICY ENGINE ---
def classify_query(query: str):
    q = query.lower()
    
    allowed_domains = [
        "data", "science", "machine learning", "ml", "deep learning", "dl", 
        "nlp", "artificial intelligence", "ai", "neural network", 
        "regression", "classification", "clustering", "decision tree", 
        "random forest", "svm", "bayes", "reinforcement learning",
        "computer vision", "transformer", "lstm", "rnn", "cnn", 
        "gradient descent", "algorithm", "model", "python", "tf-idf", "word2vec", "gpt", "bert", "gru", "ner"
    ]
    is_allowed = any(k in q for k in allowed_domains)
    
    if any(k in q for k in ["m.tech", "mtech", "masters", "derivation", "proof"]):
        level = "MTECH"
    elif any(k in q for k in ["b.tech", "btech", "undergraduate"]):
        level = "BTECH"
    else:
        level = "KID" 

    return {"allowed": is_allowed, "level": level}

# --- 4. DATA CONTRACTS ---
class AcademicResponse(BaseModel):
    explanation: str
    dot_code: Optional[str] = None
    is_degraded: bool = False
    error_msg: Optional[str] = None

# --- 5. LLM SERVICE ---
@st.cache_resource
def get_llm_service():
    return LLMService()

class LLMService:
    def __init__(self):
        self.client = InferenceClient(token=AppConfig.HF_TOKEN)

    def _parse_safely(self, raw_text: str) -> dict:
        clean_text = raw_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text.replace("```json", "").replace("```", "")
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError:
            exp_match = re.search(r'"explanation":\s*"(.*?)(?<!\\)"', clean_text, re.DOTALL)
            dot_match = re.search(r'"dot_code":\s*"(.*?)(?<!\\)"', clean_text, re.DOTALL)
            if exp_match:
                expl = exp_match.group(1).replace('\\"', '"').replace('\\n', '\n')
                dot = None
                if dot_match:
                    dot = dot_match.group(1).replace('\\"', '"').replace('\\n', '\n')
                return {"explanation": expl, "dot_code": dot}
            raise ValueError("Invalid JSON")

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=3))
    def _fetch_from_llm(self, user_prompt: str, level_instruction: str):
        response = self.client.chat_completion(
            model=AppConfig.MODEL_ID,
            messages=[
                {"role": "system", "content": AppConfig.SYSTEM_PROMPT},
                {"role": "system", "content": level_instruction},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=4000,
            temperature=0.3, 
        )
        return response.choices[0].message.content

    def query(self, user_prompt: str, level: str) -> AcademicResponse:
        level_instruction = f"EDUCATION_LEVEL = {level}"
        
        try:
            raw = self._fetch_from_llm(user_prompt, level_instruction)
            data = self._parse_safely(raw)
            resp = AcademicResponse(**data)
            
            # Basic Validation
            if not resp.dot_code or str(resp.dot_code).lower() == "null":
                raise ValueError("Diagram missing.")
            
            logger.info(f"Query Success | Lvl: {level}")
            return resp

        except Exception as e:
            error_str = str(e).lower()
            # CRITICAL FIX: Detect 402 Payment Error from Hugging Face
            if "402" in error_str or "payment" in error_str:
                return AcademicResponse(
                    explanation="**⚠️ API Limit Reached:** The AI service is currently unavailable due to provider limits (Error 402). Please check your Hugging Face token quota.",
                    dot_code=AppConfig.DEFAULT_DOT,
                    is_degraded=True,
                    error_msg="Payment Required"
                )
            
            logger.warning(f"Retry failed: {e}. Switching to Recovery Mode.")
            return AcademicResponse(
                explanation=f"**System Note:** The AI encountered an error generating the response. Please try simplifying your query.\n\n*Error details: {e}*",
                dot_code=AppConfig.DEFAULT_DOT,
                is_degraded=True
            )

# --- 6. ENGINES ---
class TextProcessor:
    @staticmethod
    def normalize_math(text: str) -> str:
        """
        Converts ( x ) -> $x$ and fixes delimiters.
        """
        text = re.sub(r'\(\s*([a-zA-Z0-9_+\-\\^]+|\\[a-zA-Z]+.*?)\s*\)', r'$\1$', text)
        text = text.replace(r'\[', '$$').replace(r'\]', '$$')
        text = text.replace(r'\(', '$').replace(r'\)', '$')
        return text

    @staticmethod
    def clean_leakage(text: str) -> str:
        """
        Removes ANY code block that resembles Graphviz to prevent UI leakage.
        """
        parts = text.split('```')
        cleaned_parts = []
        for i, part in enumerate(parts):
            if i % 2 == 1: # Inside a code block
                # If it smells like a graph, kill it.
                if any(kw in part.lower() for kw in ['digraph', 'graph {', 'rankdir=', '[shape=', '->']):
                    continue 
                else:
                    cleaned_parts.append('```' + part + '```')
            else:
                cleaned_parts.append(part)
        
        text = "".join(cleaned_parts)
        # Regex cleanup for unfenced code
        text = re.sub(r'(?:digraph|graph)\s+\w*\s*\{[^}]*\}', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Cleanup dangling headers
        text = re.sub(r'(?:###|##|\*\*)\s*(?:Diagram|Visual Map|Visualization).*?', '', text, flags=re.IGNORECASE)
        return text

class DiagramEngine:
    @staticmethod
    def repair_dot_code(dot_code: str) -> str:
        code = dot_code.strip()
        if not code.endswith("}"): code += "}"
        
        # 1. Strip LaTeX from Diagram (Graphviz cannot render it)
        code = code.replace("$", "").replace("\\", "")

        # 2. Force White Background (Fixes "Black Box" Issue)
        # 3. Optimize Layout (Fixes "Alignment" Issue)
        layout_config = ' bgcolor="white"; graph [rankdir=LR, nodesep=0.6, ranksep=0.7, splines=curved, overlap=false]; node [fontname="Helvetica", shape=box, style="filled", fillcolor="white", color="black"]; edge [fontname="Helvetica"]; '
        
        if "bgcolor" not in code:
            code = code.replace("{", '{' + layout_config, 1)
        if "rankdir" not in code:
             code = code.replace("{", '{ rankdir=LR; ', 1)
        if "->" in code and code.lower().startswith("graph"):
            code = re.sub(r"^graph", "digraph", code, flags=re.IGNORECASE)
        return code

    @staticmethod
    @st.cache_data
    def render_svg(dot_code: str) -> Optional[str]:
        if not dot_code: return None
        clean_code = DiagramEngine.repair_dot_code(dot_code)
        try:
            src = graphviz.Source(clean_code)
            raw_svg = src.pipe(format="svg").decode("utf-8")
            return re.sub(r"<script.*?>.*?</script>", "", raw_svg, flags=re.DOTALL|re.IGNORECASE)
        except Exception:
            return None

class PDFEngine:
    @staticmethod
    def create_html(title: str, body: str, svg: Optional[str]) -> str:
        safe_title = html.escape(title)
        
        # --- UUID MATH PROTECTION (Markdown-Safe) ---
        # Uses Alphanumeric UUIDs (no underscores) to bypass Markdown parsing
        math_map = {}
        def protect_math_block(match):
            # "MATHBLOCK" + hex uuid + "END" is safe from Markdown
            uid = f"MATHBLOCK{uuid.uuid4().hex}END"
            math_map[uid] = match.group(0)
            return uid

        # Protect $$...$$ then $...$
        body = re.sub(r'\$\$.*?\$\$', protect_math_block, body, flags=re.DOTALL)
        body = re.sub(r'(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)', protect_math_block, body)

        # Convert to HTML
        html_body = markdown.markdown(body, extensions=['extra', 'tables'])

        # --- RESTORE MATH ---
        for uid, latex_code in math_map.items():
            # Replace < with \lt to prevent browser tag confusion
            safe_latex = latex_code.replace("<", " \\lt ").replace(">", " \\gt ")
            html_body = html_body.replace(uid, safe_latex)
        
        return f"""
        <!DOCTYPE html><html><head><meta charset="utf-8">
        <script>
        MathJax = {{
            tex: {{ inlineMath: [['$', '$'], ['\\\\(', '\\\\)']], displayMath: [['$$', '$$']] }},
            startup: {{ typeset: true }}
        }};
        </script>
        <script async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.min.js"></script>
        <style>
            @page {{ size: A4; margin: 2.5cm; }}
            body {{ font-family: 'Times New Roman', serif; line-height: 1.6; font-size: 12pt; }}
            h1 {{ text-align: center; color: #004e98; border-bottom: 2px solid #004e98; padding-bottom: 10px; }}
            h2, h3 {{ color: #333; margin-top: 20px; }}
            .diagram-container {{ text-align: center; margin: 20px 0; }}
            svg {{ max-width: 100%; height: auto; }}
            mjx-container[display="true"] {{ display: block; text-align: center; margin: 1em 0; }}
            table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; }}
        </style>
        </head><body>
            <h1>{safe_title}</h1>
            <div class="diagram-container">{svg if svg else ""}</div>
            <div>{html_body}</div>
        </body></html>
        """

    @staticmethod
    async def generate_pdf(html_content: str) -> bytes:
        browser = get_browser_instance()
        page = await browser.new_page()
        await page.set_content(html_content)
        try:
            await page.wait_for_load_state("networkidle")
            await page.evaluate("MathJax.typesetPromise()")
            await page.wait_for_timeout(2000) 
        except Exception:
            await page.wait_for_timeout(3000)
        pdf = await page.pdf(format="A4", print_background=True)
        await page.close()
        return pdf

# --- 7. MAIN UI ---
def main():
    st.title("🤖 Intelligent DS Tutor")
    st.caption("Auto-adapts: 🧒 Kid | 🎓 B.Tech | 🔬 M.Tech")
    
    if "result" not in st.session_state:
        st.session_state.result = None
        st.session_state.svg = None

    query = st.text_input("Ask a concept:", placeholder="e.g., 'GRU M.Tech' or 'Word2Vec'")

    if query and (st.session_state.get("last_query") != query):
        
        policy = classify_query(query)
        
        if not policy["allowed"]:
            st.warning("⚠️ This assistant is designed exclusively for Data Science, ML, AI, and NLP topics.")
            return

        with st.spinner(f"🧠 Synthesizing {policy['level']} response..."):
            try:
                llm = get_llm_service()
                resp = llm.query(query, policy['level'])
                
                st.session_state.result = resp
                st.session_state.svg = DiagramEngine.render_svg(resp.dot_code)
                st.session_state.last_query = query
                st.session_state.current_level = policy['level']
                
            except Exception as e:
                st.error("Service unavailable. Please try again.")
                logger.error(f"UI Error: {e}")

    if st.session_state.result:
        data = st.session_state.result
        lvl = st.session_state.get("current_level", "KID")
        
        color = "green" if lvl == "KID" else "orange" if lvl == "BTECH" else "red"
        st.markdown(f":{color}[**Mode: {lvl}**]")

        if data.is_degraded:
            if data.error_msg == "Payment Required":
                st.error(data.explanation) # Show specific payment error
            else:
                st.warning("⚠️ Formatting degraded: The model provided a best-effort response.")

        clean_text = TextProcessor.clean_leakage(data.explanation)
        clean_text = TextProcessor.normalize_math(clean_text)
        
        # Safety Cap
        if len(clean_text) > 8000:
            clean_text = clean_text[:8000] + "...\n\n*(Response truncated for performance)*"

        chunks = re.split(r'(\$\$.*?\$\$)', clean_text, flags=re.DOTALL)
        for chunk in chunks:
            if chunk.startswith('$$'):
                st.latex(chunk.strip('$'))
            elif chunk.strip():
                st.markdown(chunk)

        st.markdown("---")
        st.markdown("### 🗺️ Visual Map")
        
        if st.session_state.svg: 
            # FIXED: Updated to modern Streamlit parameter
            st.image(st.session_state.svg, use_container_width=True)
        elif data.dot_code and not data.is_degraded:
            try:
                fixed_code = DiagramEngine.repair_dot_code(data.dot_code)
                st.graphviz_chart(fixed_code)
            except Exception:
                st.error("Diagram structure invalid.")

        if st.button("📄 Download PDF"):
            with st.spinner("Compiling PDF..."):
                html_pl = PDFEngine.create_html(st.session_state.last_query, clean_text, st.session_state.svg)
                try:
                    pdf = run_async(PDFEngine.generate_pdf(html_pl))
                    st.download_button("Download", pdf, file_name=f"{lvl}_report.pdf", mime="application/pdf")
                except Exception:
                    st.error("PDF generation failed.")

if __name__ == "__main__":
    main()