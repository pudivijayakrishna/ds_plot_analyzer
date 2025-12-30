import streamlit as st
import streamlit.components.v1 as components
import asyncio
import json
import re
import html
import os
import atexit
import logging
import markdown
import uuid
import warnings
from typing import Optional
from pydantic import BaseModel
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
import nest_asyncio
import graphviz
from playwright.async_api import async_playwright, Browser, Playwright

# --- 0. PRODUCTION SETUP ---
PROMPT_VERSION = "46.0.0-Platinum"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- CONFIGURATION: Silence Warnings ---
os.environ["STREAMLIT_hush"] = "1"
warnings.filterwarnings("ignore")

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
            _BROWSER = await _PLAYWRIGHT.chromium.launch(
                args=[
                    "--no-sandbox", 
                    "--disable-setuid-sandbox", 
                    "--disable-dev-shm-usage",
                    "--single-process", 
                    "--disable-gpu"
                ]
            )
        return _BROWSER
    return run_async(init_browser())

@atexit.register
def cleanup_resources():
    if _BROWSER: run_async(_BROWSER.close())
    if _PLAYWRIGHT: run_async(_PLAYWRIGHT.stop())
    logger.info("♻️ Resources cleaned up.")

# --- 2. CONFIGURATION ---
class AppConfig:
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    MODEL_LIST = [
        "gemma-3-27b-it",
        "gemini-2.5-flash-lite", 
        "gemini-1.5-flash"
    ]

    DEFAULT_DOT = """
    digraph G {
        rankdir=LR; bgcolor="white";
        node [shape=box, style=filled, fillcolor="white", fontname="Helvetica"];
        Error [label="Service Unavailable", color="red"];
    }
    """

    SYSTEM_PROMPT = f"""
    You are an academic teaching assistant STRICTLY LIMITED to:
    Data Science, Machine Learning, Deep Learning, NLP, and AI.
    
    CRITICAL FORMATTING RULE (MANDATORY):
    - EVERY block equation MUST be on its own independent line.
    - NEVER place text, headings, or markdown on the same line as a block equation ($$...$$).
    - Use this format ONLY:
    
    Text explanation...
    
    $$
    equation here
    $$
    
    Next text block...
    
    - Output math variables in standard LaTeX format.
    
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

if not AppConfig.GEMINI_KEY:
    st.error("🚨 Critical: `GEMINI_API_KEY` missing. Please get one from Google AI Studio.")
    st.stop()

genai.configure(api_key=AppConfig.GEMINI_KEY)

# --- 3. POLICY ENGINE ---
def classify_query(query: str):
    q = query.lower()
    allowed_domains = ["data", "science", "machine", "learning", "ml", "deep", "dl", "nlp", "ai", "neural", "regression", "classification", "clustering", "tree", "forest", "svm", "bayes", "reinforcement", "vision", "transformer", "lstm", "rnn", "cnn", "gradient", "algorithm", "model", "python", "word2vec", "gpt", "bert", "gru", "ner", "correlation", "matrix", "optimization"]
    is_allowed = any(k in q for k in allowed_domains)
    level = "MTECH" if any(k in q for k in ["m.tech", "mtech", "masters"]) else "BTECH" if any(k in q for k in ["b.tech", "btech"]) else "KID"
    return {"allowed": is_allowed, "level": level}

# --- 4. DATA CONTRACTS ---
class AcademicResponse(BaseModel):
    explanation: str
    dot_code: Optional[str] = None
    is_degraded: bool = False
    error_type: Optional[str] = None

# --- 5. LLM SERVICE ---
@st.cache_resource
def get_llm_service():
    return LLMService()

class LLMService:
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
    def query(self, user_prompt: str, level: str) -> AcademicResponse:
        level_instruction = f"EDUCATION_LEVEL = {level}"
        full_prompt = f"{AppConfig.SYSTEM_PROMPT}\n\n{level_instruction}\n\nUSER QUERY: {user_prompt}"
        
        last_error = None
        for model_name in AppConfig.MODEL_LIST:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(full_prompt)
                data = self._parse_safely(response.text)
                resp = AcademicResponse(**data)
                
                if not resp.dot_code or str(resp.dot_code).lower() == "null":
                    raise ValueError("Diagram missing.")
                
                return resp
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
                last_error = e
                continue

        return AcademicResponse(
            explanation=f"**System Note:** All AI models are currently unavailable. Error: {last_error}",
            dot_code=AppConfig.DEFAULT_DOT,
            is_degraded=True
        )

# --- 6. TEXT PROCESSOR ---
class TextProcessor:
    @staticmethod
    def clean_leakage(text: str) -> str:
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(
            r'(digraph|graph)\s+\w*\s*\{.*?\}',
            '',
            text,
            flags=re.DOTALL | re.IGNORECASE
        )
        # Force math blocks to be isolated
        text = re.sub(
            r'(\$\$.*?\$\$)',
            r'\n\1\n',
            text,
            flags=re.DOTALL
        )
        return text.strip()

# --- 7. RENDERING LOGIC ---
def render_content(text: str):
    """
    Streamlit-safe renderer:
    - Math ONLY via st.latex
    - Text ONLY via st.markdown
    """
    blocks = re.split(r'(\$\$.*?\$\$)', text, flags=re.DOTALL)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        if block.startswith('$$') and block.endswith('$$'):
            st.latex(block[2:-2])
        else:
            st.markdown(block)

class DiagramEngine:
    @staticmethod
    def repair_dot_code(dot_code: str) -> str:
        code = dot_code.strip()
        if not code.endswith("}"): code += "}"
        code = code.replace("$", "").replace("\\", "")
        layout_config = ' bgcolor="white"; graph [rankdir=LR, nodesep=0.6, ranksep=0.7, splines=true, overlap=false]; node [fontname="Helvetica", shape=box, style="filled", fillcolor="white", color="black"]; edge [fontname="Helvetica"]; '
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
            
            # Ensure SVG scales properly
            raw_svg = raw_svg.replace("<svg ", '<svg width="100%" height="auto" ')
            
            return re.sub(r"<script.*?>.*?</script>", "", raw_svg, flags=re.DOTALL|re.IGNORECASE)
        except Exception:
            return None

# --- 8. PDF ENGINE (SAFE STATEFUL HTMLIFY) ---
def simple_htmlify(text: str) -> str:
    """
    Safe text → HTML conversion.
    - Preserves $$...$$ for MathJax (NO ESCAPING)
    - Escapes normal text only
    - Handles lists statefully to create valid <ul> structures
    """
    lines = text.split("\n")
    html_lines = []
    in_math_block = False
    in_list = False

    for line in lines:
        stripped = line.strip()

        # 1. Detect pure delimiter line ($$)
        if stripped == "$$":
            in_math_block = not in_math_block
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append("$$")
            continue
        
        # 2. Inside math block? Pass raw LaTeX
        if in_math_block:
            html_lines.append(line)
            continue
            
        # 3. Single-line block equation? Pass raw LaTeX
        if stripped.startswith("$$") and stripped.endswith("$$"):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(stripped)
            continue

        # 4. List Handling (Stateful)
        if stripped.startswith("- "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{html.escape(stripped[2:])}</li>")
            continue
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False

        # 5. Normal Text Formatting
        if stripped.startswith("###"):
            html_lines.append(f"<h3>{html.escape(stripped[3:].strip())}</h3>")
        elif stripped.startswith("##"):
            html_lines.append(f"<h2>{html.escape(stripped[2:].strip())}</h2>")
        elif stripped.startswith("#"):
            html_lines.append(f"<h1>{html.escape(stripped[1:].strip())}</h1>")
        elif not stripped:
            # POLISH: Skip empty lines to reduce noise
            continue
        else:
            html_lines.append(f"<p>{html.escape(stripped)}</p>")

    if in_list:
        html_lines.append("</ul>")

    return "\n".join(html_lines)

class PDFEngine:
    @staticmethod
    def create_html(title: str, body: str, svg: Optional[str]) -> str:
        safe_title = html.escape(title)
        html_body = simple_htmlify(body)

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="utf-8">
        
        <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>

        
        <script>
        window.MathJax = {{
          tex: {{
            inlineMath: [['$', '$']],
            displayMath: [['$$', '$$']],
            processEscapes: true
          }},
          startup: {{
            typeset: true
          }}
        }};
        </script>

        <style>
        body {{
          font-family: Helvetica, Arial, sans-serif;
          line-height: 1.6;
          max-width: 900px;
          margin: auto;
          padding: 30px;
          color: #333;
        }}
        h1 {{
          text-align: center;
          border-bottom: 2px solid #004e98;
          padding-bottom: 10px;
        }}
        .diagram {{
          text-align: center;
          margin: 30px 0;
          display: flex;
          justify-content: center;
        }}
        img {{ max-width: 100%; height: auto; }}
        </style>
        </head>

        <body>
            <h1>{safe_title}</h1>
            <div class="diagram">{svg if svg else ""}</div>
            {html_body}
        </body>
        </html>
        """

    @staticmethod
    async def generate_pdf(html_content: str) -> tuple[bytes, str]:
        try:
            browser = get_browser_instance()
            page = await browser.new_page()
            await page.set_content(html_content)
            await page.wait_for_load_state("networkidle")
            
            # POLISH: Safe MathJax Wait
            await page.evaluate("""
                if (window.MathJax && MathJax.typesetPromise) {
                    await MathJax.typesetPromise();
                }
            """)
            await page.wait_for_timeout(3000) 
            
            pdf = await page.pdf(format="A4", print_background=True)
            await page.close()
            return pdf, "application/pdf"
        except Exception as e:
            return html_content.encode('utf-8'), "text/html"

# --- 10. MAIN UI ---
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

    if st.session_state.result:
        data = st.session_state.result
        lvl = st.session_state.get("current_level", "KID")
        
        color = "green" if lvl == "KID" else "orange" if lvl == "BTECH" else "red"
        st.markdown(f":{color}[**Mode: {lvl}**]")

        # --- PIPELINE ---
        clean_text = TextProcessor.clean_leakage(data.explanation)
        
        # --- UI RENDER ---
        render_content(clean_text)

        st.markdown("---")
        st.markdown("### 🗺️ Visual Map")
        
        # --- FIX: Safe SVG Isolation ---
        if st.session_state.svg: 
            components.html(
                f"""
                <div style="width:100%; text-align:center;">
                    {st.session_state.svg}
                </div>
                """,
                height=500,
                scrolling=True
            )
        elif data.dot_code and not data.is_degraded:
            st.graphviz_chart(DiagramEngine.repair_dot_code(data.dot_code))
        else:
            st.info("Diagram unavailable for this response.")

        if st.button("📄 Download Report"):
            with st.spinner("Generating File..."):
                html_pl = PDFEngine.create_html(st.session_state.last_query, clean_text, st.session_state.svg)
                file_data, mime_type = run_async(PDFEngine.generate_pdf(html_pl))
                
                ext = "pdf" if mime_type == "application/pdf" else "html"
                label = "Download PDF" if ext == "pdf" else "Download HTML (Offline Viewable)"
                
                st.download_button(label, file_data, file_name=f"{lvl}_report.{ext}", mime=mime_type)

if __name__ == "__main__":
    main()