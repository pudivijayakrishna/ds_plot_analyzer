
# 🤖 Intelligent DS Tutor

An advanced **AI-powered teaching assistant** built for **Data Science, Machine Learning, and NLP** education. Using **gemma-3-27b-it**.
The system dynamically adapts explanations across education levels (Kid, B.Tech, M.Tech) and produces **research-quality, mathematically correct** outputs with robust visualization and export support.

🔗 **Live Application:**  
👉 https://dsplotanalyzer-aa3ukbsyygiutul5whz4by.streamlit.app/

---

## ✨ Key Highlights

### 🎯 Adaptive Learning Modes
- **🧒 Kid Mode** – Intuitive explanations using analogies and storytelling.
- **🎓 B.Tech Mode** – Conceptual clarity with standard algorithms and intuition.
- **🔬 M.Tech Mode** – Full mathematical formulations, derivations, symbol definitions, and research-level discussion.

### 📐 Reliable Mathematical Rendering
- End-to-end **LaTeX-safe rendering pipeline**.
- Ensures consistency across:
  - Streamlit UI (KaTeX)
  - Exported HTML (MathJax)
  - Exported PDF (Headless Chromium)
- Eliminates common AI issues such as:
  - Broken formulas
  - Escaped LaTeX symbols
  - Inline/block math corruption

### 📊 Visual Learning
- Automatically generates **Graphviz diagrams** for every concept.
- SVGs are safely isolated and rendered responsively.
- Supports both architectural diagrams and abstract concept maps.

### 📄 Production-Grade Exports
- **PDF Export**
  - Uses Playwright (headless Chromium)
  - Waits for MathJax typesetting before rendering
- **HTML Export**
  - Fully offline-viewable
  - Preserves math, text, and SVG diagrams without external dependencies

---

## 🏗️ Architecture & Tech Stack

The application follows a **content-sanitized micro-frontend architecture** with strict separation between text, math, and visuals.

### Core Stack
- **Frontend:** Streamlit (Python)
- **LLM Engine:** Google Gemini  
  (`gemma-3-27b-it`, `gemini-1.5-flash`)
- **Math Rendering:**
  - UI: Streamlit native `st.latex`
  - HTML/PDF: MathJax v3 (async loading)
- **Visualization:** Graphviz (DOT → SVG)
- **Export Engine:** Playwright (Headless Chromium)

---

## ⚙️ Installation & Local Setup

### 1️⃣ Prerequisites
- Python **3.9+**
- Google AI Studio API Key (free tier supported)

---

### 2️⃣ Clone the Repository

git clone https://github.com/pudivijayakrishna/ds_plot_analyzer.git
cd intelligent-ds-tutor


---

### 3️⃣ Install Dependencies

pip install -r requirements.txt
python -m playwright install chromium

---

### 4️⃣ Configure API Key

Create the following file:

toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "YOUR_API_KEY_HERE"

> ⚠️ Never commit this file to GitHub.

---

### 5️⃣ Run the Application


streamlit run app.py

---

## 🧠 How It Works — The Math Rendering Pipeline

One of the core engineering challenges solved in this project is **reliable rendering of AI-generated mathematics**.

Large Language Models often output mixed Markdown + LaTeX that breaks traditional renderers.
This project implements a **strict isolation pipeline** to solve that.

### Pipeline Overview

1. **Sanitization**

   * Removes leaked code blocks (Graphviz, Markdown fences)
   * Forces isolation of `$$ ... $$` math blocks

2. **Logic Separation**

   * **Streamlit UI**

     * Math → `st.latex()`
     * Text → `st.markdown()`
   * **HTML Export**

     * Custom state-machine (`simple_htmlify`)
     * Escapes text only
     * Preserves raw LaTeX for MathJax

3. **Export Handling**

   * **PDF**

     * Loads HTML in Playwright
     * Awaits `MathJax.typesetPromise()`
     * Renders pixel-perfect output
   * **HTML**

     * Fully offline compatible
     * Server-rendered SVG diagrams embedded directly

This guarantees **identical mathematical output** across UI, HTML, and PDF.

---

## 📁 Project Structure


├── app.py                  # Main application logic
├── requirements.txt        # Python dependencies
├── packages.txt            # System deps (Graphviz, Chromium)
└── .streamlit/
    └── secrets.toml        # API keys (ignored by Git)


---

## 🤝 Contributions

Contributions are welcome!

If you encounter:

* A query that breaks math rendering
* A visualization edge case
* Performance improvements

Please open an issue with:

* The query
* Selected mode (Kid / B.Tech / M.Tech)
* Screenshot or exported output

### Contribution Flow

1. Fork the repository
2. Create a feature branch

   
   git checkout -b feature/your-feature-name
   
3. Commit your changes
4. Push and open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License**.
See the `LICENSE` file for details.

---

### ⭐ If you find this project useful, consider starring the repository!


