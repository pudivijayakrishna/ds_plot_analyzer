# 🤖 Data Science Logic & Flow Assistant

A specialized, multimodal AI application built for **M.Tech Data Science** students and educators. This tool provides dual-persona explanations (Kid-friendly vs. Academic) and generates dynamic logical flowcharts using Graphviz.

**🚀 Live App:** [PASTE_YOUR_STREAMLIT_URL_HERE]

---

## 🌟 Key Features

* **Dual-Persona Intelligence:** Automatically detects the user's intent. It provides simple analogies for beginners and rigorous technical deep-dives for M.Tech students.
* **Dynamic Logic Mapping:** Replaces static images with Graphviz-generated flowcharts, visualizing architectures like Neural Networks or NLP pipelines.
* **Strict Domain Guardrails:** Implementation of an expert-system filter that ensures the AI stays focused strictly on Data Science, ML, and DL topics.
* **Professional PDF Export:** Generates high-quality technical notes with sanitized text and original user prompts as filenames.

---

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **LLM Brain:** `Qwen/Qwen2.5-7B-Instruct` via Hugging Face Inference API
* **Diagram Engine:** [Graphviz](https://graphviz.org/)
* **Document Generation:** `fpdf2`
* **Deployment:** Streamlit Cloud with secure Secret Management

---

## 📐 Example Usage

1.  **Kid Mode:** Ask *"What is a Neural Network?"* for a story about "tiny friends" passing notes.
2.  **M.Tech Mode:** Ask *"What is a Neural Network M.Tech?"* for an architectural diagram and mathematical explanation of backpropagation.
3.  **Guardrail Test:** Ask *"How do I bake a cake?"* to see the specialized refusal message.

---

## ⚙️ Local Setup

1. Clone the repo:
   ```bash
   git clone [https://github.com/pudivijayakrishna/ds_plot_analyzer.git](https://github.com/pudivijayakrishna/ds_plot_analyzer.git)
2. Install dependencies:
   pip install -r requirements.txt
3. Run the app:
   streamlit run app.py
---

### **How to Add This to Your Project**

1.  **In VS Code:** Create a new file named `README.md`.
2.  **Paste the content:** Copy the text above and paste it in. Make sure to replace `[PASTE_YOUR_STREAMLIT_URL_HERE]` with your actual link.
3.  **Save and Sync:**
    * Click the **Source Control** icon.
    * Click **+** to stage the README.
    * Type `"Add professional README"` as the message.
    * Click **Commit** and then **Sync Changes**.



### **Final Touch: The "About" Section**
On your GitHub repository page (in your browser), click the **Gear Icon** ⚙️ next to "About" on the right side. Paste your Streamlit URL into the **Website** field. This makes it incredibly easy for anyone to click and see your work instantly!

**Would you like me to help you draft a short LinkedIn post to share this project with your network?**
