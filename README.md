# UniMate – AI University Assistant (RAG PDF Q&A)


A minimalist RAG demo for students: upload 1..N PDFs, ask questions, get concise answers with page citations. If the answer isn’t in your PDFs, UniMate falls back to a general answer using the same LLM.


## Quickstart


```bash
python -m venv .venv
. .venv/Scripts/activate # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env # or create manually
streamlit run app.py