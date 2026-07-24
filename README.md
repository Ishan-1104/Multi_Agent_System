# 🔬 ResearchMind · Multi-Agent AI Research System

**ResearchMind** is a multi-agent research pipeline that takes any topic and automatically searches the web, scrapes full-page content, drafts a structured research report, and critiques its own output — all orchestrated through a LangChain agent pipeline with a polished Streamlit front end.

🔗 **Live App:** [research-mind-18.streamlit.app](https://research-mind-18.streamlit.app/)
📦 **Repo:** [Multi_Agent_System](https://github.com/Ishan-1104/Multi_Agent_System)

---

## ✨ Overview

Give ResearchMind a topic — e.g. *"Impact of AI on global semiconductor supply chains"* — and it runs a four-stage pipeline to produce a fully cited, structured report along with an honest critique of its own work.

| Stage | Component | What it does |
|-------|-----------|---------------|
| 01 | **Search Agent** | A LangChain tool-calling agent (Mistral) uses a `web_search` tool (Tavily) to gather recent news, research, statistics, and expert analysis on the topic. |
| 02 | **Scraper** | Deterministically extracts URLs from the search results and scrapes full-page content from them (with automatic fallback across multiple candidate URLs if a page is blocked, paywalled, or too short). |
| 03 | **Writer Chain** | Synthesizes the search results + scraped content into a structured 900–1200 word report (introduction, key findings with concrete figures, conclusion, sources). |
| 04 | **Critic Chain** | Reviews the report and returns a score out of 10, strengths, areas to improve, and a one-line verdict. |

The Streamlit UI shows each stage as a live-updating pipeline card (waiting → running → done/failed), and keeps a session history of past runs.

---

## 🖥️ Features

- **Multi-agent pipeline** — separate search, scrape, write, and critique stages, each independently observable
- **Live progress UI** — step cards and progress bar update in real time as each stage runs
- **Configurable settings** (sidebar):
  - Max scrape attempts (how many candidate URLs to try before falling back to snippets only)
  - Preferred sources (e.g. Reuters, Bloomberg, IMF, World Bank, research papers) to nudge the search agent
- **Run history** — revisit any past research run from the sidebar without re-running the pipeline
- **Robust scraping** — blocked/paywalled/CAPTCHA pages are detected and skipped automatically, falling back to the next candidate URL
- **Downloadable reports** — export the final report as a `.md` file
- **Raw output inspection** — expandable panels to view raw search results and scraped content

---

## 🏗️ Architecture

```
User Topic
    │
    ▼
┌─────────────────┐
│  Search Agent    │  LangChain agent + web_search tool (Tavily)
└────────┬─────────┘
         │ URLs extracted via regex
         ▼
┌─────────────────┐
│     Scraper      │  requests + BeautifulSoup, with blocked-page
│                   │  detection & multi-URL fallback
└────────┬─────────┘
         ▼
┌─────────────────┐
│  Writer Chain     │  Prompt → ChatMistralAI → structured report
└────────┬─────────┘
         ▼
┌─────────────────┐
│  Critic Chain     │  Prompt → ChatMistralAI → score + feedback
└────────┬─────────┘
         ▼
   Final Report + Critique (rendered in Streamlit)
```

---

## 📁 Project Structure

```
Multi_Agent_System/
├── app.py          # Streamlit UI — pipeline orchestration & rendering
├── agents.py        # Agent/chain definitions (search agent, writer & critic chains)
├── tools.py          # web_search (Tavily) and scrape_url (requests + BeautifulSoup) tools
├── pipeline.py       # CLI-runnable pipeline (URL extraction, scrape fallback, full run)
├── requirements.txt  # Python dependencies
└── .env               # API keys (not committed)
```

---

## 🛠️ Tech Stack

- **[Streamlit](https://streamlit.io/)** — front-end UI
- **[LangChain](https://www.langchain.com/)** (`create_agent`, prompt templates, output parsers) — agent & chain orchestration
- **[Mistral AI](https://mistral.ai/)** (`langchain-mistralai`, `mistral-small-2506`) — LLM backing every agent/chain
- **[Tavily](https://tavily.com/)** — web search API
- **BeautifulSoup4 + Requests** — deterministic web scraping
- **python-dotenv** — environment variable management

---

## ⚙️ Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/Ishan-1104/Multi_Agent_System.git
cd Multi_Agent_System
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file in the project root:
```env
TAVILY_API_KEY=your_tavily_api_key
MISTRAL_API_KEY=your_mistral_api_key
```

### 5. Run the app
```bash
streamlit run app.py
```

Alternatively, run the pipeline directly from the command line:
```bash
python pipeline.py
```

---

## 🎛️ Usage

1. Enter a research topic (or pick one of the example chips).
2. Optionally adjust **max scrape attempts** and **preferred sources** in the sidebar.
3. Click **⚡ Run Research Pipeline**.
4. Watch each stage light up in the pipeline panel as it runs.
5. Review the generated **report** and **critic feedback**, expand raw search/scrape output if needed, and download the report as Markdown.
6. Revisit any past run from the **History** panel in the sidebar.

---

## 🚧 Known Limitations

- Scraping relies on static HTML parsing (no JavaScript rendering), so heavily JS-rendered pages may fail and fall back to search snippets.
- Search queries are capped (150 characters for the agent query, 400 characters after enhancement) to stay within Tavily's limits.
- The keyboard shortcut (⌘/Ctrl + Enter to run) reaches into the parent DOM via an iframe and may break across Streamlit versions.

---

## 📄 License

This project is open source — feel free to fork, adapt, and build on it. (Add your preferred license, e.g. MIT, here.)

---

## 🙌 Acknowledgements

Built with LangChain, Mistral AI, Tavily, and Streamlit.
