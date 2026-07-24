import time
from datetime import datetime
from langchain_core.messages import ToolMessage

from agents import build_search_agent, writer_chain, critic_chain
from pipeline import extract_urls, get_valid_scraped_content

import streamlit as st
import streamlit.components.v1 as components

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchMind · AI Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #e8e4dc;
}

.stApp {
    background: #0a0a0f;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(255,140,50,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(255,80,30,0.08) 0%, transparent 55%);
}

#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1200px; }

.hero {
    text-align: center;
    padding: 3.5rem 0 2.5rem;
    position: relative;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #ff8c32;
    margin-bottom: 1rem;
    opacity: 0.9;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.8rem, 6vw, 5rem);
    font-weight: 800;
    line-height: 1.0;
    letter-spacing: -0.03em;
    color: #f0ebe0;
    margin: 0 0 1rem;
}
.hero h1 span { color: #ff8c32; }
.hero-sub {
    font-size: 1.05rem;
    font-weight: 300;
    color: #a09890;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.65;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,140,50,0.3), transparent);
    margin: 2rem 0;
}

.stTextArea > div > div > textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,140,50,0.25) !important;
    border-radius: 14px !important;
    color: #f0ebe0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    line-height: 1.55 !important;
    padding: 0.9rem 1.1rem !important;
    resize: vertical !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextArea > div > div > textarea:focus {
    border-color: #ff8c32 !important;
    box-shadow: 0 0 0 3px rgba(255,140,50,0.12) !important;
}
.stTextArea > label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #ff8c32 !important;
    font-weight: 500 !important;
}

/* ── Primary button (Run Research Pipeline) ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #ff8c32 0%, #ff5a1a 100%) !important;
    color: #0a0a0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2.2rem !important;
    cursor: pointer !important;
    transition: transform 0.15s, box-shadow 0.15s, opacity 0.15s !important;
    box-shadow: 0 4px 20px rgba(255,140,50,0.3) !important;
    width: 100%;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(255,140,50,0.4) !important;
    opacity: 0.95 !important;
}
.stButton > button[kind="primary"]:active { transform: translateY(0) !important; }

/* ── Secondary buttons (example topic chips) ── */
.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 6px !important;
    padding: 0.25rem 0.7rem !important;
    font-size: 0.75rem !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #a09890 !important;
    font-weight: 400 !important;
    letter-spacing: normal !important;
    box-shadow: none !important;
    width: auto !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: rgba(255,140,50,0.4) !important;
    color: #ff8c32 !important;
    transform: none !important;
    box-shadow: none !important;
}

.step-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
}
.step-card.active {
    border-color: rgba(255,140,50,0.4);
    background: rgba(255,140,50,0.04);
}
.step-card.done {
    border-color: rgba(80,200,120,0.3);
    background: rgba(80,200,120,0.03);
}
.step-card.failed {
    border-color: rgba(220,80,80,0.4);
    background: rgba(220,80,80,0.04);
}
.step-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    border-radius: 14px 0 0 14px;
    background: rgba(255,255,255,0.05);
    transition: background 0.3s;
}
.step-card.active::before { background: #ff8c32; }
.step-card.done::before   { background: #50c878; }
.step-card.failed::before { background: #dc5050; }

.step-header {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.3rem;
}
.step-num {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    color: #ff8c32;
    opacity: 0.7;
}
.step-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: #f0ebe0;
}
.step-status {
    margin-left: auto;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
}
.status-waiting  { color: #555; }
.status-running  { color: #ff8c32; }
.status-done     { color: #50c878; }
.status-failed   { color: #dc5050; }

.result-panel {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.8rem 2rem;
    margin-top: 1rem;
    margin-bottom: 1.5rem;
}
.result-panel-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #ff8c32;
    margin-bottom: 1rem;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid rgba(255,140,50,0.15);
}
.result-content {
    font-size: 0.92rem;
    line-height: 1.8;
    color: #cdc8bf;
    white-space: pre-wrap;
    font-family: 'DM Sans', sans-serif;
}

.report-panel {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,140,50,0.2);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-top: 1rem;
}
.feedback-panel {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(80,200,120,0.2);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-top: 1rem;
}
.panel-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    padding-bottom: 0.7rem;
}
.panel-label.orange { color: #ff8c32; border-bottom: 1px solid rgba(255,140,50,0.15); }
.panel-label.green   { color: #50c878; border-bottom: 1px solid rgba(80,200,120,0.15); }

.stSpinner > div { color: #ff8c32 !important; }

details summary {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #a09890 !important;
    letter-spacing: 0.1em !important;
    cursor: pointer;
}

.section-heading {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #f0ebe0;
    margin: 2rem 0 1rem;
}

.notice {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #605850;
    text-align: center;
    margin-top: 3rem;
    letter-spacing: 0.08em;
}

.url-chip {
    display: inline-block;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.72rem;
    color: #a09890;
    font-family: 'DM Mono', monospace;
    margin: 0.15rem 0.3rem 0.15rem 0;
}

/* ── Sidebar (settings + history) ── */
[data-testid="stSidebar"] {
    background: #0d0d12;
    border-right: 1px solid rgba(255,140,50,0.08);
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
    width: 100% !important;
    text-align: left !important;
    justify-content: flex-start !important;
}
[data-testid="stSidebar"] label {
    color: #a09890 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] input[type="range"] { accent-color: #ff8c32 !important; }
[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
    background-color: rgba(255,140,50,0.22) !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #ff8c32, #ff5a1a) !important;
}

/* ── Report / feedback panels get slightly tighter padding side-by-side ── */
.report-panel, .feedback-panel { padding: 1.6rem 1.8rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Helper: render a step card ────────────────────────────────────────────────
def step_card(num: str, title: str, state: str, desc: str = ""):
    status_map = {
        "waiting": ("WAITING", "status-waiting"),
        "running": ("● RUNNING", "status-running"),
        "done":    ("✓ DONE",   "status-done"),
        "failed":  ("✕ FAILED", "status-failed"),
    }
    label, cls = status_map.get(state, ("", ""))
    card_cls = {"running": "active", "done": "done", "failed": "failed"}.get(state, "")
    st.markdown(f"""
    <div class="step-card {card_cls}">
        <div class="step-header">
            <span class="step-num">{num}</span>
            <span class="step-title">{title}</span>
            <span class="step-status {cls}">{label}</span>
        </div>
        {"<div style='font-size:0.82rem;color:#706860;margin-top:0.3rem;'>"+desc+"</div>" if desc else ""}
    </div>
    """, unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
DEFAULT_SOURCES = ["Reuters", "Bloomberg", "CNBC", "Investopedia", "IMF", "World Bank",
                    "Government reports", "Research papers"]

for key, default in (
    ("results", {}),
    ("running", False),
    ("done", False),
    ("topic_locked", ""),
    ("history", []),
    ("history_recorded", False),
    ("max_scrape_tries", 4),
    ("preferred_sources", list(DEFAULT_SOURCES)),
):
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar: settings + run history ─────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-heading" style="margin-top:0;font-size:1.1rem;">⚙️ Settings</div>',
                unsafe_allow_html=True)

    st.session_state.max_scrape_tries = st.slider(
        "Max scrape attempts",
        min_value=1, max_value=8,
        value=st.session_state.max_scrape_tries,
        help="How many candidate URLs the scraper will try before falling back to search snippets only.",
    )

    st.session_state.preferred_sources = st.multiselect(
        "Preferred sources",
        options=DEFAULT_SOURCES,
        default=st.session_state.preferred_sources,
        help="Nudges the search agent toward these kinds of sources.",
    )

    st.markdown('<div class="divider" style="margin:1.2rem 0;"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-heading" style="font-size:1.1rem;">🕘 History</div>',
                unsafe_allow_html=True)

    if not st.session_state.history:
        st.caption("No past runs yet — completed research will show up here.")
    else:
        for run in reversed(st.session_state.history):
            label = run["topic"] if len(run["topic"]) <= 42 else run["topic"][:42] + "…"
            if st.button(label, key=f"history_{run['id']}", type="secondary"):
                st.session_state.results = dict(run["results"])
                st.session_state.topic_locked = run["topic"]
                st.session_state.topic_input = run["topic"]
                st.session_state.running = False
                st.session_state.done = True
                st.rerun()
            st.caption(run["timestamp"])

        if st.button("🗑️ Clear history", key="clear_history", type="secondary"):
            st.session_state.history = []
            st.rerun()


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Multi-Agent AI System</div>
    <h1>Research<span>Mind</span></h1>
    <p class="hero-sub">
        Search, scrape, write, and critique — four coordinated stages
        collaborate to deliver a polished research report on any topic.
    </p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)


# ── Layout: input left, pipeline right ───────────────────────────────────────
col_input, col_spacer, col_pipeline = st.columns([5, 0.5, 4])

def _set_topic(value: str):
    st.session_state.topic_input = value

with col_input:
    if "topic_input" not in st.session_state:
        st.session_state.topic_input = ""

    topic = st.text_area(
        "Research Topic",
        placeholder="e.g. Impact of AI on global semiconductor supply chains",
        key="topic_input",
        height=110,
    )
    st.caption(f"{len(topic)} characters · {len(topic.split())} words   ·   Tip: ⌘/Ctrl + Enter to run")

    run_btn = st.button(
        "⚡  Run Research Pipeline",
        use_container_width=True,
        type="primary",
    )

    # Best-effort keyboard shortcut: Ctrl/Cmd+Enter while focused in the topic
    # box clicks the Run button. Runs inside its own iframe and reaches into
    # the parent document (same-origin), which is how Streamlit serves pages —
    # this can break across Streamlit versions since it relies on DOM structure.
    components.html("""
        <script>
        const parentDoc = window.parent.document;
        if (!window.__researchmind_shortcut_bound) {
            window.__researchmind_shortcut_bound = true;
            parentDoc.addEventListener('keydown', function(e) {
                const isSubmitCombo = (e.metaKey || e.ctrlKey) && e.key === 'Enter';
                if (!isSubmitCombo) return;
                const active = parentDoc.activeElement;
                if (!active || active.tagName !== 'TEXTAREA') return;
                const buttons = parentDoc.querySelectorAll('button');
                for (const btn of buttons) {
                    if (btn.innerText.includes('Run Research Pipeline')) {
                        btn.click();
                        break;
                    }
                }
            });
        }
        </script>
    """, height=0)

    st.markdown(
        '<span style="font-family:\'DM Mono\',monospace;font-size:0.68rem;'
        'color:#605850;letter-spacing:0.1em;">TRY →</span>',
        unsafe_allow_html=True,
    )
    examples = ["LLM agents in 2026", "CRISPR gene editing", "Fusion energy progress"]
    chip_cols = st.columns(len(examples))
    for c, ex in zip(chip_cols, examples):
        with c:
            st.button(ex, key=f"example_{ex}", type="secondary", on_click=_set_topic, args=(ex,))

with col_pipeline:
    st.markdown('<div class="section-heading">Pipeline</div>', unsafe_allow_html=True)

    r = st.session_state.results
    steps = ["search_results", "scraped_content", "report", "feedback"]

    if st.session_state.running or r:
        completed = sum(1 for k in steps if k in r)
        st.progress(completed / len(steps), text=f"{completed}/{len(steps)} stages complete")

    def step_state(step_key):
        if step_key in r:
            return "failed" if r.get(f"{step_key}_failed") else "done"
        if st.session_state.running:
            for k in steps:
                if k not in r:
                    return "running" if k == step_key else "waiting"
        return "waiting"

    step_card("01", "Search Agent",  step_state("search_results"),
               "Web search for news, research & statistics")
    step_card("02", "Scraper",       step_state("scraped_content"),
               "Deterministic fallback scraping across top URLs")
    step_card("03", "Writer Chain",  step_state("report"),
               "Drafts the full research report")
    step_card("04", "Critic Chain",  step_state("feedback"),
               "Reviews & scores the report")


# ── Trigger run ───────────────────────────────────────────────────────────────
if run_btn:
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        st.session_state.results = {}
        st.session_state.running = True
        st.session_state.done = False
        st.session_state.history_recorded = False
        st.session_state.topic_locked = topic.strip()
        st.rerun()


# ── Execute pipeline: ONE step per script run, then rerun ──────────────────
# Streamlit only redraws the page on rerun. Running all 4 steps inline in a
# single pass (as before) meant the step cards were drawn once at the top
# and never refreshed until the whole pipeline finished. Doing exactly one
# step per pass — then calling st.rerun() — forces a fresh redraw of the
# step cards (via step_state()) before the next stage starts, so the
# correct card lights up "running" while that stage is actually executing.
if st.session_state.running and not st.session_state.done:
    results = dict(st.session_state.results)
    topic_val = st.session_state.topic_locked

    if "search_results" not in results:
        # ── Step 1: Search Agent ──
        sources_line = (
            ", ".join(st.session_state.preferred_sources)
            if st.session_state.preferred_sources
            else "reputable, high-quality sources"
        )
        try:
            with st.spinner("🔍  Search Agent is gathering sources…"):
                search_agent = build_search_agent()
                search_result = search_agent.invoke({
                    "messages": [
                        ("user", f"""
                You are a research assistant.

                You MUST use the web_search tool.

                Search for comprehensive, factual, and up-to-date information about:

                Topic:
                {topic_val}

                Search for:
                - Recent news
                - Academic research
                - Expert analysis
                - Statistics
                - Historical context

                Prefer sources like {sources_line}.

                Return the tool results only.
                """)
                    ]
                })
                tool_message = next(
                    msg for msg in search_result["messages"]
                    if isinstance(msg, ToolMessage)
                )
                results["search_results"] = tool_message.content
                st.session_state.results = dict(results)
        except Exception as e:
            results["search_results"] = f"Search failed: {e}"
            results["search_results_failed"] = True
            st.session_state.results = dict(results)
            st.session_state.running = False
        st.rerun()

    elif "scraped_content" not in results:
        # ── Step 2: Scraper ──
        max_tries = st.session_state.max_scrape_tries
        with st.spinner("📄  Scraper is pulling full-page content from top URLs…"):
            urls = extract_urls(results["search_results"])
            scraped = get_valid_scraped_content(urls, max_tries=max_tries)
            if scraped:
                results["scraped_content"] = scraped
            else:
                results["scraped_content"] = (
                    "(No full-page content could be retrieved; relying on search snippets only.)"
                )
            results["scraped_urls_tried"] = urls[:max_tries]
            st.session_state.results = dict(results)
        st.rerun()

    elif "report" not in results:
        # ── Step 3: Writer Chain ──
        with st.spinner("✍️  Writer is drafting the report…"):
            research_combined = (
                f"SEARCH RESULTS : \n {results['search_results']} \n\n"
                f"DETAILED SCRAPED CONTENT : \n {results['scraped_content']}"
            )
            results["report"] = writer_chain.invoke({
                "topic": topic_val,
                "research": research_combined,
            })
            st.session_state.results = dict(results)
        st.rerun()

    elif "feedback" not in results:
        # ── Step 4: Critic Chain ──
        with st.spinner("🧐  Critic is reviewing the report…"):
            results["feedback"] = critic_chain.invoke({"report": results["report"]})
            st.session_state.results = dict(results)

        if not st.session_state.history_recorded:
            st.session_state.history.append({
                "id": f"{time.time():.6f}",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "topic": topic_val,
                "results": dict(results),
            })
            st.session_state.history_recorded = True

        st.session_state.running = False
        st.session_state.done = True
        st.rerun()


# ── Results display ───────────────────────────────────────────────────────────
r = st.session_state.results

if r:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Results</div>', unsafe_allow_html=True)

    if "search_results" in r:
        with st.expander("🔍 Search Results (raw)", expanded=False):
            st.markdown(
                f'<div class="result-panel"><div class="result-panel-title">Search Agent Output</div>'
                f'<div class="result-content">{r["search_results"]}</div></div>',
                unsafe_allow_html=True,
            )

    if "scraped_content" in r:
        with st.expander("📄 Scraped Content (raw)", expanded=False):
            if r.get("scraped_urls_tried"):
                chips = "".join(f'<span class="url-chip">{u}</span>' for u in r["scraped_urls_tried"])
                st.markdown(f"<div style='margin-bottom:0.8rem;'>{chips}</div>", unsafe_allow_html=True)
            st.markdown(
                f'<div class="result-panel"><div class="result-panel-title">Scraper Output</div>'
                f'<div class="result-content">{r["scraped_content"]}</div></div>',
                unsafe_allow_html=True,
            )

    if "report" in r:
        st.markdown("""
        <div class="report-panel">
            <div class="panel-label orange">📝 Final Research Report</div>
        """, unsafe_allow_html=True)
        st.markdown(r["report"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.download_button(
            label="⬇  Download Report (.md)",
            data=str(r["report"]),
            file_name=f"research_report_{int(time.time())}.md",
            mime="text/markdown",
        )

    if "feedback" in r:
        st.markdown("""
        <div class="feedback-panel">
            <div class="panel-label green">🧐 Critic Feedback</div>
        """, unsafe_allow_html=True)
        st.markdown(r["feedback"])
        st.markdown("</div>", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="notice">
    ResearchMind · Powered by your LangChain multi-agent pipeline · Built with Streamlit
</div>
""", unsafe_allow_html=True)