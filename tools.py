from langchain.tools import tool 
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os 
from dotenv import load_dotenv
from rich import print
load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

BLOCKED_MARKERS = [
    "access issue", "captcha", "are you a robot", "enable javascript",
    "subscribe to continue", "subscription required", "please verify you are human",
    "403 forbidden", "just a moment", "checking your browser",
]

def _looks_blocked(text: str) -> bool:
    if not text or len(text.strip()) < 400:
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in BLOCKED_MARKERS)

@tool
def web_search(query: str) -> str:
    """Search the web for reliable research articles."""

    # Defensive: agent-provided query could be arbitrarily long
    query = query.strip()[:150]

    enhanced_query = (
        f"{query} — recent news, academic research, statistics. "
        f"Prefer Reuters, Bloomberg, CNBC, Investopedia, IMF, World Bank, "
        f"ScienceDirect, academic journals. Ignore dictionary definitions."
    )

    # Hard safety net against Tavily's 400-char limit
    enhanced_query = enhanced_query[:400]

    results = tavily.search(
        query=enhanced_query,
        topic="general",
        search_depth="advanced",
        max_results=5,
    )

    out = []
    for r in results["results"]:
        out.append(
            f"Title: {r['title']}\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r['content']}\n"
        )
    return "\n----\n".join(out)

@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading.
    Returns a string starting with 'SCRAPE_FAILED:' if the page could not be
    read (blocked, paywalled, too short, or errored) so the caller can try
    a different URL instead of treating this as real content."""
    try:
        resp = requests.get(
            url,
            timeout=8,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        if resp.status_code != 200:
            return f"SCRAPE_FAILED: HTTP {resp.status_code} for {url}"

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "form"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        if _looks_blocked(text):
            return f"SCRAPE_FAILED: blocked or empty content for {url}"

        return text[:3000]
    except Exception as e:
        return f"SCRAPE_FAILED: {str(e)} for {url}"