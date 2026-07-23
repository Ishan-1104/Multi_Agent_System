import re
from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain
from tools import scrape_url
from pprint import pprint
from langchain_core.messages import ToolMessage


def extract_urls(search_results: str) -> list[str]:
    return re.findall(r"URL:\s*(\S+)", search_results)


def get_valid_scraped_content(urls: list[str], max_tries: int = 4) -> str | None:
    for url in urls[:max_tries]:
        print(f"  trying to scrape: {url}")
        result = scrape_url.invoke({"url": url})
        if not result.startswith("SCRAPE_FAILED"):
            return result
        print(f"  -> failed ({result[:80]})")
    return None


def run_research_pipeline(topic: str) -> dict:
    state = {}

    print("\n" + " =" * 50)
    print("step 1 - search agent is working ...")
    print("=" * 50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [
            ("user", f"""
    You are a research assistant.

    You MUST use the web_search tool.

    Search for comprehensive, factual, and up-to-date information about:

    Topic:
    {topic}

    Search for:
    - Recent news
    - Academic research
    - Expert analysis
    - Statistics
    - Historical context

    Prefer sources like Reuters, Bloomberg, CNBC, Investopedia,
    IMF, World Bank, government reports, and research papers.

    Return the tool results only.
    """)
        ]
    })

    tool_message = next(
        msg for msg in search_result["messages"]
        if isinstance(msg, ToolMessage)
    )
    state["search_results"] = tool_message.content
    print("\n search result ", state['search_results'])

    # step 2 - deterministic scraping with fallback across URLs
    print("\n" + " =" * 50)
    print("step 2 - scraping top resources ...")
    print("=" * 50)

    urls = extract_urls(state["search_results"])
    scraped = get_valid_scraped_content(urls)

    if scraped:
        state["scraped_content"] = scraped
    else:
        print("⚠️ All scrape attempts failed — continuing with search results only.")
        state["scraped_content"] = "(No full-page content could be retrieved; relying on search snippets only.)"

    print("\nscraped content: \n", state['scraped_content'])

    # step 3 - writer chain
    print("\n" + " =" * 50)
    print("step 3 - Writer is drafting the report ...")
    print("=" * 50)

    research_combined = (
        f"SEARCH RESULTS : \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined,
    })
    print("\n Final Report\n", state['report'])

    # step 4 - critic
    print("\n" + " =" * 50)
    print("step 4 - critic is reviewing the report ")
    print("=" * 50)

    state["feedback"] = critic_chain.invoke({"report": state['report']})
    print("\n critic report \n", state['feedback'])

    return state


if __name__ == "__main__":
    topic = input("\n Enter a research topic : ")
    run_research_pipeline(topic)