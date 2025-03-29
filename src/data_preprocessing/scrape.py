import os

from src.logging.logging import log_error


def load_data():
    pass


def tavily_search(query: str) -> str:
    """
    Perform a web search using Tavily API.
    
    Args:
        query (str): The search query.
        
    Returns:
        str: JSON response from Tavily as a string.
    """
    tavily_api_key = os.tavily_api_key
        
    url = "https://api.tavily.com/search"
    
    payload = {
        "query": query,
        "topic": "general",
        "search_depth": "basic",
        "max_results": 5,
        "include_answer": True,
        "include_raw_content": False,
        "include_images": False,
        "include_image_descriptions": False,
    }
    
    headers = {
        "Authorization": f"Bearer {tavily_api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = await make_http_request(
            url=url,
            method="POST",
            data=payload,
            headers=headers,
        )
        
        return response.text
    except Exception as e:
        log_error("Unexpected error in Tavily search", e, {"query": query})
        raise ExternalAPIError(f"Error in Tavily search: {str(e)}")