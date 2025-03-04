from smolagents import CodeAgent, ToolCallingAgent, HfApiModel, ManagedAgent, DuckDuckGoSearchTool ,tool
import requests
from typing import Tuple
from markdownify import markdownify
import re

model = HfApiModel()

@tool
def visit_webpage(url: str) -> Tuple[str, bool]:
    """Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        A tuple containing (content, success_status) where:
        - content is either the markdown content or error message
        - success_status is a boolean indicating if the operation succeeded
    """
    try:
        # Input validation
        if not url.strip():
            return "Error: URL cannot be empty", False
        
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        # Send a GET request to the URL with timeout and headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Convert the HTML content to Markdown
        markdown_content = markdownify(response.text).strip()

        # Clean up the markdown
        markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content)  # Remove excess newlines
        markdown_content = re.sub(r'\s+$', '', markdown_content, flags=re.MULTILINE)  # Remove trailing whitespace

        return markdown_content, True

    except RequestException as e:
        error_msg = f"Error fetching the webpage: {str(e)}"
        return error_msg, False
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        return error_msg, False

# Create the travel research agent
travel_research_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool(), visit_webpage],
    model=model,
    max_steps=10
)

# Wrap the travel research agent in a ManagedAgent
managed_travel_research_agent = ManagedAgent(
    agent=travel_research_agent,
    name="travel_research",
    description="Researches travel options, destinations, and activities. Provide a specific travel-related query."
)

# Create the itinerary planning agent
itinerary_planning_agent = ToolCallingAgent(
    tools=[],  # No specific tools, will use the research from the travel research agent
    model=model,
    max_steps=5
)

# Wrap the itinerary planning agent in a ManagedAgent
managed_itinerary_planning_agent = ManagedAgent(
    agent=itinerary_planning_agent,
    name="itinerary_planner",
    description="Creates detailed travel itineraries based on research. Provide travel details and preferences."
)

# Create the budget optimization agent
budget_optimization_agent = CodeAgent(
    tools=[],  # No specific tools, will use code execution for calculations
    model=model,
    additional_authorized_imports=["numpy", "pandas"]
)

# Wrap the budget optimization agent in a ManagedAgent
managed_budget_optimization_agent = ManagedAgent(
    agent=budget_optimization_agent,
    name="budget_optimizer",
    description="Optimizes travel plans for cost and time efficiency. Provide itinerary and budget constraints."
)

# Create the manager agent
travel_manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[
        managed_travel_research_agent,
        managed_itinerary_planning_agent,
        managed_budget_optimization_agent
    ],
    additional_authorized_imports=["time", "numpy", "pandas"]
)

# Function to run the travel planner
def plan_travel(query):
    return travel_manager_agent.run(query)

# Example usage
travel_query = """
Plan a 8 day iternary from Bhubaneswar to Darjeeling and then to Gangtok including travel expenses with beautiful places and
hotels with most days in Gangtok.Find low budget air travel from Bhubaneswar to Darjeeling and then by rented 4 wheelers 
"""

result = plan_travel(travel_query)
print(result)