from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from crewai.tools import tool
import os
from dotenv import load_dotenv

load_dotenv()

from crewai import LLM
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
llm = LLM(
    model="huggingface/Qwen/Qwen2.5-72B-Instruct",
    api_key=HUGGINGFACE_API_KEY,
)

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
search_tool = SerperDevTool()

@tool("Make a calculation")
def calculate(operation: str):
    """Perform mathematical calculations."""
    return eval(operation)

def create_crew(destination):
    research_agent = Agent(
        role="Travel Researcher",
        goal=f"Find the best travel options for {destination}",
        tools=[search_tool],
        backstory="An expert at discovering and analyzing travel opportunities.",
        llm=llm,
        verbose=True,
        allow_delegation=True,
    )

    planner_agent = Agent(
        role="Travel Planner",
        goal=f"Create optimal itineraries for {destination}",
        backstory="Specializes in creating well-balanced and efficient travel plans.",
        llm=llm,
        verbose=True,
        allow_delegation=True,
    )

    calculate_agent = Agent(
        role="Budget Optimizer",
        goal=f"Optimize travel plans for {destination}",
        backstory="Expert at balancing budget and value for travel plans.",
        tools=[calculate],
        llm=llm,
        verbose=True,
        allow_delegation=True,
    )

    tasks = [
        Task(
            description=f"Research travel options for flights, hotels, and activities in {destination}",
            agent=research_agent,
            expected_output="A list of travel options matching user preferences",
        ),
        Task(
            description=f"Create initial itinerary for {destination}",
            agent=planner_agent,
            expected_output="A detailed travel itinerary covering all days",
        ),
        Task(
            description=f"Optimize itinerary for {destination} considering budget and time",
            agent=calculate_agent,
            expected_output="A final optimized itinerary with cost breakdown",
        ),
    ]

    crew = Crew(agents=[research_agent, planner_agent, calculate_agent], tasks=tasks, process=Process.sequential)
    return crew

destination = input("Enter your travel destination: ")
crew = create_crew(destination)
result = crew.kickoff()

print("\nFinal Itinerary:\n", result)
