import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import json

load_dotenv()

# Initialize tools and agents
llm = LLM(model="gpt-4")
search_tool = SerperDevTool(n=10)

# Define agents
Senior_research_analyst = Agent(
    role="senior research analyst",
    goal="Research, analyze, and synthesize comprehensive information on the medical industry using generative AI",
    backstory="You're an expert research analyst...",
    allow_delegation=False,
    verbose=True,
    tools=[search_tool],
    llm=llm
)

content_writer = Agent(
    role="Content Writer",
    goal="Transform the research findings into engaging blog posts by maintaining accuracy",
    backstory="You're an expert content creator...",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# Define tasks
research_tasks = Task(
    description=("""Conduct comprehensive research on the medical industry using generative AI..."""),
    expected_output="""A detailed research report containing...""",
    agent=Senior_research_analyst
)

writing_task = Task(
    description=("""Using the research brief provided, create an engaging blog post..."""),
    expected_output="""A polished blog post in markdown format...""",
    agent=content_writer
)

crew = Crew(
    agents=[Senior_research_analyst, content_writer],
    tasks=[research_tasks, writing_task],
    verbose=True
)

def run_crew(topic):
    result = crew.kickoff(input={"topic": topic})
    return result

# Streamlit App
st.title("Medical Industry Generative AI Research")
st.sidebar.header("Input Options")
topic = st.sidebar.text_input("Enter Topic", "Medical Industry using Generative AI")

if st.sidebar.button("Run Research"):
    with st.spinner("Running research..."):
        result = run_crew(topic)
        st.success("Research complete!")

    # Display the results
    st.header("Research Output")
    st.write(result)

    # Download button
    st.download_button(
        label="Download Result",
        data=json.dumps(result, indent=4),
        file_name="research_result.json",
        mime="application/json"
    )
