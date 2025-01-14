from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool

from dotenv import load_dotenv

load_dotenv()

topic = "Medical Industry using Generative Ai"

#Tool1

llm= LLM(model="gpt-4")

#Tool2

search_tool = SerperDevTool(n=10)

#Agent1

Senior_research_analyst = Agent (

    role = "senior research analyst",
    goal = f" Reaseach analyzie, and synthesize zomprehensive information on {topic} from relaible web search",
    backstory = "You're an expert research analyst with advanced web research skills. "
                "You excel at finding, analyzing, and synthesizing information from "
                "across the internet using search tools. You're skilled at "
                "distinguishing reliable sources from unreliable ones, "
                "fact-checking, cross-referencing information, and "
                "identifying key patterns and insights. You provide "
                "well-organized research briefs with proper citations "
                "and source verification. Your analysis includes both "
                "raw data and interpreted insights, making complex "
                "information accessible and actionable.",

    allow_delegation = False,
    verbose = True,
    tools = [search_tool], 
    llm=llm

    )

#agent2

content_writer = Agent(
    role = "Content Writer",
    goal = "Transform the research findings into engaging blog posts by maintainig accuracy",
    backstory = "You're an expert content creater with advanced in creating. "
                "You excel at finding, analyzing, and synthesizing information from "
                "across the internet using creation tools. You're skilled at "
                "distinguishing reliable sources from unreliable ones, "
                "fact-checking, cross-referencing information, and "
                "identifying key patterns and insights. You provide "
                "well-organized research briefs with proper citations ",
    allow_delegation = False,
    verbose = True,
    llm=llm    
    
    )

#task1

research_tasks = Task(
    description = ("""

            1. conduct the comprehensive research on {topic} including : 
                   -recent development and the news
                   -key industry trends and innovations
                   -expert opinnions and analysis
                   -statistical data and the market insights

            2.  Evaluate source credibility and the fact-check all information     
            3.  Organize findings into a structured research brief
            4. Include all relevant citations and sources
                   

    """),
    expected_output = """A detailed research report containing:
            - Executive summary of key findings
            - Comprehensive analysis of current trends and developments
            - List of verified facts and statistics
            - All citations and links to original sources
            - Clear categorization of main themes and patterns """,

    agent = Senior_research_analyst


)

#task2

writing_task = Task(

    description=("""
        Using the research brief provided, create an engaging blog post that:
        1. Transforms technical information into accessible content
        2. Maintains all factual accuracy and citations from the research
        3. Includes:
        - Attention-grabbing introduction
        - Well-structured body sections with clear headings
        - Compelling conclusion
        4. Preserves all source citations in [Source: URL] format
        5. Includes a References section at the end""" ),
        
    expected_output= """"A polished blog post in markdown fromat that : 

                - engages readers while maintaining accuracy
                - Contains properly structured sections
                - Includes inline cititations hyperlinked to the original source url
                -Presents infromation in an acessable yet informative way
                - follows proper markdown formatting use H1 for the title and H3 for the sub-sectios """,

    agent = content_writer


)


crew = Crew(

    agents = [Senior_research_analyst, content_writer],
    tasks = [research_tasks, writing_task],
    verbose = True


)

result = crew.kickoff(input = {"topic":topic})




