# streamlit_app.py
import streamlit as st
import os
from typing import Annotated, Sequence, List, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.riza.command import ExecPython
from langchain_groq import ChatGroq
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent
from pprint import pformat
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve API keys from environment variables
groq_api_key = os.environ.get("Groq")
riza_api_key = os.environ.get("RIZA_API_KEY")
tavily_api_key = os.environ.get("TAVILY_API_KEY")

# Initialize ChatGroq
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

# Define Tools
tool_tavily = TavilySearchResults(max_results=2, api_key=tavily_api_key)
tool_code_interpreter = ExecPython(api_key=riza_api_key)
tools = [tool_tavily, tool_code_interpreter]

# Define Supervisor Agent
supervisor_prompt = (
    '''You are a workflow supervisor managing a team of three agents: Prompt Enhancer, Researcher, and Coder. Your role is to direct the flow of tasks by selecting the next agent based on the current stage of the workflow. For each task, provide a clear rationale for your choice, ensuring that the workflow progresses logically, efficiently, and toward a timely completion.

**Team Members**:
1. Enhancer: Use prompt enhancer as the first preference, to Focuse on clarifying vague or incomplete user queries, improving their quality, and ensuring they are well-defined before further processing.
2. Researcher: Specializes in gathering information.
3. Coder: Handles technical tasks related to caluclation, coding, data analysis, and problem-solving, ensuring the correct implementation of solutions.

**Responsibilities**:
1. Carefully review each user request and evaluate agent responses for relevance and completeness.
2. Continuously route tasks to the next best-suited agent if needed.
3. Ensure the workflow progresses efficiently, without terminating until the task is fully resolved.

Your goal is to maximize accuracy and effectiveness by leveraging each agent’s unique expertise while ensuring smooth workflow execution.
'''
)

class Supervisor(BaseModel):
    next: Literal["enhancer", "researcher", "coder"] = Field(
        description="Specifies the next worker in the pipeline: "
                    "'enhancer' for enhancing the user prompt if it is unclear or vague, "
                    "'researcher' for additional information gathering, "
                    "'coder' for solving technical or code-related problems."
    )
    reason: str = Field(
        description="The reason for the decision, providing context on why a particular worker was chosen."
    )

def supervisor_node(state: MessagesState) -> Command[
    Literal["enhancer", "researcher", "coder"]
]:
    """
    Supervisor node for routing tasks based on the current state and LLM response.
    Args:
        state (MessagesState): The current state containing message history.
    Returns:
        Command: A command indicating the next state or action.
    """
    messages = [
        {"role": "system", "content": supervisor_prompt},
    ] + state["messages"]

    response = llm.with_structured_output(Supervisor).invoke(messages)
    goto = response.next
    reason = response.reason
    st.session_state.workflow_output.append(f"**Supervisor**:  Going to '{goto}' because: {reason}")
    return Command(
        update={
            "messages": [
                HumanMessage(content=reason, name="supervisor")
            ]
        },
        goto=goto,
    )

def enhancer_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    """
    Enhancer node for refining and clarifying user inputs.

    Args:
        state (MessagesState): The current state containing the conversation history.

    Returns:
        Command: A command to update the state with the enhanced query and route back to the supervisor.
    """
    enhancer_prompt = (
        "You are an advanced query enhancer. Your task is to:\n"
        "Don't ask anything to the user, select the most appropriate prompt"
        "1. Clarify and refine user inputs.\n"
        "2. Identify any ambiguities in the query.\n"
        "3. Generate a more precise and actionable version of the original request.\n"
    )

    messages = [
        {"role": "system", "content": enhancer_prompt},
    ] + state["messages"]

    enhanced_query = llm.invoke(messages)
    st.session_state.workflow_output.append(f"**Enhancer**: Enhanced query: {enhanced_query.content}")
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=enhanced_query.content,
                    name="enhancer"
                )
            ]
        },
        goto="supervisor",
    )

def research_node(state: MessagesState) -> Command[Literal["validator"]]:
    """
    Research node for leveraging a ReAct agent to process research-related tasks.

    Args:
        state (MessagesState): The current state containing the conversation history.

    Returns:
        Command: A command to update the state with the research results and route to the validator.
    """
    research_agent = create_react_agent(
        llm,
        tools=[tool_tavily],
        state_modifier="You are a researcher. Focus on gathering information and generating content. Do not perform any other tasks"
    )
    result = research_agent.invoke(state)
    st.session_state.workflow_output.append(f"**Researcher**: Result: {result['messages'][-1].content}")
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=result["messages"][-1].content,
                    name="researcher"
                )
            ]
        },
        goto="validator",
    )

def code_node(state: MessagesState) -> Command[Literal["validator"]]:
    """
    Coder node for leveraging a ReAct agent to process analyzing, solving math questions, and executing code.

    Args:
        state (MessagesState): The current state containing the conversation history.

    Returns:
        Command: A command to update the state with the research results and route to the validator.
    """
    code_agent = create_react_agent(
        llm,
        tools=[tool_code_interpreter],
        state_modifier=(
            "You are a coder and analyst. Focus on mathematical caluclations, analyzing, solving math questions, "
            "and executing code. Handle technical problem-solving and data tasks."
        )
    )

    result = code_agent.invoke(state)

    st.session_state.workflow_output.append(f"**Coder**: Result: {result['messages'][-1].content}")

    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="coder")
            ]
        },
        goto="validator",
    )

validator_prompt = '''
You are a workflow validator. Your task is to ensure the quality of the workflow. Specifically, you must:
- Review the user's question (the first message in the workflow).
- Review the answer (the last message in the workflow).
- If the answer satisfactorily addresses the question, signal to end the workflow.
- If the answer is inappropriate or incomplete, signal to route back to the supervisor for re-evaluation or further refinement.
Ensure that the question and answer match logically and the workflow can be concluded or continued based on this evaluation.

Routing Guidelines:
1. 'supervisor' Agent: For unclear or vague state messages.
2. Respond with 'FINISH' to end the workflow.
'''

class Validator(BaseModel):
    next: Literal["supervisor", "FINISH"] = Field(
        description="Specifies the next worker in the pipeline: 'supervisor' to continue or 'FINISH' to terminate."
    )
    reason: str = Field(
        description="The reason for the decision."
    )

def validator_node(state: MessagesState) -> Command[Literal["supervisor", "__end__"]]:
    """
    Validator node for checking if the question and the answer are appropriate.

    Args:
        state (MessagesState): The current state containing message history.

    Returns:
        Command: A command indicating whether to route back to the supervisor or end the workflow.
    """
    user_question = state["messages"][0].content
    agent_answer = state["messages"][-1].content

    messages = [
        {"role": "system", "content": validator_prompt},
        {"role": "user", "content": user_question},
        {"role": "assistant", "content": agent_answer},
    ]

    response = llm.with_structured_output(Validator).invoke(messages)

    goto = response.next
    reason = response.reason

    if goto == "FINISH" or goto == END:
        goto = END
        st.session_state.workflow_output.append(f"**Validator**: Workflow finished because: {reason}")
    else:
        st.session_state.workflow_output.append(f"**Validator**: Going back to Supervisor because: {reason}")
    return Command(
        update={
            "messages": [
                HumanMessage(content=reason, name="validator")
            ]
        },
        goto=goto,
    )

# Initialize the StateGraph
builder = StateGraph(MessagesState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("enhancer", enhancer_node)
builder.add_node("researcher", research_node)
builder.add_node("coder", code_node)
builder.add_node("validator", validator_node)

builder.add_edge(START, "supervisor")
graph = builder.compile()

# Streamlit App
st.title("LangGraph Workflow")

if "workflow_output" not in st.session_state:
    st.session_state.workflow_output = []

user_query = st.text_input("Enter your query:")

if st.button("Run Workflow"):
    if user_query:
        st.session_state.workflow_output = []  # Clear previous outputs
        inputs = {"messages": [("user", user_query)]}
        for output in graph.stream(inputs):
            for key, value in output.items():
                if value is None:
                    continue
                if isinstance(value, dict):
                    st.session_state.workflow_output.append(f"**Output from node '{key}'**: {pformat(value, indent=2, width=80)}")
                else:
                    st.session_state.workflow_output.append(f"**Output from node '{key}'**: {value}")

        st.subheader("Workflow Execution:")
        for line in st.session_state.workflow_output:
            st.write(line)
    else:
        st.warning("Please enter a query.")