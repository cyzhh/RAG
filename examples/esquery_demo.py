"""\
DEPRECATED WARNING:
langchain_community.vectorstores.elasticsearch is deprecated since langchain-community==0.0.27.
"""


from dotenv import dotenv_values, load_dotenv
from elasticsearch import Elasticsearch
from langchain.agents import (
    AgentExecutor,
    AgentOutputParser,
    LLMSingleActionAgent,
    Tool,
)
from langchain.chains import LLMChain  # from langchain.chains.base import Chain
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.elasticsearch import ElasticsearchStore
from langchain_openai import ChatOpenAI
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish  # langchain_core.agents
from langchain.tools.retriever import create_retriever_tool
from modules.ES_query_retriever import ESQueryRetriever

# from qwen_chat import QwenChatModel
import re
from typing import List, Union

load_dotenv()  # ES_HOST, ES_USER, ES_SECRET, OPENAI_API_KEY
config = dotenv_values()

llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0)
# llm = QwenChatModel()

es = Elasticsearch(
    config["ES_HOST"], http_auth=(config["ES_USER"], config["ES_SECRET"])
)

retriever = ESQueryRetriever(index="chem_papers.en.*")
retriever_tool = create_retriever_tool(
    retriever,
    name="Elasticsearch Query API Retriever",
    description="A tool to access chemistry papers in Elasticsearch database with query API.",
)
tools = [retriever_tool]

template = """
Answer the following questions by searching informations on Internet. You need to figure out what knowledge you need to answer the questions and generate the corresponding queries to look up on the Internet using a search engine.
After you generate the corresponding query, you should search it. The results will be returned in string. 
Summarize results in natural language.

When generating query:
* Try to avoid any non-alphabetical symbols if possible
* Never end the query in question mark

You have access to the following tools:

{tools}

Use the following format:

Question: the input question for which you must provide a natural language answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
{agent_scratchpad}"""


# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"],
)


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )


output_parser = CustomOutputParser()
# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

# input_str = "what will happen if I throw a piece of sodium into water?"
input_str = "How to extend Li-ion battery life?"

while True:
    input_str = input()
    if input_str == "exit":
        break
    print(agent_executor.run(input_str))
