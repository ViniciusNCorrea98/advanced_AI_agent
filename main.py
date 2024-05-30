from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str
from note_engine import  note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from pdf import brazil_engine

population_path = os.path.join('data', 'population.csv')
population_df = pd.read_csv(population_path)

print(population_df.head())

population_query_engine = PandasQueryEngine(df=population_df, verbose=True)
population_query_engine.update_prompts({"pandas_prompt": new_prompt})
population_query_engine.query("What is the population of brazil")

tools = [
    note_engine,
    QueryEngineTool(query_engine=population_query_engine, metadata=ToolMetadata(
        name="population_data",
        description='This give information at the world population and demographics'
    )),
    QueryEngineTool(query_engine=brazil_engine, metadata=ToolMetadata(
        name="brazil_data",
        description='This give detailed information about Brazil country'
    ))
]


llm = OpenAI(model='gpt-3.5-turbo-0613')
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context="")


while(prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)