from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama

from neo4j import GraphDatabase
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.retrievers import Text2CypherRetriever

# taken from https://www.youtube.com/watch?v=nZnwExutgsY

load_dotenv()

llama_model = ChatOllama(model="llama3.1")

driver = GraphDatabase.driver("bolt://localhost:7687", auth=('neo4j', 'password123'))

driver.execute_query("""
    MERGE (f:Person {name: 'M Imran', country:'United States'})
    MERGE (c: Occupation {title: 'student'})
    MERGE (x: OS {name: "macOS"})
    MERGE (l: Person {name: 'Something Something', country: 'Australia'})

    MERGE (f)-[:OWNS]->(c)
    MERGE (f)-[:USES]->(x)
    MERGE (l)-[:CREATED]->(x)
""")

SCHEMA = """Node Labels:
Person(name, country)
Occupation(title)
OS(name)

Relationships:
    (Person)-[:OWNS]->(Occupation)
    (Person)-[:CREATED]->(OS)
    (Person)-[:USES]->(OS)
"""

retriever = Text2CypherRetriever(driver=driver, 
                                 llm=OllamaLLM(model_name="llama3.1"),
                                 neo4j_schema=SCHEMA)

@tool
def query_kg(query: str) -> str:
    """Query the knowledge graph for information. You can pass entire user questions, 
    queries. Returns graph rows"""
    results = retriever.search(query)

    #return '\n'.join(item.content for item in results.items()) or 'no content'
    #print(f'Len results: {len(results)}')
    return '\n'.join(item.content for item in results.items) or 'no content'

agent = create_agent(model=llama_model, 
                     tools=[query_kg], 
                     system_prompt="You are a helpful assistant with access to a knowledge graph. Query it whenever you need to. ")

if __name__ == "__main__":
    result = retriever.get_search_results(query_text="What is the occupation of the person who uses macos?")
    print(result)
    q = 'What is the occupation of the person who uses macOS?'
    response = agent.invoke({'messages': [('user', q)]})
    print(f'Response: {response['messages'][-1].content}')