from remembr.memory.memory import MemoryItem
from remembr.memory.milvus_memory import MilvusMemory
from remembr.agents.remembr_agent import ReMEmbRAgent

memory = MilvusMemory("test_collection", db_ip='127.0.0.1')
memory.reset()

memory_item = MemoryItem(
    caption="I see a desk", 
    time=1.1, 
    position=[0.0, 0.0, 0.0], 
    theta=3.14
)
memory.insert(memory_item)

agent = ReMEmbRAgent(llm_type='llama3.1:8b')
agent.set_memory(memory)
response = agent.query("Where can I sit?")

print(response.position)
print(response.text) 
