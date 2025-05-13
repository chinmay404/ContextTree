
from app.agent.helpers.load_prompt import load_prompt_from_yaml
from langchain_core.messages import HumanMessage
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tool_node, tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from app.agent.helpers.load_prompt import load_prompt_from_yaml
from datetime import datetime
from langgraph.checkpoint.memory import InMemorySaver
from app.agent.nodes.assistant_node import AgentNodes
from app.agent.state import State
from app.agent.prompts.prompt_formation import get_formated_prompt
from langchain.schema import AIMessage, HumanMessage
from datetime import datetime
from uuid import uuid4
from app.agent.utils.saver import redis_saver
from app.agent.store.MongoStore import MongoConversationStore
from app.agent.utils.embeddings import get_embedding

Nodes = AgentNodes()


class getGraphResponse():
    def __init__(self):
        self.memory = redis_saver()
        if self.memory is None:
            raise MemoryError(
                "Failed to initialize RedisSaver: memory is None")
        # self.memory = InMemorySaver()   
        self.mongo_store = MongoConversationStore()
        self.nodes = Nodes.get_nodes()
        self.graph = self.build_graph()
        self.sys_msg = load_prompt_from_yaml("REACT_LANGGRAPH_PROMPT")

    def build_graph(self):
        builder = StateGraph(State)
        builder.add_node("assistant", self.nodes["assistant"])

        builder.add_edge(START, "assistant")
        builder.add_edge("assistant", END)

        graph = builder.compile(checkpointer=self.memory)
        return graph

    def get_response(self, query: str, config: dict, user_id: str, thread_id: str):
        # 1) Format prompt, attach metadata
        prompt = get_formated_prompt(query, user_id)
        timestamp = datetime.utcnow().isoformat()
        msg_id = f"{timestamp}-{uuid4().hex}"
        user_msg = HumanMessage(
            content=prompt,
            metadata={"user_id": user_id,
                      "msg_id": msg_id, "thread_id": thread_id}
        )

        # 2) Step the graph (writes to Redis)
        result = self.graph.invoke(
            {"messages": [user_msg], "system_message": self.sys_msg},
            config
        )
        ai_messages = result.get("messages", [])

        # 3) Extract the first non-empty AI response
        final_message = None
        for msg in reversed(ai_messages):
            if getattr(msg, "content", None):
                final_message = msg.content
                break

        if final_message is None:
            raise RuntimeError("Agent produced no output")

        # 4) Persist the user turn to Mongo
        self.mongo_store.add_message(
            user_id=user_id,
            thread_id=thread_id,
            role="user",
            text=query,
            embedding=get_embedding(query),
            summarize_fn="None",
            embed_summary_fn=get_embedding,
            context_fn=[]
        )

        # 5) Persist the assistant turn to Mongo
        self.mongo_store.add_message(
            user_id=user_id,
            thread_id=thread_id,
            role="assistant",
            text=final_message,
            embedding=get_embedding(final_message),
            summarize_fn="None",
            embed_summary_fn=get_embedding,
            context_fn=[]
        )

        # 6) Return the AI’s response
        return final_message
