
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
from app.agent.helpers.draw_graph import draw_graph
from app.core.logger import logger


from fastapi import APIRouter, HTTPException, status


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
        if self.graph is None:
            raise RuntimeError(
                "Failed to initialize graph: build_graph returned None")
        self.sys_msg = load_prompt_from_yaml("REACT_LANGGRAPH_PROMPT")

    def build_graph(self):
        try:
            builder = StateGraph(State)
            builder.add_node("assistant", self.nodes["assistant"])
            builder.add_node("summurize",
                             self.nodes["summurize"])

            builder.add_edge(START, "assistant")
            builder.add_conditional_edges(
                "assistant", self.nodes["summury_decision"],
                {
                    True: "summurize",
                    False: END
                })
            builder.add_edge("summurize", END)
            builder.add_edge("assistant", END)

            graph = builder.compile(checkpointer=self.memory)
            # draw_graph(graph)
            return graph
        except Exception as e:
            logger.error(f"getGraphResponse - graph builder - {e}")
            return None

    def get_response(self, query: str, config: dict, user_id: str, thread_id: str, msg_id: str):
        try:
            prompt = get_formated_prompt(query, user_id)
            timestamp = datetime.utcnow().isoformat()
            user_msg = HumanMessage(
                content=prompt,
                id=msg_id,
                metadata={"user_id": user_id,
                          "thread_id": thread_id}
            )

            try:
                result = self.graph.invoke(
                    {"messages": [user_msg], "system_message": self.sys_msg}, config)
                if result:
                    ai_messages = result.get("messages", [])
                    final_message = None
                    AI_RESPONSE_ = [
                        msg for msg in ai_messages if isinstance(msg, AIMessage)]
                    if AI_RESPONSE_:
                        final_message = AI_RESPONSE_[-1].content
                    else:
                        logger.error(f"No AI Response Found In State")
                        raise HTTPException(
                            status_code=500, detail="No AI Message")
                else:
                    logger.error(f"No AI Response Found In State")
                    raise HTTPException(
                        status_code=500, detail="Invoke Messgae Failed")

            except HTTPException as http_exc:
                raise http_exc
            except Exception as e:
                logger.error(f"getGraphResponse Graph invoke : {e}")
                return False

            if final_message is None:
                raise RuntimeError("Agent produced no output")
                return False

            try:
                self.mongo_store.add_message(
                    user_id=user_id,
                    thread_id=thread_id,
                    role="user",
                    text=query,
                    message_id=msg_id,
                    embedding=get_embedding(query),
                    summarize_fn="None",
                    embed_summary_fn=get_embedding,
                    context_fn=[]
                )

                self.mongo_store.add_message(
                    user_id=user_id,
                    thread_id=thread_id,
                    role="assistant",
                    message_id=msg_id,
                    text=final_message,
                    embedding=get_embedding(final_message),
                    summarize_fn="None",
                    embed_summary_fn=get_embedding,
                    context_fn=[]
                )
            except Exception as e:
                logger.error(f"getGraphResponse Convo save : {e}")
                return False

            return final_message
        except Exception as e:
            logger.error(f"get_graph_res : {e}")
            return False
