
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
from langchain_core.messages import AIMessage, HumanMessage
from datetime import datetime
from uuid import uuid4
from app.agent.utils.saver import redis_saver
from app.agent.store.MongoStore import MongoConversationStore
from app.agent.utils.embeddings import get_embedding
from app.agent.helpers.draw_graph import draw_graph
from app.core.logger import logger


from fastapi import APIRouter, HTTPException, status


class getGraphResponse():
    def __init__(self):
        self.memory = redis_saver()
        if self.memory is None:
            raise MemoryError(
                "Failed to initialize RedisSaver: memory is None")
        # self.memory = InMemorySaver()
        self.mongo_store = MongoConversationStore()
        Nodes = AgentNodes(mongo_store=self.mongo_store)
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
            # Check for existing state in LangGraph
            current_state = self.graph.get_state(config)
            initial_messages = []
            
            # If no state in Redis (new thread or forked thread), verify if we have history in Mongo
            # This handles the "fork" scenario where Mongo has history but Redis matches a new thread ID
            if not current_state.values or not current_state.values.get("messages"):
                logger.info(f"Hydrating state from Mongo for thread {thread_id}")
                mongo_msgs = self.mongo_store.get_thread_messages(user_id, thread_id)
                for m in mongo_msgs:
                    if m['role'] == 'user':
                        initial_messages.append(HumanMessage(content=m['text'], id=m['message_id']))
                    elif m['role'] in ['assistant', 'ai']:
                        initial_messages.append(AIMessage(content=m['text'], id=m['message_id']))

            # Get existing summary from MongoDB for this thread
            existing_summary = self.mongo_store.get_thread_summary(user_id, thread_id)
            
            prompt = get_formated_prompt(query, user_id)
            timestamp = datetime.utcnow().isoformat()
            user_msg = HumanMessage(
                content=prompt,
                id=msg_id,
                metadata={"user_id": user_id,
                          "thread_id": thread_id,
                          "timestamp": timestamp}
            )

            # Combine hydrated history with new message
            messages_input = initial_messages + [user_msg]

            try:
                result = self.graph.invoke(
                    {"messages": messages_input, "system_message": self.sys_msg}, config)
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
                    
                    # Get updated summary from result
                    updated_summary = result.get("summary", existing_summary)
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
                    embedding=get_embedding(query) or [],
                    summary=updated_summary,
                    summarize_fn="None",
                    embed_summary_fn=get_embedding,
                    context_fn=[]
                )

                self.mongo_store.add_message(
                    user_id=user_id,
                    thread_id=thread_id,
                    role="assistant",
                    message_id=msg_id,
                    text=str(final_message) if not isinstance(final_message, str) else final_message,
                    embedding=get_embedding(str(final_message)) or [],
                    summary=updated_summary,
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

    async def get_stream_response(self, query: str, config: dict, user_id: str, thread_id: str, msg_id: str):
        try:
            # Check for existing state in LangGraph
            current_state = await self.graph.aget_state(config)
            initial_messages = []
            
            # If no state in Redis (new thread or forked thread), verify if we have history in Mongo
            if not current_state.values or not current_state.values.get("messages"):
                # logger.info(f"Hydrating state from Mongo for thread {thread_id}")
                mongo_msgs = self.mongo_store.get_thread_messages(user_id, thread_id)
                for m in mongo_msgs:
                    if m['role'] == 'user':
                        initial_messages.append(HumanMessage(content=m['text'], id=m['message_id']))
                    elif m['role'] in ['assistant', 'ai']:
                        initial_messages.append(AIMessage(content=m['text'], id=m['message_id']))

            # Get existing summary from MongoDB for this thread
            existing_summary = self.mongo_store.get_thread_summary(user_id, thread_id)
            
            prompt = get_formated_prompt(query, user_id)
            timestamp = datetime.utcnow().isoformat()
            user_msg = HumanMessage(
                content=prompt,
                id=msg_id,
                metadata={"user_id": user_id,
                          "thread_id": thread_id,
                          "timestamp": timestamp}
            )

            # Combine hydrated history with new message
            messages_input = initial_messages + [user_msg]
            
            full_response = ""
            updated_summary = existing_summary
            
            async for event in self.graph.astream_events(
                {"messages": messages_input, "system_message": self.sys_msg}, 
                config, 
                version="v1"
            ):
                kind = event["event"]
                
                # Yield tokens from the assistant's LLM
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        full_response += content
                        yield content
                
                # Check for summary updates in output of stream if possible, 
                # but usually summary is a separate node. 
                # If 'summurize' node runs, we might see its output in 'on_chain_end' or similar, 
                # but extracting it from stream events is tricky.
                # For now, we'll try to get state after stream to see if summary updated, 
                # OR just rely on the fact that summarize node updates the DB itself in the current implementation.
            
            # The 'summurize' node in existing code does: self.mongo_store.update_thread_summary(...)
            # So we don't need to manually save summary here if the node ran.
            # But we DO need to save the user message and the final assistant message.

            if not full_response:
                 logger.error("Agent produced no output in stream")
                 return

            try:
                self.mongo_store.add_message(
                    user_id=user_id,
                    thread_id=thread_id,
                    role="user",
                    text=query,
                    message_id=msg_id,
                    embedding=get_embedding(query) or [],
                    summary=updated_summary,
                    summarize_fn="None",
                    embed_summary_fn=get_embedding,
                    context_fn=[]
                )

                self.mongo_store.add_message(
                    user_id=user_id,
                    thread_id=thread_id,
                    role="assistant",
                    message_id=msg_id,
                    text=full_response,
                    embedding=get_embedding(full_response) or [],
                    summary=updated_summary,
                    summarize_fn="None",
                    embed_summary_fn=get_embedding,
                    context_fn=[]
                )
            except Exception as e:
                logger.error(f"getStreamResponse Convo save : {e}")

        except Exception as e:
            logger.error(f"get_stream_response : {e}")
            yield f"Error: {str(e)}"
