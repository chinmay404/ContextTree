
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
import json
from app.agent.utils.saver import postgres_saver
from app.agent.store.PostgresStore import PostgresConversationStore
from app.agent.utils.embeddings import get_embedding
from app.agent.helpers.draw_graph import draw_graph
from app.core.logger import logger


from fastapi import APIRouter, HTTPException, status


class getGraphResponse():
    def __init__(self):
        self.memory = postgres_saver()
        if self.memory is None:
            raise MemoryError(
                "Failed to initialize PostgresSaver: memory is None")
        # self.memory = InMemorySaver()
        self.mongo_store = PostgresConversationStore()
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

            # --- RAG Context: similar messages + connected files ---
            query_embedding = get_embedding(query) or []
            context_snippets = []
            external_context_snippets = []

            if query_embedding:
                try:
                    # RAG Logic: Use ancestry to respect strict tree architecture
                    # Search messages in: Current Node + Ancestors + Global Files (if linked?)
                    # Wait, core doc says: "Ownership never flows backward or sideways"
                    # So we should search messages ONLY in the ancestry chain.
                    # Global search across all user history violates "No context leaks".
                    
                    ancestry_ids = self.mongo_store.get_thread_ancestry(thread_id)
                    # Convert to list of (id, None) tuples for the store method
                    thread_scope = [(aid, None) for aid in ancestry_ids]
                    
                    sims = self.mongo_store.find_similar_by_message_id(
                        user_id=user_id,
                        thread_queries=thread_scope,
                        query_embeddings=query_embedding,
                        top_k=3,
                    ) or []
                    for idx, s in enumerate(sims):
                        score = s.get("score")
                        role = s.get("role")
                        text = s.get("text", "")
                        msg_id = s.get("message_id")
                        score_str = f"{score:.3f}" if score is not None else "n/a"
                        context_snippets.append(
                            f"[sim {idx+1} | role={role} | score={score_str}] id={msg_id}: {text}"
                        )
                except Exception as e:
                    logger.error(f"Similarity search failed: {e}")

                try:
                    file_chunks = self.mongo_store.get_related_file_context(
                        node_id=thread_id,
                        query_embedding=query_embedding,
                        limit=3,
                    ) or []
                    for idx, c in enumerate(file_chunks):
                        meta = c.get("metadata")
                        try:
                            meta = json.loads(meta) if isinstance(meta, str) else (meta or {})
                        except Exception:
                            meta = meta or {}
                        external_context_snippets.append(
                            f"[file {idx+1} | {c.get('file_name')} | type={c.get('file_type')} | chunk={c.get('chunk_index')}] {c.get('chunk_text', '')}\nmeta: {meta}"
                        )
                except Exception as e:
                    logger.error(f"File context search failed: {e}")

            external_context_str = "\n".join(external_context_snippets) if external_context_snippets else None

            # --- RAG Context: similar messages + connected files ---
            query_embedding = get_embedding(query) or []
            context_snippets = []
            external_context_snippets = []

            if query_embedding:
                try:
                    sims = self.mongo_store.find_similar_by_message_id(
                        user_id=user_id,
                        thread_queries=[],
                        query_embeddings=query_embedding,
                        top_k=3,
                    ) or []
                    for idx, s in enumerate(sims):
                        score = s.get("score")
                        role = s.get("role")
                        text = s.get("text", "")
                        msg_id = s.get("message_id")
                        score_str = f"{score:.3f}" if score is not None else "n/a"
                        context_snippets.append(
                            f"[sim {idx+1} | role={role} | score={score_str}] id={msg_id}: {text}"
                        )
                except Exception as e:
                    logger.error(f"Similarity search failed: {e}")

                try:
                    file_chunks = self.mongo_store.get_related_file_context(
                        node_id=thread_id,
                        query_embedding=query_embedding,
                        limit=3,
                    ) or []
                    for idx, c in enumerate(file_chunks):
                        meta = c.get("metadata")
                        try:
                            meta = json.loads(meta) if isinstance(meta, str) else (meta or {})
                        except Exception:
                            meta = meta or {}
                        external_context_snippets.append(
                            f"[file {idx+1} | {c.get('file_name')} | type={c.get('file_type')} | chunk={c.get('chunk_index')}] {c.get('chunk_text', '')}\nmeta: {meta}"
                        )
                except Exception as e:
                    logger.error(f"File context search failed: {e}")

            external_context_str = "\n".join(external_context_snippets) if external_context_snippets else None
            
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
            
            # Deduplicate logic: If we hydrated from DB and the last message matches the current user message, don't append it again.
            is_duplicate = False
            if initial_messages:
                last_msg = initial_messages[-1]
                
                # Check for ID match or exact content match
                if last_msg.id == msg_id:
                    is_duplicate = True
                elif isinstance(last_msg.id, str) and isinstance(msg_id, str):
                    # Handle suffix differences (e.g. _u)
                    if last_msg.id.startswith(msg_id) or msg_id.startswith(last_msg.id):
                        is_duplicate = True
                    if last_msg.id.replace("_u", "") == msg_id.replace("_u", ""):
                        is_duplicate = True
                
                # Content fallback check
                if not is_duplicate and isinstance(last_msg, HumanMessage) and last_msg.content == prompt:
                     is_duplicate = True
                     
                if is_duplicate:
                    logger.info(f"Duplicate message detected in hydration: {msg_id}. Using DB version.")
                    messages_input = initial_messages

            try:
                result = self.graph.invoke(
                    {
                        "messages": messages_input, 
                        "system_message": self.sys_msg,
                        "summary": existing_summary,
                        "context": context_snippets,
                        "external_context": external_context_str,
                        "model": config.get("configurable", {}).get("model"),
                        "temperature": config.get("configurable", {}).get("temperature"),
                    }, 
                    config
                )
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

            if not is_duplicate:
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
                        message_id=f"{msg_id}_ai",
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
            try:
                current_state = await self.graph.aget_state(config)
            except NotImplementedError:
                # Fallback for sync checkpointers (e.g., PostgresSaver)
                current_state = self.graph.get_state(config)
            initial_messages = []

            # --- RAG Context: similar messages + connected files ---
            query_embedding = get_embedding(query) or []
            context_snippets: list[str] = []
            external_context_snippets: list[str] = []

            if query_embedding:
                try:
                    # Strict Ancestry Search
                    ancestry_ids = self.mongo_store.get_thread_ancestry(thread_id)
                    thread_scope = [(aid, None) for aid in ancestry_ids]
                    
                    sims = self.mongo_store.find_similar_by_message_id(
                        user_id=user_id,
                        thread_queries=thread_scope,
                        query_embeddings=query_embedding,
                        top_k=3,
                    ) or []
                    for idx, s in enumerate(sims):
                        score = s.get("score")
                        role = s.get("role")
                        text = s.get("text", "")
                        msg_id_sim = s.get("message_id")
                        score_str = f"{score:.3f}" if score is not None else "n/a"
                        context_snippets.append(
                            f"[sim {idx+1} | role={role} | score={score_str}] id={msg_id_sim}: {text}"
                        )
                except Exception as e:
                    logger.error(f"Similarity search failed: {e}")

                try:
                    file_chunks = self.mongo_store.get_related_file_context(
                        node_id=thread_id,
                        query_embedding=query_embedding,
                        limit=3,
                    ) or []
                    for idx, c in enumerate(file_chunks):
                        meta = c.get("metadata")
                        try:
                            meta = json.loads(meta) if isinstance(meta, str) else (meta or {})
                        except Exception:
                            meta = meta or {}
                        external_context_snippets.append(
                            f"[file {idx+1} | {c.get('file_name')} | type={c.get('file_type')} | chunk={c.get('chunk_index')}] {c.get('chunk_text', '')}\nmeta: {meta}"
                        )
                except Exception as e:
                    logger.error(f"File context search failed: {e}")

            external_context_str = "\n".join(external_context_snippets) if external_context_snippets else None
            
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
            
            # Deduplicate logic: If we hydrated from DB and the last message matches the current user message, don't append it again.
            is_duplicate = False
            if initial_messages:
                last_msg = initial_messages[-1]
                
                # Check for ID match or exact content match
                if last_msg.id == msg_id:
                    is_duplicate = True
                elif isinstance(last_msg.id, str) and isinstance(msg_id, str):
                    # Handle suffix differences (e.g. _u)
                    if last_msg.id.startswith(msg_id) or msg_id.startswith(last_msg.id):
                        is_duplicate = True
                    if last_msg.id.replace("_u", "") == msg_id.replace("_u", ""):
                        is_duplicate = True
                
                # Content fallback check
                if not is_duplicate and isinstance(last_msg, HumanMessage) and last_msg.content == prompt:
                     is_duplicate = True
                     
                if is_duplicate:
                    logger.info(f"Duplicate message detected in stream hydration: {msg_id}. Using DB version.")
                    messages_input = initial_messages
            
            full_response = ""
            updated_summary = existing_summary
            
            try:
                async for event in self.graph.astream_events(
                    {
                        "messages": messages_input, 
                        "system_message": self.sys_msg,
                        "summary": existing_summary,
                        "context": context_snippets,
                        "external_context": external_context_str,
                        "model": config.get("configurable", {}).get("model"),
                        "temperature": config.get("configurable", {}).get("temperature"),
                    }, 
                    config, 
                    version="v1"
                ):
                    kind = event["event"]
                    
                    # Yield tokens from the assistant's LLM
                    if kind == "on_chat_model_stream":
                        content = event["data"]["chunk"].content
                        if content:
                            full_response += content
                            yield f"data: {json.dumps({'message': content})}\n\n"
            except NotImplementedError:
                # Sync fallback when async streaming is unsupported by the checkpointer
                result = self.graph.invoke(
                    {
                        "messages": messages_input, 
                        "system_message": self.sys_msg,
                        "summary": existing_summary,
                        "context": context_snippets,
                        "external_context": external_context_str,
                        "model": config.get("configurable", {}).get("model"),
                        "temperature": config.get("configurable", {}).get("temperature"),
                    },
                    config
                )
                ai_messages = result.get("messages", []) if result else []
                final_message = None
                AI_RESPONSE_ = [
                    msg for msg in ai_messages if isinstance(msg, AIMessage)
                ]
                if AI_RESPONSE_:
                    final_message = AI_RESPONSE_[-1].content
                if final_message:
                    full_response = str(final_message)
                    yield f"data: {json.dumps({'message': full_response})}\n\n"
                updated_summary = result.get("summary", existing_summary) if result else existing_summary
                
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
                # Only save user message if it wasn't already in the DB (deduplicated)
                if not is_duplicate:
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
                    message_id=f"{msg_id}_ai",
                    text=full_response,
                    embedding=get_embedding(full_response) or [],
                    summary=updated_summary,
                    summarize_fn="None",
                    embed_summary_fn=get_embedding,
                    context_fn=[]
                )
            except Exception as e:
                logger.error(f"getStreamResponse Convo save : {e}")

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"get_stream_response : {repr(e)}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
