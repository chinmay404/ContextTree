
from Agents.helpers.get_llm import LLM
from Agents.tools.tools import tools_list
from Agents.helpers.load_prompt import load_prompt_from_yaml
from langchain_core.messages import HumanMessage
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tool_node, tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from Agents.helpers.load_prompt import load_prompt_from_yaml
from Agents.DB_ops.main import MongoCRUD
from Agents.helpers.get_state_data import get_next_state, get_ideal_states
from datetime import datetime
from langgraph.checkpoint.mongodb import MongoDBSaver
from Agents.utils.graph_state import State
from Agents.utils.nodes import BaseAgentNodes
from Agents.utils.logger import logger
# asd

db = MongoCRUD()
Nodes = BaseAgentNodes()


class getGraphResponse():
    def __init__(self):
        logger.info("Initializing getGraphResponse..")
        self.llm = LLM.get_groq_llm()
        self.memory = MongoDBSaver(db.client)
        self.llm_with_tools = self.llm.bind_tools(tools_list)
        self.state = State
        self.nodes = Nodes.get_nodes()
        self.routes = Nodes.get_routes()
        self.graph = self.build_graph()
        self.sys_msg = load_prompt_from_yaml("REACT_LANGGRAPH_PROMPT")
        self.complex_querry_decison_prompt = load_prompt_from_yaml(
            "complex_querry_decison")
        self.simple_assistant_prompt = load_prompt_from_yaml(
            "simple_assistant")
        logger.info("getGraphResponse initialized.")

    def build_graph(self):
        logger.info("Building LangGraph..")
        builder = StateGraph(self.state)
        builder.add_node("complex_assistant", self.nodes["assistant"])
        builder.add_node("tools", ToolNode(tools_list))
        builder.add_node("simple_assistant", self.nodes["simple_assistant"])

        # Add edges
        builder.add_conditional_edges(
            START,
            self.nodes["complex_querry_decison"],
            {"True": "complex_assistant", "False": "simple_assistant"}
        )
        builder.add_edge("simple_assistant", END)

        builder.add_conditional_edges("complex_assistant", tools_condition)
        builder.add_edge("tools", "complex_assistant")
        builder.add_edge("complex_assistant", END)

        graph = builder.compile(checkpointer=self.memory)
        # draw_graph(graph)
        logger.info("LangGraph built successfully.")
        return graph

    def get_response(self, query: str, config: dict, user_id: str):
        now = datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d %H:%M:%S %A")
        try:
            logger.info(f"Received query: {query} | User ID: {user_id} | thread ID: {config['configurable']['thread_id']}")
            user_current_state = db.get_user_state(user_id)
            logger.info(f"User current state: {user_current_state}")
            current = user_current_state.get("current") if user_current_state else None

            next_state, next_state_details = get_next_state(country_id="Germany", current_state=current)
            logger.info(
                f"Next state: {next_state}, Details: {next_state_details}")

        except Exception as e:
            logger.error(f"Error fetching state: {e}", exc_info=True)
            current = None
            next_state, next_state_details = None, None
        # TODO :  Add Prper Formatting of message
        human_message = [HumanMessage(content="" + " USER ID : " + user_id + " User Current State : " + str(current)
                                      + " Next State and Details : " +
                                      str(next_state) +
                                      str(next_state_details) +
                                      " current date " +
                                      str(formatted_datetime)
                                      + "\n User query : " + query)]

        try:
            res = self.graph.invoke({"messages": human_message}, config)
        except Exception as e:
            logger.error(f"Error invoking graph: {e}", exc_info=True)
            return "Something went wrong during graph execution."

        messages = res.get("messages", [])
        for msg in reversed(messages):
            print("Message: ", msg)
            if hasattr(msg, 'content') and msg.content:
                # print("Message content: ", msg.content)
                final_message = msg.content
                # logger.info(f"Final message: {final_message}")
                return final_message

        logger.warning("No message content returned from graph.")
        return None
