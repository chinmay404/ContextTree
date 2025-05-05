
from langgraph.graph import StateGraph, START, END
import re
import json

db = MongoCRUD()


class BaseAgentNodes:
    def __init__(self):
        self.llm = LLM.get_groq_llm()
        self.gemini = LLM.get_Gemini()
        self.memory = MongoDBSaver(db.client)
        self.llm_with_tools = self.llm.bind_tools(tools_list)
        self.state = State()
        self.sys_msg = load_prompt_from_yaml("REACT_LANGGRAPH_PROMPT")
        self.complex_querry_decison_prompt = load_prompt_from_yaml(
            "complex_querry_decison")
        self.simple_assistant_prompt = load_prompt_from_yaml(
            "simple_assistant")

    def assistant(self, state: MessagesState) -> AnyMessage:
        return {"messages": [self.llm_with_tools.invoke([self.sys_msg] + state["messages"])]}

    def summurize(self, state: MessagesState) -> AnyMessage:
        return {"messages": [self.llm.invoke([self.sys_msg] + state["messages"])]}

    def should_continue(self, state: MessagesState):
        """Return The Next Node To Execute"""
        messages = state["messages"]
        print("messages  lenght: ", len(messages))
        if len(messages) > 5:
            print("CALLED FOR SUMMURISATION")
            return True
        else:
            return END

    def complex_querry_decison(self, state: MessagesState) -> dict:
        """Decide Whether Complex Handling or Normal Response is Required"""
        try:
            response = self.llm.invoke(
                self.complex_querry_decison_prompt +
                state["messages"][-1].content
            )
            json_match = re.search(
                r'\{.*?"complex_querry_decison".*?\}', str(response), re.DOTALL)
            if json_match:
                decision_data = json.loads(json_match.group())
                value = decision_data.get("complex_querry_decison", False)
                print("complex_querry_decison: ", value)
                return "True" if value else "False"

            return "False"
        except Exception as e:
            print(f"Error in complex_querry_decison: {e}")
            return "False"

    def simple_assistant(self, state: MessagesState) -> AnyMessage:
        return {"messages": [self.gemini.invoke([self.simple_assistant_prompt] + state["messages"])]}

    def get_nodes(self) -> dict:
        return {
            "assistant": self.assistant,
            "summurize": self.summurize,
            "should_continue": self.should_continue,
            "complex_querry_decison": self.complex_querry_decison,
            "simple_assistant": self.simple_assistant,
        }
        
    def get_routes(self) -> dict:
        return {
            "complex_querry_decison_route" : self.complex_querry_decison,
        }