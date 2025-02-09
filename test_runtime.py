import json
import openai
from dataclasses import dataclass
from Typing import List
from pydantic import BaseModel
from autogen_core import (
    AgentId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    message_handler,
    FunctionCall,
    TopicId,
    TypeSubscription
)

from autogen_core.models import (
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage
)

from autogen_core.tools import FunctionTool, Tool

# Orchestrator to webnav agent message types


@dataclass
class initial_goal_message(BaseModel):
    body: str


@dataclass
class state_request_message(BaseModel):
    body: str

# Webnav agent to orchestrator message types


@dataclass
class webnav_tool_message:
    body: str


@dataclass
class webnav_state_message:
    body: str

# Webnav agent class definition

class webnav_agent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        self.__init__("A web navigator agent with tools")
        self._system_message: List[LLMMessage] = [SystemMessage(content="You are a helpful web navigation assistant. When the state tracking agent publishes a task, please use any past actions you've taken and the tools available to you, please outline your thought process and select the next tool action to take to get closer to the goal. Only select one tool action.")]
        self._model_client = model_client
        self._chat_history: List[LLMMessage] = []
        self._obtain_website_tool = FunctionTool(self._obtain_website, name="obtain_website", description="Call this to navigate to a website via url.")
        self._click_tool = FunctionTool(self._click, name="click", description="Call this to click on an element by name.")
        self._scroll_tool = FunctionTool(self._scroll, name="scroll", description="Call this to scroll and see more of the webpage.")
        self._type_tool = FunctionTool(self._type, name="type", description="Call this to type the passed string in a specified field.")

    #Dummy obtain website tool
    async def _obtain_website(web_url:str) -> int:
        #Returns 1 if successful
        print(f"Ran: driver.get({web_url})")
        return 1

    #Dummy click tool
    async def _click(element:str) -> int:
        print(f"selected_element = driver.find_element(By.NAME,\"{element}\")")
        print("selected_element.click()")
        return 1

    #Dummy scroll tool
    async def _scroll(distance:int) -> int:
        print(f"ActionChains(driver).scroll_by_amount(0,{distance}).perform()")
        return 1

    #Dummy type tool
    async def _type(field_name:str, text:str) -> int:
        print(f"input_field = driver.find_element(By.Name, {field_name})")
        print("input_field.clear()")
        print(f"input_field.send_keys({text})")
        return 1

    @message_handler
    async def handle_goal_message(self, message: goal_message, ctx: MessageContext):
        self._chat_history.append("Message from Orchestrator", goal_message)
        model_completion = await self._model_client.create([self._system_message] + self._chat_history, tools=[self._obtain_website_tool, self._click_tool, self._scroll_tool, self._type_tool])
        print(model_completion.content)
        self._chat_history.append(AssistantMessage(content=model_completion.content, source=self.id.type))
        await self.publish_message(webnav_tool_message(body=AssistantMessage(content=model_completion.content, source=self.id.type)), topic_id=DefaultTopicId(type="nav"))

    @message_handler
    async def handle_state_request_message(self, message:state_request_message, ctx: MessageContext):
        model_completion = await self._model_client.create([state_request_message] + self._chat_history)
        print(model_completion.content)
        await self.publish_message(webnav_state_message(body=AssistantMessage(content=model_completion.content, source=self.id.type)), topic_id=DefaultTopicID(type="state"))

# Orchestrator agent

class orchestrator_agent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("orchestrator_agent")
        self._state_history: List[LLMMessage] = []
        self._model_client = model_client
        self._system_message = "You are a state tracking orchestrator agent for a web navigation assistant."
        self._state_variables = "{
        'location_bar_clicked': 'value of True or False. True if the location bar has been clicked and false otherwise',
        'location_type': 'value of True or False. True if the location has been typed into the location bar and false otherwise',
        'location_value': 'The name you typed into the location bar, otherwise NONE',
        'pop_up_present': 'value of True or False. True if a popup is on the screen, False otherwise.',
        'check_in_month_clicked': 'value of True or False. True if the check in month has been clicked, false otherwise.',
        'check_in_day_clicked': 'value of True or False. True if the check in day has been clicked, false otherwise.',
        'check_out_month_clicked': 'value of True or False. True if the check out month has been clicked, false otherwise.'
        'check_out_day_clicked': 'value of True or False. True if the check out day has been clicked, false otherwise.',
        }"
        self._goal_state = "{
        'location_bar_clicked': True,
        'location_type': True,
        'location_value': 'Jakarta, Indonesia',
        'pop_up_present': False,
        'check_in_month_clicked': True,
        'check_in_day_clicked': True,
        'check_in_month_clicked': True,
        'check_out_day_clicked': True,
        'check_out_month_clicked': True
        }"

        self._STATE_REQUEST_MESSAGE = "Please use your previous tool action history and current tool action to anlyze the states relating to the task. Please return the state variables in the following format, with the type value replaced with the description replaced with the correct value or NONE if the state is ot applicable yet:" + self._state_variables

    @message_handler
    async def handle_webnav_tool_message(self, message:webnav_tool_message, ctx: MessageContext) -> None:
        await self.publish_message(state_request_message(body=self._STATE_REQUEST_MESSAGE), topic_id="state")
publish_message() # TODO: New prompt message

    @message_handler
    async def handle_webnav_state_message(self, message:webnav_state_message, ctx:MessageContext) -> None:
        self._state_history.append(message)


runtime = SingleThreadedAgentRuntime()

state_tracking_topic_type = "state"
web_navigation_topic_type = "nav"

web_agent_type = await web_agent.register(
    runtime,
    "web_nav_agent",
    lambda: webnav_agent(
        model_client=OpenAIChatCompletionClient(
            model="meta-llama/Llama-3.1-8B-Instruct",
            base_url="http://localhost:8000/v1",
            api_key="placeholder",
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "unknown",
            },
        )
    )
)

await runtime.add_subscription(TypeSubscription(topic_type="nav", agent_type=web_agent_type))
await runtime.add_subscription(TypeSubscription(topic_type="state", agent_type=web_agent_type))

state_agent_type = await state_agent.register(
    runtime,
    "state_tracker_agent",
    lambda: state_tracker_agent(
        model_client=OpenAIChatCompletionClient(
            model="meta-llama/Llama-3.1-8B-Instruct",
            base_url="http://localhost:8000/v1",
            api_key="placeholder",
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "unknown",
            },
        )
    )
)

await runtime.add_subscription(TypeSubscription(topic_type="nav", agent_type=state_agent_type))
await runtime.add_subscription(TypeSubscription(topic_type="state", agent_type=state_agent_type))

runtime.start()
session_id = str(uuid.uuid4())
await runtime.publish_message(
    initial_goal_message(
        body=AssistantMessage(
            content="Please use the tools at your disposal to complete the following task: 'Find the cheapest available hotel room for a three night   stay from 1st Jan in Jakarta. The room is for 2 adults, just answer the cheapest hotel room and the price.' Please explain your thought process and select one tool function to use.",
            source="state_tracker_agent",
        )
    ),
    TopicId(type="nav", source=session_id),
)

await runtime.stop_when_idle()
