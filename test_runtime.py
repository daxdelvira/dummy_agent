import json
import asyncio
import uuid
import openai
from dataclasses import dataclass
from typing import List
from pydantic import BaseModel
from autogen_core import (
    AgentId,
    DefaultTopicId,
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
    AssistantMessage,
    UserMessage)

from autogen_core.tools import FunctionTool, Tool

from autogen_ext.models.openai import OpenAIChatCompletionClient

# Orchestrator to webnav agent message types


class initial_goal_message(BaseModel):
    content: UserMessage


class state_request_message(BaseModel):
    content: UserMessage

# Webnav agent to orchestrator message types


class webnav_tool_message(BaseModel):
    content: UserMessage


class webnav_state_message(BaseModel):
    content: UserMessage

# Webnav agent class definition

class webnav_agent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, nav_topic_type: str) -> None:
        super().__init__("A web navigator agent with tools")
        self._nav_topic_type = nav_topic_type
        self._system_message: List[LLMMessage] = [SystemMessage(content="You are a helpful web navigation assistant. When the state tracking agent publishes a task, please use any past actions you've taken and the tools available to you, please outline your thought process and select the next tool action to take to get closer to the goal. Only select one tool action.")]
        self._model_client = model_client
        self._chat_history: List[LLMMessage] = []
        self._obtain_website_tool = FunctionTool(self._obtain_website, name="_obtain_website_tool", description="Call this to navigate to a website via url.")
        self._click_tool = FunctionTool(self._click, name="_click_tool", description="Call this to click on an element by name.")
        self._scroll_tool = FunctionTool(self._scroll, name="_scroll_tool", description="Call this to scroll and see more of the webpage.")
        self._type_tool = FunctionTool(self._type, name="_type_tool", description="Call this to type the passed string in a specified field.")

    #Dummy obtain website tool
    async def _obtain_website(self, web_url: str) -> int:
        #Returns 1 if successful
        print(f"Ran: driver.get({web_url})")
        return f"Arrived at website {web_url}"

    #Dummy click tool
    async def _click(self, element: str) -> str:
        print(f"selected_element = driver.find_element(By.NAME,\"{element}\")")
        print("selected_element.click()")
        return f"Clicked on {element}"

    #Dummy scroll tool
    async def _scroll(self, distance: int) -> int:
        print(f"ActionChains(driver).scroll_by_amount(0,{distance}).perform()")
        return f"Scrolled by {distance}"

    #Dummy type tool
    async def _type(self, field_name:str, text:str) -> int:
        print(f"input_field = driver.find_element(By.Name, {field_name})")
        print("input_field.clear()")
        print(f"input_field.send_keys({text})")
        return f"Typed {text} into {field_name}"

    @message_handler
    async def handle_goal_message(self, message: initial_goal_message, ctx: MessageContext) -> None:
        self._chat_history.append(message.content)
        print("Message here")
        print(self._chat_history)
        print("Second Message")
        model_completion = await self._model_client.create(self._system_message + self._chat_history, tools=[self._obtain_website_tool, self._click_tool, self._scroll_tool, self._type_tool],)
        print(model_completion.content)
        assert isinstance(model_completion.content, list) and all(
            isinstance(item, FunctionCall) for item in model_completion.content
        )
        for tool_call in model_completion.content:
            tool_name = tool_call.name
            arguments = json.loads(tool_call.arguments)
            tool_result = await getattr(self, tool_name).run_json(arguments, ctx.cancellation_token)
        print("Goal message received")
        self._chat_history.append(webnav_tool_message(content=UserMessage(content=tool_result, source=self.id.type)))
        await self.publish_message(webnav_tool_message(content=UserMessage(content=tool_result, source=self.id.type)), topic_id=DefaultTopicId(type="nav"))


    @message_handler
    async def handle_state_request_message(self, message:state_request_message, ctx: MessageContext) -> None:
        print("State request message received")
        model_completion = await self._model_client.create([UserMessage(content="Please use your previous tool action history and current tool action to analyze the states relating to the task. Please return nothing besides the state variables in the following format, with the type value replaced with the description replaced with the correct value or NONE if the state is not applicable yet:{\n        'location_bar_clicked': 'value of True or False. True if the location bar has been clicked and false otherwise',\n        'location_type': 'value of True or False. True if the location has been typed into the location bar and false otherwise',\n        'location_value': 'The name you typed into the location bar, otherwise NONE',\n        'pop_up_present': 'value of True or False. True if a popup is on the screen, False otherwise.',\n        'check_in_month_clicked': 'value of True or False. True if the check in month has been clicked, false otherwise.',\n        'check_in_day_clicked': 'value of True or False. True if the check in day has been clicked, false otherwise.',\n        'check_out_month_clicked': 'value of True or False. True if the check out month has been clicked, false otherwise.'\n        'check_out_day_clicked': 'value of True or False. True if the check out day has been clicked, false otherwise.',\n        }\n [Format Ended]. The chat history can be found below.\n", source='state', type='UserMessage')]+self._chat_history)
        print(model_completion.content)
        await self.publish_message(webnav_state_message(content=UserMessage(content=model_completion.content, source=self.id.type)), topic_id=DefaultTopicID(type="state"))

# Orchestrator agent

class state_tracker_agent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("state_tracker_agent")
        self._state_history: List[LLMMessage] = []
        self._model_client = model_client
        self._system_message = "You are a state tracking orchestrator agent for a web navigation assistant."
        self._state_variables = """{
        'location_bar_clicked': 'value of True or False. True if the location bar has been clicked and false otherwise',
        'location_type': 'value of True or False. True if the location has been typed into the location bar and false otherwise',
        'location_value': 'The name you typed into the location bar, otherwise NONE',
        'pop_up_present': 'value of True or False. True if a popup is on the screen, False otherwise.',
        'check_in_month_clicked': 'value of True or False. True if the check in month has been clicked, false otherwise.',
        'check_in_day_clicked': 'value of True or False. True if the check in day has been clicked, false otherwise.',
        'check_out_month_clicked': 'value of True or False. True if the check out month has been clicked, false otherwise.'
        'check_out_day_clicked': 'value of True or False. True if the check out day has been clicked, false otherwise.',
        }"""
        self._goal_state = """{
        'location_bar_clicked': True,
        'location_type': True,
        'location_value': 'Jakarta, Indonesia',
        'pop_up_present': False,
        'check_in_month_clicked': True,
        'check_in_day_clicked': True,
        'check_in_month_clicked': True,
        'check_out_day_clicked': True,
        'check_out_month_clicked': True
        }"""

        self._STATE_REQUEST_MESSAGE = "Please use your previous tool action history and current tool action to anlyze the states relating to the task. If you don't have a chat history yet, use the states you infer from the booking taskMoUti given you haven't taken any action. No matter what, PLEASE return something in the correct format. Please return the state variables in the following format, with the type value replaced with the description replaced with the correct value or NONE if the state is ot applicable yet:" + self._state_variables

    @message_handler
    async def handle_webnav_tool_message(self, message:webnav_tool_message, ctx: MessageContext) -> None:
        print("Tool message received")
        await self.publish_message(state_request_message(content=UserMessage(content=self._STATE_REQUEST_MESSAGE, source=self.id.type)), topic_id=DefaultTopicId("state"))
        print(state_request_message(content=UserMessage(content=self._STATE_REQUEST_MESSAGE, source=self.id.type)))
        #await self.publish_message()
        # TODO: New prompt message

    @message_handler
    async def handle_webnav_state_message(self, message:webnav_state_message, ctx:MessageContext) -> None:
        self._state_history.append(message)
        print(self._state_history)

async def main():
    runtime = SingleThreadedAgentRuntime()

    state_tracking_topic_type = "state"
    web_navigation_topic_type = "nav"

    web_agent_type = await webnav_agent.register(
        runtime,
        "nav",
        lambda: webnav_agent(
            model_client=OpenAIChatCompletionClient(
                model="meta-llama/Llama-3.1-8B-Instruct",
                base_url="http://localhost:8000/v1/",
                api_key="EMPTY",
                model_info={
                    "vision": False,
                    "function_calling": True,
                    "json_output": True,
                    "family": "unknown",
                },
            ),
        nav_topic_type="nav",
        )
    )

    await runtime.add_subscription(TypeSubscription(topic_type="nav", agent_type=web_agent_type))
    await runtime.add_subscription(TypeSubscription(topic_type="state", agent_type=web_agent_type))

    state_agent_type = await state_tracker_agent.register(
        runtime,
        "state",
        lambda: state_tracker_agent(
            model_client=OpenAIChatCompletionClient(
                model="meta-llama/Llama-3.1-8b-Instruct",
                base_url="http://localhost:8000/v1/",
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
            content=UserMessage(content="Please use the tools at your disposal to complete the following task: 'Find the cheapest available hotel room for a three night   stay from 1st Jan in Jakarta. The room is for 2 adults, just answer the cheapest hotel room and the price.' Please explain your thought process and select one tool function to use.",
            source="User",
            )
        ),
        TopicId(type="nav", source="user"),
    )

    print("Message published")

    await runtime.stop_when_idle()

asyncio.run(main())
