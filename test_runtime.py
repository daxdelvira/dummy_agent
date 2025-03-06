import csv
import json
import sys
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

class state_correction_message(BaseModel):
    content: UserMessage

class retry_message(BaseModel):
    content: UserMessage

# Webnav agent to orchestrator message types


class webnav_tool_message(BaseModel):
    content: UserMessage


class webnav_state_message(BaseModel):
    content: UserMessage


def add_task_by_id(task_id:str):
        with open("test_tasks.json", "r") as task_file:
            tasks = json.load(task_file)

        for task in tasks:
            if task["id"] == task_id:
                return task

selected_task = add_task_by_id("Coursera--0")

# Webnav agent class definition

class webnav_agent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, nav_topic_type: str) -> None:
        super().__init__("A web navigator agent with tools")
        self._nav_topic_type = nav_topic_type
        self._system_message: List[LLMMessage] = [SystemMessage(content="You are a helpful web navigation assistant. When the state tracking agent publishes a task, please use any past actions you've taken, your state history, and the tools available to you, please outline your thought process and select the next tool action to take to get closer to the goal. Only select one tool action.")]
        self._model_client = model_client
        self._chat_history: List[LLMMessage] = []
        self._obtain_website_tool = FunctionTool(self._obtain_website, name="_obtain_website_tool", description="Call this to navigate to a website via url.")
        self._click_tool = FunctionTool(self._click, name="_click_tool", description="Call this to click on an element by name.")
        self._scroll_tool = FunctionTool(self._scroll, name="_scroll_tool", description="Call this to scroll and see more of the webpage.")
        self._type_tool = FunctionTool(self._type, name="_type_tool", description="Call this to type the passed string in a specified field.")
        self._tool_call_count = 0
        self._state_history: List[LLMMessage] = []
        self._prev_state: List[LLMMessage] = []
        self._state_intro: List[LLMMessage] = []

    #Dummy obtain website tool
    async def _obtain_website(self, web_url: str) -> int:
        #Returns 1 if successful
        print(f"Ran: driver.get({web_url})")
        return f"Arrived at website {web_url}, successfully, time for the next step"

    #Dummy click tool
    async def _click(self, element: str) -> str:
        print(f"selected_element = driver.find_element(By.NAME,\"{element}\")")
        print("selected_element.click()")
        return f"Clicked on {element} successfully, time for the next step"

    #Dummy scroll tool
    async def _scroll(self, distance: int) -> int:
        print(f"ActionChains(driver).scroll_by_amount(0,{distance}).perform()")
        return f"Scrolled by {distance}, successfully found the element you were looking for. Please click to select."

    #Dummy type tool
    async def _type(self, field_name:str, text:str) -> int:
        print(f"input_field = driver.find_element(By.Name, {field_name})")
        print("input_field.clear()")
        print(f"input_field.send_keys({text})")
        return f"Typed {text} into {field_name} successfully, time for the next step"

    @message_handler
    async def handle_goal_message(self, message: initial_goal_message, ctx: MessageContext) -> None:
        print("Handling Goal Message. . .\n")
        print("Received:\n", message.content, "\n")
        needRetry = True
        while needRetry:
            try:
                model_completion = await self._model_client.create([message.content] + self._chat_history + self._state_intro + self._prev_state, tools=[self._obtain_website_tool, self._click_tool, self._scroll_tool, self._type_tool],)
                assert isinstance(model_completion.content, list) and all(
                    isinstance(item, FunctionCall) for item in model_completion.content
                    )
                needRetry = False
            except AssertionError as e:
                print("Assertion error", e)
                needRetry = True
            

            for tool_call in model_completion.content:
                print("Executing tool call: \n", tool_call, "\n")
                tool_name = tool_call.name
                arguments = json.loads(tool_call.arguments)
                try:
                    tool_result = await getattr(self, tool_name).run_json(arguments, ctx.cancellation_token)
                    self._tool_call_count += 1
                    print("Tool call iteration: ", self._tool_call_count)
                except AttributeError:
                    tool_result = "Invalid tool name, try again."
                    needRetry = True
            print("Tool result: \n", tool_result, "\n")
        self._chat_history.append(UserMessage(content=tool_result, source=self.id.type))
        await self.publish_message(webnav_tool_message(content=UserMessage(content=tool_result, source=self.id.type)), topic_id=DefaultTopicId(type="nav"))


    @message_handler
    async def handle_state_request_message(self, message:state_request_message, ctx: MessageContext) -> None:
        print("State request message received\n")
        model_completion = await self._model_client.create([message.content]+self._chat_history)
        self._state_history.append(UserMessage(content=model_completion.content, source=self.id.type))
        self._prev_state = [UserMessage(content=model_completion.content, source=self.id.type)]
        self._prev_state.append(UserMessage(content=model_completion.content, source=self.id.type))
        await self.publish_message(webnav_state_message(content=UserMessage(content=model_completion.content, source=self.id.type)), topic_id=DefaultTopicId(type="state"))

    @message_handler
    async def handle_retry_message(self, message:retry_message, ctx: MessageContext) -> None:
        print("Retry message received\n")
        self._chat_history.append(message.content)
        needRetry = True
        while needRetry:
            try:
                model_completion = await self._model_client.create(self._system_message + [message.content] + self._chat_history, tools=[self._obtain_website_tool, self._click_tool, self._scroll_tool, self._type_tool],)
                assert isinstance(model_completion.content, list) and all(
                    isinstance(item, FunctionCall) for item in model_completion.content
                    )
                needRetry = False
            except AssertionError:
                print("Assertion error")
                needRetry = True
            

            for tool_call in model_completion.content:
                print("Executing tool call: \n", tool_call, "\n")
                tool_name = tool_call.name
                arguments = json.loads(tool_call.arguments)
                try:
                    tool_result = await getattr(self, tool_name).run_json(arguments, ctx.cancellation_token)
                    self._tool_call_count += 1
                    print("Tool call iteration: ", self._tool_call_count)
                except AttributeError:
                    tool_result = "Invalid tool name, try again."
                    needRetry = True
            print("Tool result: \n", tool_result, "\n")
        
        self._chat_history.append(UserMessage(content=tool_result, source=self.id.type))
        await self.publish_message(webnav_tool_message(content=UserMessage(content=tool_result, source=self.id.type)), topic_id=DefaultTopicId(type="nav"))

    @message_handler
    async def handle_state_correction_message(self, message:state_correction_message, ctx: MessageContext) -> None:
        print("State correction message received\n")
        old_state = self._state_history.pop()
        self._state_history.append(message.content)
        self._prev_state=[message.content]

# Orchestrator agent

class state_tracker_agent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient, intervention_interval: int) -> None:
        super().__init__("state_tracker_agent")
        self._state_history: List[LLMMessage] = []
        self._model_client = model_client
        self._system_message = "You are a state tracking orchestrator agent for a web navigation assistant."
        self._state_variables = json.dumps(selected_task["state_variables"], indent=4)
        self._goal_state = selected_task["goal_state_variables"]
        self._current_state = None
        self._prev_state = None
        self._last_tool_call = None
        self._intervention_interval = intervention_interval
        self._iter_count = 0

        self._STATE_REQUEST_MESSAGE="""Analyze the state variables based on your previous tool actions and the current tool action.

- Use your tool action history to infer which state variables have changed.
- If a state variable has been updated, return its new value.
- For Boolean state variables, return true or false. DO NOT RETURN A STRING.
- For String state variables, return a string value. DO NOT RETURN A BOOLEAN.
- If a state variable is not applicable yet, return None.

Return only the state variables in the following JSON format, with each description replaced by its correct value:

{
    "state_variable_1": correct_value_or_NONE,
    "state_variable_2": correct_value_or_NONE,
    ...
}

Do not include any explanations, reasoning, or additional text—only the correctly formatted JSON output.""" + self._state_variables

    def count_matching_pairs(self, json1, json2):
        """
        Compares two JSON objects and returns the number of matching key-value pairs.
        """
        matches = sum(1 for key in json1 if key in json2 and json1[key] == json2[key])
        return matches

    def all_pairs_exist(self, json1, json2):
        """
        Checks if all key-value pairs in json1 exist in json2.
        """
        return all(key in json2 and json1[key] == json2[key] for key in json1)

    @message_handler
    async def handle_webnav_tool_message(self, message:webnav_tool_message, ctx: MessageContext) -> None:
        print("Tool message received\n")
        self._last_tool_call = message.content.content
        await self.publish_message(state_request_message(content=UserMessage(content=self._STATE_REQUEST_MESSAGE, source=self.id.type)), topic_id=DefaultTopicId("state"))
        #await self.publish_message()
        # TODO: New prompt message

    @message_handler
    async def handle_webnav_state_message(self, message: webnav_state_message, ctx: MessageContext) -> None:
        print("State message received\n")
        self._iter_count += 1

        try:
            # Attempt to parse JSON
            print("Loading in message. . .")
            self._current_state = json.loads(message.content.content)
            
            print("Previous state: ", self._prev_state)
            print("Current state: ", self._current_state)
            print("Last Tool Call Made:", self._last_tool_call)
            print("Message content: ", message.content.content)

            try:
                if self.all_pairs_exist(self._goal_state, self._current_state):
                    print("Goal state reached")
                    return
            except Exception as e:
                print(f"Exception on pair check: {e}")

            if self._iter_count % self._intervention_interval == 0:
                approval = input("Is the state correct? (y/n): ")
                if approval == "y":
                    await self.publish_message(
                        initial_goal_message(
                            content=UserMessage(
                                content="Please select the next tool action for the task. As a reminder, the task is: " + selected_task["system_message"] + "Your previous actions are: ",
                                source=self.id.type
                            )
                        ),
                        topic_id=DefaultTopicId("nav")
                    )
                else:
                    complete = input("Is the task complete?")
                    if complete == "y":
                        print("Goal state reached")
                        return
                    mistake = input("What was the mistake? (action/state): ")
                    if mistake == "action":
                        correct_state = input("What should the state be? (JSON format): ")
                        await self.publish_message(
                            retry_message(
                                content=UserMessage(
                                    content="This is not the correct tool call, please run a tool call so the state becomes this:" + correct_state,
                                    source=self.id.type
                                )
                            ),
                            topic_id=DefaultTopicId("retry")
                        )

                    elif mistake == "state":
                        correct_state = input("What should the state be? (JSON format): ")
                        await self.publish_message(
                            state_correction_message(
                                content=UserMessage(
                                    content=correct_state,
                                    source=self.id.type
                                )
                            ),
                            topic_id=DefaultTopicId("state_correction")
                        )
                        
                        await self.publish_message(
                        initial_goal_message(
                            content=UserMessage(
                                content="Please select the next tool action for the task. As a reminder, the task is: " + selected_task["system_message"],
                                source=self.id.type
                            )
                        ),
                        topic_id=DefaultTopicId("nav")
                    )

        except json.JSONDecodeError:
            print("Invalid JSON format")
            await self.publish_message(
                state_request_message(
                    content=UserMessage(
                        content="This was not correctly formatted JSON, please try to complete the following task again" 
                        + self._STATE_REQUEST_MESSAGE, 
                        source=self.id.type
                    )
                ), 
                topic_id=DefaultTopicId("state")
            )

        
#Agent instances to store graphs
agent_instances = {}        

async def main():
    iter_counts = []
    intervention_interval = int(input("What's the intervention interval?"))

    for run in range(10):
        runtime = SingleThreadedAgentRuntime()

        state_tracking_topic_type = "state"
        web_navigation_topic_type = "nav"
        retry_topic_type = "retry"
        state_correction_topic_type = "state_correction"

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
        await runtime.add_subscription(TypeSubscription(topic_type="retry", agent_type=web_agent_type))
        await runtime.add_subscription(TypeSubscription(topic_type="state_correction", agent_type=web_agent_type))

        #register state tracker agent in global agent_instances
        async def state_agent_factory():
            agent_instance = state_tracker_agent(
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
                ),
                intervention_interval=intervention_interval,
            )
            agent_instances["state_tracker_agent"] = agent_instance
            return agent_instance

        #Stick the factory function in the register function
        state_agent_type = await state_tracker_agent.register(
            runtime,
            "state",
            state_agent_factory
        )


        await runtime.add_subscription(TypeSubscription(topic_type="nav", agent_type=state_agent_type))
        await runtime.add_subscription(TypeSubscription(topic_type="state", agent_type=state_agent_type))
        await runtime.add_subscription(TypeSubscription(topic_type="retry", agent_type=state_agent_type))
        await runtime.add_subscription(TypeSubscription(topic_type="state_correction", agent_type=state_agent_type))

        runtime.start()
        session_id = str(uuid.uuid4())
    
        goal_state = json.dumps(selected_task["goal_state_variables"], indent=4)
   
        await runtime.publish_message(
            initial_goal_message(
                content=UserMessage(content=selected_task["system_message"] + "Remember the eventual goal state we want to reach is represented as follows:" + goal_state,
                source="orchestrator_agent",
                )
            ),
            TopicId(type="nav", source="user"),
        )

        print("Message published")

        await runtime.stop_when_idle()
    
        state_agent = agent_instances["state_tracker_agent"]
        iter_counts.append(state_agent._iter_count)

        #Clean up for next run
        del agent_instances["state_tracker_agent"]
        runtime = None
        await asyncio.sleep(2)

    csv_file = "iteration_count_results.csv"
    with open(csv_file, mode="a+",newline="") as file:
        writer = csv.writer(file)
         # Check if the file is empty to write the header first
        file.seek(0)  # Move to the start of the file
        if not file.read(1):  # If file is empty, write the header
            writer.writerow(["experiment_type"] + [f"Run{i}" for i in range(1, 11)])
        
        experiment_name = input("Write the experiment name:")
        writer.writerow([experiment_name] + iter_counts)


asyncio.run(main())
