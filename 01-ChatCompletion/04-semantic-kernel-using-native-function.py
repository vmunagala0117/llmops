from dotenv import load_dotenv
import asyncio
import os
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import KernelArguments
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory
from typing import Annotated
from semantic_kernel.functions import kernel_function

# Load environment variables from .env file
load_dotenv()

# Get the plugin directory
current_dir = os.path.abspath("")
plugin_dir = os.path.join(current_dir, "plugins")

# Get Azure OpenAI credentials from environment
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# create a new native plugin


class LightsPlugin:
    lights = [
        {"id": 1, "name": "Table Lamp", "is_on": False},
        {"id": 2, "name": "Porch light", "is_on": False},
        {"id": 3, "name": "Chandelier", "is_on": True},
    ]

    @kernel_function(
        name="get_lights",
        description="Gets a list of lights and their current state",
    )
    def get_state(
        self,
    ) -> Annotated[str, "the output is a string"]:
        """Gets a list of lights and their current state."""
        return self.lights

    @kernel_function(
        name="change_state",
        description="Changes the state of the light",
    )
    def change_state(
        self,
        id: int,
        is_on: bool,
    ) -> Annotated[str, "the output is a string"]:
        """Changes the state of the light."""
        for light in self.lights:
            if light["id"] == id:
                light["is_on"] = is_on
                return light
        return None


async def main():
    # Check for missing environment variables
    if not deployment_name or not endpoint or not api_key:
        raise ValueError(
            "Missing necessary Azure OpenAI environment variables")

    # Initialize the kernel
    kernel = Kernel()
    kernel.remove_all_services()

    service_id = "native_func"

    chat_completion = AzureChatCompletion(
        deployment_name=deployment_name,
        api_key=api_key,
        endpoint=endpoint,
        api_version="2024-02-15-preview",
        service_id=service_id
    )
    kernel.add_service(chat_completion)

    # Load a Plugin and run a semantic function:
    kernel.add_plugin(
        LightsPlugin(),
        plugin_name="Lights"
    )

    # Enable planning
    execution_settings = kernel.get_prompt_execution_settings_from_service_id(
        service_id=service_id
    )
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto(
        filters={"included_plugins": ["Lights"]}
    )
    
    print(
        await kernel.invoke_prompt(
            function_name="prompt_test",
            plugin_name="light_test",
            prompt="Get me the list of all the lights",
            settings=execution_settings,
        )
    )
    
    # Create a history of the conversation
    history = ChatHistory()
    # Initiate a back-and-forth chat
    userInput = None
    while True:
        # Collect user input
        userInput = input("User > ")

        # Terminate the loop if the user says "exit"
        if userInput == "exit":
            break

        # Add user input to the history
        history.add_user_message(userInput)

        # Get the response from the AI
        result = await chat_completion.get_chat_message_content(
            chat_history=history,
            settings=execution_settings,
            kernel=kernel,
            arguments=KernelArguments(),
        )

        # Print the results
        print("Assistant > " + str(result))

        # Add the message from the agent to the chat history
        history.add_message(result)

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
