from azure.identity import ClientSecretCredential
from plugins.BookingPlugin.bookings import BookingsPlugin
from msgraph import GraphServiceClient
from dotenv import load_dotenv
import asyncio
import os
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import KernelArguments
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions import kernel_function


# https://github.com/microsoft/semantic-kernel/blob/main/python/samples/demos/booking_restaurant/README.md

# Load environment variables from .env file
load_dotenv()

# Get the plugin directory
current_dir = os.path.abspath("")
plugin_dir = os.path.join(current_dir, "plugins")

# Get Azure OpenAI credentials from environment
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

tenant_id = os.getenv("BOOKING_SAMPLE_TENANT_ID")
client_id = os.getenv("BOOKING_SAMPLE_CLIENT_ID")
client_secret = os.getenv("BOOKING_SAMPLE_CLIENT_SECRET")
client_secret_credential = ClientSecretCredential(
    tenant_id=tenant_id,
    client_id=client_id,
    client_secret=client_secret)

graph_client = GraphServiceClient(
    credentials=client_secret_credential,
    scopes=["https://graph.microsoft.com/.default"])

booking_business_id = os.getenv("BOOKING_SAMPLE_BUSINESS_ID")
booking_service_id = os.getenv("BOOKING_SAMPLE_SERVICE_ID")

bookings_plugin = BookingsPlugin(
    graph_client=graph_client,
    booking_business_id=booking_business_id,
    booking_service_id=booking_service_id,
)

kernel = Kernel()
kernel.remove_all_services()

service_id = "booking_svc"
kernel.add_service(AzureChatCompletion(
    deployment_name=deployment_name,
    api_key=api_key,
    endpoint=endpoint,
    api_version="2024-02-15-preview",
    service_id=service_id
)
)
kernel.add_plugin(bookings_plugin, "BookingsPlugin")

chat_function = kernel.add_function(
    plugin_name="ChatBot",
    function_name="Chat",
    prompt="{{$chat_history}}{{$user_input}}",
    template_format="semantic-kernel",
)

# Enable planning
execution_settings = kernel.get_prompt_execution_settings_from_service_id(
    service_id=service_id
)
execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto(
    filters={"excluded_plugins": ["ChatBot"]}
)

chat_history = ChatHistory(
    system_message="When responding to the user's request to book a table, include the reservation ID."
)


async def chat() -> bool:
    try:
        user_input = input("User:> ")
    except KeyboardInterrupt:
        print("\n\nExiting chat...")
        return False
    except EOFError:
        print("\n\nExiting chat...")
        return False

    if user_input == "exit":
        print("\n\nExiting chat...")
        return False

    # Note the reservation returned contains an ID. That ID can be used to cancel the reservation,
    # when the bookings API supports it.
    answer = await kernel.invoke(
        chat_function, KernelArguments(settings=execution_settings, user_input=user_input, chat_history=chat_history)
    )
    chat_history.add_user_message(user_input)
    chat_history.add_assistant_message(str(answer))
    print(f"Assistant:> {answer}")
    return True


async def main() -> None:
    chatting = True
    print(
        "Welcome to your Restaurant Booking Assistant.\
        \n  Type 'exit' to exit.\
        \n  Please enter the following information to book a table: the restaurant, the date and time, \
        \n the number of people, your name, phone, and email. You may ask me for help booking a table, \
        \n listing reservations, or cancelling a reservation. When cancelling please provide the reservation ID."
    )
    while chatting:
        chatting = await chat()


if __name__ == "__main__":
    asyncio.run(main())
