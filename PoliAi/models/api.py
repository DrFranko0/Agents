from gradio_client import Client

# Initialize the Hugging Face API client
client = Client("ysharma/Chat_with_Meta_llama3_8b")

def get_response(message, request=0.95, max_tokens=512):
    """
    Query the hosted Hugging Face LLM via the API.

    Args:
        message (str): User's query/message.
        request (float): Nucleus sampling parameter (default 0.95).
        max_tokens (int): Maximum number of tokens for the response (default 512).

    Returns:
        str: Generated response from the API.
    """
    try:
        # Call the API
        result = client.predict(
            message=message,
            request=request,
            param_3=max_tokens,
            api_name="/chat",
        )
        return result
    except Exception as e:
        return f"An error occurred while contacting the API: {e}"