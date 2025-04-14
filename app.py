import streamlit as st
import boto3
import json
import os
import asyncio
from typing import Dict, List, Any, Optional
import pandas as pd
from get_prices import get_bedrock_prices

BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
PRICE_FILE = "bedrock_prices.json"


def load_pricing_data(region: str) -> Dict:
    """
    Load model pricing data from cache file or fetch fresh data.

    Args:
        region (str): AWS region to get pricing data for

    Returns:
        Dict: Dictionary containing pricing information
    """
    if os.path.exists(PRICE_FILE):
        try:
            with open(PRICE_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            # Handle corrupted cache file
            pass

    # Fetch fresh pricing data
    prices = get_bedrock_prices(region)

    # Cache the pricing data
    with open(PRICE_FILE, "w") as f:
        json.dump(prices, f)

    return prices


def extract_token_pricing(prices: Dict, region: str, model_name: str) -> Dict[str, float]:
    """
    Extract token pricing information for a specific model.

    Args:
        prices (Dict): The pricing data dictionary
        region (str): AWS region
        model_name (str): Name of the model

    Returns:
        Dict[str, float]: Dictionary with input and output token prices
    """
    model_pricing = prices.get(region, {}).get(model_name, {})

    return {
        "input": float(model_pricing.get("input", -1)),
        "output": float(model_pricing.get("output", -1))
    }


def calculate_model_price(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate the price of a model given its name, input token count, and output token count.

    Args:
    - model_name (str): The name of the model.
    - input_tokens (int): The number of input tokens.
    - output_tokens (int): The number of output tokens.

    Returns:
    - float: The total cost of using the model.
    """
    # Load pricing data
    prices = load_pricing_data(BEDROCK_REGION)

    # Get token pricing for the specific model
    token_pricing = extract_token_pricing(prices, BEDROCK_REGION, model_name)

    try:
        # Calculate the cost
        input_cost = (input_tokens / 1000) * token_pricing["input"]
        output_cost = (output_tokens / 1000) * token_pricing["output"]
        total_cost = input_cost + output_cost

        return total_cost
    except (TypeError, KeyError):
        return None

# Set page configuration
st.set_page_config(
    page_title="Bedrock Model Comparison",
    page_icon="🧠",
    layout="wide",
)

# Error handling decorator
def handle_error(func):
    """Decorator for error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    return wrapper

# Initialize Bedrock client
@st.cache_resource
def get_bedrock_clients():
    try:
        bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name=BEDROCK_REGION
        )
        bedrock = boto3.client(
            service_name="bedrock",
            region_name=BEDROCK_REGION
        )
        return bedrock_runtime, bedrock
    except Exception as e:
        st.error(f"Failed to initialize Bedrock clients: {str(e)}")
        return None, None

# Get available models from Bedrock API
@st.cache_data(ttl=3600)  # Cache for one hour
def get_available_models(_bedrock_client):
    bedrock_client = _bedrock_client
    try:
        # Get foundation models
        response = bedrock_client.list_foundation_models()

        # Filter for models that support the converse API and are available
        available_models = {}
        model_name_mapping = {}
        for model in response.get('modelSummaries', []):
            model_id = model.get('modelId')
            model_name_mapping[model_id] = model.get("modelName")

            if (model.get('inferenceTypesSupported') and
                'ON_DEMAND' in model.get('inferenceTypesSupported') and
                model.get('outputModalities') and
                'TEXT' in model.get('outputModalities') and
                model.get('inputModalities') and
                'TEXT' in model.get('inputModalities')):

                # Extract model name for display
                provider = model_id.split('.')[0].capitalize()
                display_name = f"{provider} {model_id}"
                display_name = f"{model_id}"

                available_models[display_name] = model_id

        # Also get the custom inference profiles
        try:
            profiles_response = bedrock_client.list_inference_profiles()
            for profile in profiles_response.get('inferenceProfileSummaries', []):
                profile_id = profile.get('inferenceProfileId')
                inference_profile_name = profile.get('inferenceProfileName')

                # Create a display name for the profile
                if inference_profile_name:
                    display_name = f"Profile: {inference_profile_name}"
                else:
                    # If no name is provided, use the ID
                    display_name = f"Profile: {profile_id}"

                # Add to available models
                available_models[display_name] = profile_id
        except Exception as e:
            st.warning(f"Failed to load custom inference profiles: {str(e)}")

        return available_models, model_name_mapping
    except Exception as e:
        st.error(f"Failed to get available models: {str(e)}")
        # Fallback to a minimal set of models
        return {}, {}

async def invoke_bedrock_model(client, model_id: str, prompt: str, max_retries=3, retry_delay=1) -> Dict[str, Any]:
    """Invoke Bedrock model with retry logic for throttling"""
    retry_count = 0

    while retry_count <= max_retries:
        try:
            # Create an async wrapper for the synchronous boto3 client call
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.converse(
                    modelId=model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": [{"text": prompt}]
                        }
                    ]
                )
            )

            # Extract the response text
            output = response.get("output", {})
            message = output.get("message", {})
            content = message.get("content", [{}])[0]
            response_text = content.get("text", "No response generated")

            # Extract usage metrics including separate input and output token counts
            usage = response.get("usage", {})
            input_tokens = usage.get("inputTokens", 0)
            output_tokens = usage.get("outputTokens", 0)
            total_tokens = usage.get("totalTokens", 0)

            return {
                "model_id": model_id,
                "response_text": response_text,
                "finish_reason": output.get("stopReason", ""),
                "latency_ms": response.get("metrics", {}).get("latencyMs", 0),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            }

        except Exception as e:
            if hasattr(e, "__class__") and e.__class__.__name__ == "ThrottlingException":
                retry_count += 1
                if retry_count <= max_retries:
                    await asyncio.sleep(retry_delay * 2 ** retry_count)  # Exponential backoff
                else:
                    return {
                        "model_id": model_id,
                        "response_text": "Request was throttled after multiple retries",
                        "error": "ThrottlingException",
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0
                    }
            else:
                return {
                    "model_id": model_id,
                    "response_text": f"Error: {str(e)}",
                    "error": str(e),
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0
                }

async def process_models_parallel(bedrock_client, prompt: str, selected_model_ids: List[str]) -> List[Dict[str, Any]]:
    """Process multiple model invocations in parallel using asyncio"""
    # Create tasks for each model invocation
    tasks = []
    for model_id in selected_model_ids:
        tasks.append(invoke_bedrock_model(bedrock_client, model_id, prompt))

    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    return results

@handle_error
def main():
    st.title("🧠 Amazon Bedrock Model Comparison")

    # Initialize session state if needed
    if "results" not in st.session_state:
        st.session_state.results = []

    # Get Bedrock clients
    bedrock_runtime, bedrock = get_bedrock_clients()
    if not bedrock_runtime or not bedrock:
        st.error("Failed to initialize AWS Bedrock clients. Please check your credentials and try again.")
        return

    # Get available models from API
    with st.spinner("Loading available models..."):
        model_options, model_name_mapping = get_available_models(bedrock)

    # Sidebar for model selection
    with st.sidebar:
        st.header("Settings")

        # Default to models containing "nova-pro" or "sonnet" in the name
        default_models = [model for model in model_options.keys()
                          if "nova-pro" in model.lower() or "sonnet" in model.lower()]

        # If no matching models found, default to empty list
        if not default_models:
            default_models = []

        st.markdown("""
        <style>
        .stMultiSelect [data-baseweb="select"] span {
            max-width: none !important; /* Remove width restriction */
            white-space: normal !important; /* Allow multiline wrapping */
            overflow: visible !important; /* Prevent hidden overflow */
            text-overflow: clip !important; /* Disable ellipsis truncation */
        }
        </style>
        """, unsafe_allow_html=True)

        selected_models = st.multiselect(
            "Select models to compare",
            options=list(model_options.keys()),
            default=default_models,
        )


        selected_model_ids = [model_options[model_name] for model_name in selected_models]

        st.divider()
        st.markdown("### About")
        st.markdown(
            "This app allows you to compare outputs from different "
            "Amazon Bedrock models for the same prompt."
        )

    # Main content area
    col1, col2 = st.columns([3, 1])

    with col1:
        prompt = st.text_area(
            "Enter your prompt:",
            height=200,
            placeholder="Type your prompt here..."
        )

    with col2:
        st.markdown("### Instructions")
        st.markdown(
            """
            1. Select models from the sidebar
            2. Enter your prompt in the text area
            3. Click 'Compare Models' to see results
            """
        )

    # Submit button
    if st.button("Compare Models", type="primary", disabled=not (prompt and selected_models)):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
            return

        if not selected_models:
            st.warning("Please select at least one model.")
            return

        # Show progress while processing
        with st.spinner("Processing your prompt across selected models..."):
            # Process the models in parallel using asyncio
            model_results = asyncio.run(process_models_parallel(
                bedrock_runtime,
                prompt,
                selected_model_ids
            ))

            # Store results
            st.session_state.results = model_results

    # Display results
    if st.session_state.results:
        st.header("Model Comparison Results")

        # Prepare data for display
        display_data = []

        for result in st.session_state.results:
            model_id = result.get('model_id', 'Unknown')
            model_name = model_name_mapping.get(model_id, None)

            # Parse token values, ensuring they're integers
            input_tokens = result.get('input_tokens', 'N/A')
            if isinstance(input_tokens, tuple):
                input_tokens = input_tokens[0]
            if not isinstance(input_tokens, int) and input_tokens != 'N/A':
                try:
                    input_tokens = int(input_tokens)
                except (ValueError, TypeError):
                    input_tokens = 0

            output_tokens = result.get('output_tokens', 'N/A')
            if isinstance(output_tokens, tuple):
                output_tokens = output_tokens[0]
            if not isinstance(output_tokens, int) and output_tokens != 'N/A':
                try:
                    output_tokens = int(output_tokens)
                except (ValueError, TypeError):
                    output_tokens = 0

            # Calculate price if we have valid token counts and model name
            if model_name and isinstance(input_tokens, int) and isinstance(output_tokens, int):
                price = calculate_model_price(model_name, input_tokens, output_tokens)
            else:
                price = -1

            display_data.append({
                "Model": result.get("model_id", "N/A"),
                "Response": result.get('response_text', 'No response'),
                "Latency (ms)": result.get('latency_ms', 'N/A'),
                "Input Tokens": input_tokens,
                "Output Tokens": output_tokens,
                "Price": price if price > 0 else None ## Negative prices indicate lack of data
            })

        # Display results in a DataFrame
        df = pd.DataFrame(display_data)

        # Display each model's response in its own expander
        for _, row in df.iterrows():
            with st.expander(f"{row['Model']} - {row.get('Latency (ms)', 'N/A')}ms"):
                st.markdown(row["Response"])

        # Also provide a comparison table view
        st.subheader("Comparison Table")
        st.text("Please note that not all pricing information is yet available in this tool. The pricing available through the AWS Pricing API is currently integrated. For models that are not available through this API, the price column will say 'None'. For accurate and up to date pricing of these models, for now, please refer to the 'Model pricing details' section on https://aws.amazon.com/bedrock/pricing/ .")
        st.dataframe(
            df[["Model", "Latency (ms)", "Input Tokens", "Output Tokens", "Price"]],
            use_container_width=True,
            hide_index=True
        )

if __name__ == "__main__":
    main()
