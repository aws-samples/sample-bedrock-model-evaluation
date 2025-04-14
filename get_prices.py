"""
Amazon Bedrock Pricing API client.

This module provides functionality to retrieve and process Amazon Bedrock pricing information.
"""
import boto3
import json
import os
import logging
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_bedrock_pricing_data(pricing_client, region: str) -> Dict[str, Any]:
    """
    Fetch raw pricing data from the AWS Pricing API for Amazon Bedrock.

    Args:
        pricing_client: Boto3 pricing client
        region: AWS region code

    Returns:
        Raw API response containing pricing data
    """
    logger.info(f"Fetching Amazon Bedrock pricing data for region {region}")
    try:
        response = pricing_client.get_products(
            ServiceCode='AmazonBedrock',
            Filters=[
                {'Type': 'TERM_MATCH', 'Field': 'regionCode', 'Value': region},
                {'Type': 'TERM_MATCH', 'Field': 'feature', 'Value': 'On-demand Inference'},
                {'Type': 'TERM_MATCH', 'Field': 'serviceCode', 'Value': 'AmazonBedrock'},
            ]
        )
        logger.info(f"Successfully retrieved {len(response.get('PriceList', []))} price items")
        return response
    except Exception as e:
        logger.error(f"Error fetching Amazon Bedrock pricing data: {str(e)}")
        raise

def save_debug_output(data: List[Dict], filepath: str = "test_prices.json") -> None:
    """
    Save pricing data to a JSON file for debugging purposes.

    Args:
        data: Pricing data to save
        filepath: Path to the output file
    """
    try:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=1)
        logger.debug(f"Debug pricing data saved to {filepath}")
    except Exception as e:
        logger.warning(f"Failed to save debug pricing data: {str(e)}")

def determine_token_type(inference_type: str) -> str:
    """
    Determine token type (input/output) based on inference type.

    Args:
        inference_type: The inference type string from the pricing data

    Returns:
        Token type classification ('input', 'output', or 'unknown')
    """
    inference_type_lower = inference_type.lower()
    if 'input' in inference_type_lower:
        return 'input'
    elif 'output' in inference_type_lower:
        return 'output'
    return 'unknown'

def extract_token_prices(parsed_price_list: List[Dict]) -> List[Dict]:
    """
    Extract token pricing information from parsed price list.

    Args:
        parsed_price_list: List of parsed price data from the API

    Returns:
        List of standardized token pricing information
    """
    logger.info("Extracting token pricing information")
    prices = []

    for price_item_json in parsed_price_list:
        # Process each pricing item
        for term_type in price_item_json.get('terms', {}).values():
            for term in term_type.values():
                for dimension in term.get('priceDimensions', {}).values():
                    # Only process items with "1K tokens" as the unit
                    if dimension.get('unit') == "1K tokens":
                        product_attrs = price_item_json.get('product', {}).get('attributes', {})
                        model_name = product_attrs.get('model', product_attrs.get('titanModel', 'Unknown Model'))
                        inference_type = product_attrs.get('inferenceType', '')

                        # Skip cache-related items
                        if "cache" in inference_type.lower():
                            continue

                        # Get token type and price
                        token_type = determine_token_type(inference_type)
                        price_per_unit = dimension.get('pricePerUnit', {}).get('USD', 'N/A')

                        # Create standardized pricing info
                        pricing_info = {
                            'model': model_name,
                            'token_type': token_type,
                            'price_per_1k_tokens': price_per_unit,
                            'region': product_attrs.get('location', 'Unknown Region')
                        }

                        prices.append(pricing_info)

    logger.info(f"Extracted {len(prices)} token pricing entries")
    return prices

def group_prices_by_model(prices: List[Dict]) -> Dict[str, Dict[str, str]]:
    """
    Group pricing data by model and token type.

    Args:
        prices: List of extracted price information

    Returns:
        Dictionary mapping model names to token types and their prices
    """
    model_price_map = {}

    for price in prices:
        model = price["model"]
        if model not in model_price_map:
            model_price_map[model] = {}

        token_type = price["token_type"]
        price1k = price["price_per_1k_tokens"]
        model_price_map[model][token_type] = price1k

    logger.debug(f"Grouped prices for {len(model_price_map)} models")
    return model_price_map

def organize_prices_by_region(model_prices: Dict[str, Dict[str, str]], region: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Organize model prices by region.

    Args:
        model_prices: Dictionary of model prices grouped by model and token type
        region: AWS region code

    Returns:
        Dictionary with region as the top-level key, containing model prices
    """
    return {region: model_prices}

def format_price_data(prices: List[Dict], region: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Format extracted price data into a structured dictionary by model and token type.

    Args:
        prices: List of extracted price information
        region: AWS region code

    Returns:
        Dictionary of prices organized by region, model, and token type
    """
    logger.info(f"Formatting price data for region {region}")

    # First group by model
    model_prices = group_prices_by_model(prices)

    # Then organize by region
    price_per_model = organize_prices_by_region(model_prices, region)

    logger.info(f"Successfully formatted price data into {len(price_per_model[region])} model entries")
    return price_per_model

def get_bedrock_prices(region='us-east-1'):
    BEDROCK_REGION = region

    # Initialize Pricing client
    pricing_client = boto3.client('pricing', region_name=BEDROCK_REGION)

    # Query for Amazon Bedrock pricing using the fetch function
    response = fetch_bedrock_pricing_data(pricing_client, BEDROCK_REGION)

    # Parse price list and save debug output
    parsed_price_list = [json.loads(p) for p in response["PriceList"]]
    save_debug_output(parsed_price_list, "test_prices.json")

    # Extract token prices using the dedicated function
    prices = extract_token_prices(parsed_price_list)

    # Format the data into the final structure
    price_per_model = format_price_data(prices, BEDROCK_REGION)

    return price_per_model

# Example usage
if __name__ == "__main__":
    prices = get_bedrock_prices()
    #print(prices)
