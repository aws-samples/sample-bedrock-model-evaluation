# Bedrock Model Testing

A streamlit application for testing and comparing AWS Bedrock AI models.

## Overview

This repository contains a Streamlit application designed to help you interact with and evaluate different AWS Bedrock foundation models. It allows for side-by-side comparison of model outputs and performance assessment.

![Screenshot of the app](img/screenshot.png?raw=true "Application screenshot")


## Features

- Test multiple AWS Bedrock models simultaneously
- Compare response quality, latency and price (where available)
- User-friendly interface with real-time feedback

## Installation

```bash
# Clone the repository
git clone https://github.com/aws-samples/sample-bedrock-model-evaluation.git
cd bedrock-model-testing

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Configure your AWS credentials
2. Run the Streamlit app
   ```bash
   streamlit run app.py
   ```
3. Open your browser at http://localhost:8501

## Requirements

- Python 3.8+
- AWS account with Bedrock access
- Configured AWS credentials

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

[@svetozarm](https://github.com/svetozarm)
