# Bedrock Model Testing

A streamlit application for testing and comparing AWS Bedrock AI models.

## Overview

This repository contains a Streamlit application designed to help you interact with and evaluate different AWS Bedrock foundation models. It allows for side-by-side comparison of model outputs, parameter tuning, and performance assessment.

## Features

- Test multiple AWS Bedrock models simultaneously
- Customize inference parameters (temperature, top_p, etc.)
- Compare response quality and latency
- Save and load prompts for consistent testing
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
