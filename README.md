# Cisco Business Aggregator

A modular Python application for aggregating business intelligence, researching companies, and analyzing product innovation opportunities.

## Overview

The Cisco Business Aggregator is a comprehensive tool designed to:

1. **Research Companies**: Gather and analyze recent news and information about companies
2. **Find Company URLs**: Automatically discover company websites based on company names
3. **Analyze Product Innovation**: Identify underpriced products and innovation opportunities in the market

The application is built with a modular architecture, making it easy to extend and maintain.

## Architecture

The application is organized into the following components:

- **Agents**: Core business logic for research, URL finding, and product innovation
- **Services**: Interfaces to external APIs like OpenAI, Tavily, and Google Cloud Storage
- **Models**: Data structures for products, companies, and reports
- **Interfaces**: User-facing interfaces for interacting with agents
- **Utils**: Utility functions for caching, storage, file operations, and reporting
- **Config**: Configuration settings and logging setup

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cisco-business-aggregator.git
   cd cisco-business-aggregator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:
   ```
   # OpenAI API settings
   OPENAI_API_KEY=your_openai_api_key
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key
   AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
   AZURE_OPENAI_API_VERSION=2024-08-01-preview
   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o

   # Tavily API settings
   TAVILY_API_KEY=your_tavily_api_key

   # Storage settings
   USE_LOCAL_STORAGE=true
   LOCAL_REPORTS_DIR=reports
   
   # Cache settings
   CACHE_ENABLED=true
   CACHE_DIRECTORY=cache
   ```

4. Create necessary directories:
   ```bash
   mkdir -p reports cache logs secrets
   ```

## Usage

The application provides a command-line interface for running various tasks:

### Research a Company

```bash
python main.py research --url https://example.com --customer-name "Example Company" --days 30
```

### Find a Company URL

```bash
python main.py find-url --company "Example Company"
```

### Analyze Product Innovation

```bash
python main.py product-innovation --manufacturer "Cisco" --category "networking equipment"
```

### Check Existing Reports

```bash
python main.py check-reports --savm-id "12345"
```

### Clear Cache

```bash
python main.py clear-cache
```

## Directory Structure

```
cisco-business-aggregator/
├── agents/                  # Agent implementations
│   ├── research_agent.py
│   ├── company_url_finder.py
│   └── product_innovation_agent.py
├── config/                  # Configuration settings
│   ├── settings.py
│   └── logging_config.py
├── interfaces/              # User interfaces
│   ├── product_innovation_interface.py
│   └── research_interface.py
├── models/                  # Data models
│   ├── product.py
│   ├── report.py
│   └── company.py
├── services/                # External service integrations
│   ├── tavily_service.py
│   ├── openai_service.py
│   └── gcs_service.py
├── utils/                   # Utility functions
│   ├── cache.py
│   ├── storage.py
│   ├── file_utils.py
│   └── report_utils.py
├── reports/                 # Generated reports
├── cache/                   # Cache storage
├── logs/                    # Log files
├── secrets/                 # API credentials
├── main.py                  # Main entry point
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```

## Extending the Application

The modular architecture makes it easy to extend the application:

1. **Add a new agent**: Create a new file in the `agents/` directory
2. **Add a new service**: Create a new file in the `services/` directory
3. **Add a new model**: Create a new file in the `models/` directory
4. **Add a new interface**: Create a new file in the `interfaces/` directory

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
