# News Research Agent

A research agent that generates reports on companies based on news and web content. This tool is designed to help Cisco sellers by providing up-to-date information about customer companies, their IT priorities, and relevant discovery questions. The agent includes enhanced JSON export functionality that incorporates customer metadata from Excel files and produces professionally formatted Word documents.

## Features

- Automated research on companies using their website URLs
- Integration with Tavily for web search capabilities
- Azure OpenAI integration with GPT-4o for advanced text generation
- Human-in-the-loop verification for URLs and company identification
- Generation of comprehensive reports in three formats:
  - Markdown: Human-readable format with structured sections
  - Word (.docx): Professionally formatted document with proper styling, tables, and clickable hyperlinks
  - JSON: Structured data format with customer metadata for programmatic processing
- Intelligent section parsing that categorizes content by type (introduction, company news, IT priorities, etc.)
- Customer metadata integration from Excel files stored in Google Cloud Storage
- Optimized JSON structure using SAVM_ID as the primary key
- Robust error handling with informative error reports
- Retry mechanism for API failures with exponential backoff
- Advanced formatting features:
  - Properly formatted tables in Word documents
  - Clickable hyperlinks for URLs in sources
  - Consistent numbered lists for discovery questions
  - Standardized source formatting

## Installation

1. Clone the repository
2. Install dependencies using pip:

```bash
pip install -e .
```

Or install directly from the requirements.txt file:

```bash
pip install -r requirements.txt
```

## Environment Setup

The project includes a `.env.example` file that you can use as a template. Copy this file to create your own `.env` file:

```bash
cp .env.example .env
```

Then edit the `.env` file to add your API keys:

```
# OpenAI API credentials
OPENAI_API_KEY=your_openai_api_key

# Azure OpenAI configuration - used for all LLM operations
AZURE_OPENAI_ENDPOINT=https://your-azure-openai-service.openai.azure.com/
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o

# Tavily API key for web search
TAVILY_API_KEY=your_tavily_api_key

# Google Cloud Storage configuration
GOOGLE_APPLICATION_CREDENTIALS=path/to/google-credentials.json
OUTCOMES_PATH=gs://your-bucket-name
```

The `.env` file is ignored by git to ensure your API keys are not committed to version control.

## Google Cloud Storage Integration

The system uses Google Cloud Storage (GCS) for both storing generated reports and loading the customer Excel file. This provides several benefits:

1. Centralized data storage accessible by all team members
2. Versioned Excel file storage with both timestamped and standard filenames
3. Secure access via signed URLs
4. Automatic fallback to local files if GCS access fails

### Required Google Cloud Setup

1. Create a Google Cloud Project
2. Create a Storage bucket
3. Generate service account credentials and save as `secrets/google-credentials-dev.json`
4. Upload the Excel database to GCS

## Usage

### Company URL Finder Tool

You can use our interactive tool to quickly search for a specific company and generate a report:

```bash
python company_url_finder.py
```

This script will:
1. Prompt you to enter a company name
2. Search for the company's official website using Tavily
3. Ask you to verify the URL (or provide the correct one)
4. Search for the URL in the database to find the matching SAVM ID
5. Ask you to verify if the correct company was matched
6. If incorrect, allow searching for the right company by name
7. Check if an existing report is available (within the last 30 days)
8. Either display links to existing reports or generate a new one

This is the recommended way to research a single company quickly, as it provides:
- Interactive verification to ensure the correct URL is used
- Interactive verification to ensure the correct company is identified
- Intelligent URL matching between search results and database entries
- Access to existing reports without unnecessary regeneration
- Direct links to all report formats (Markdown, Word, JSON)

### Human-in-the-Loop Verification

The application includes two key verification steps where human input is requested:

1. **URL Verification**: After finding a potential company URL, the user is asked to verify if it's correct or provide the correct URL
2. **Company/SAVM_ID Verification**: After matching the URL to a company in the database, the user is asked to verify if the correct company was identified, or search for the correct one by name

These verification steps ensure accuracy and allow manual correction when automatic matching fails.

### Running the Full Research Agent

Run the research agent to process all customers from the Excel file:

```bash
python research_agent.py
```

The script will:
1. Load customer data from the Excel file in Google Cloud Storage
2. Filter for valid websites
3. Research each company's website with a focus on company news and IT priorities
4. Generate structured reports in Markdown, Word, and JSON formats
5. Save reports to Google Cloud Storage

### Azure OpenAI Integration

The application uses Azure OpenAI for all language model operations. The configuration is handled at startup with proper environment variable mapping.

To verify Azure OpenAI connectivity, you can run the test script:

```bash
python test_langchain_azure.py
```

This script will test both direct OpenAI API access and the LangChain integration.

## JSON Export Structure

The JSON export includes:
- SAVM_ID as the primary key at the top level
- Common metadata (URL, topic, company name, generation date)
- All customer metadata from the Excel file
- Structured sections with:
  - Section type identification (introduction, company_news, it_priorities, etc.)
  - Section level (h1, h2, h3)
  - Section content
  - Sources attached as metadata to their parent sections

This structure facilitates easy programmatic access and analysis of the report data.

## Word Document Formatting

The Word documents generated by the agent include:
- Properly formatted headings with consistent styling
- Tables with borders and proper cell alignment
- Numbered lists with consistent indentation
- Clickable hyperlinks for all URLs in the sources section
- Standardized source formatting to avoid duplication
- Proper handling of markdown formatting (bold, italic, etc.)

## Project Structure

- `research_agent.py`: Main script for the research agent
- `company_url_finder.py`: Interactive tool for finding company URLs and generating reports
- `test_langchain_azure.py`: Test script for Azure OpenAI connectivity
- `requirements.txt`: Python dependencies
- `.env`: Environment variables configuration
- `secrets/`: Directory containing Google Cloud credentials
- `Customer Parquet top 80 select hierarchy for test.xlsx`: Customer database (locally and in GCS)

## Key Functions

### company_url_finder.py
- `find_company_url()`: Uses Tavily to find a company's official website
- `verify_url_human_in_loop()`: Gets human verification of the URL
- `search_database_for_url()`: Searches Excel database for URL match
- `verify_savm_id_match()`: Gets human verification of the company match
- `check_existing_reports()`: Checks for recent reports in GCS
- `download_report_from_gcs()`: Downloads existing reports from GCS

### research_agent.py
- `research_topic()`: Core function for researching a topic using LLMs
- `process_url()`: Processes a single URL to generate a research report
- `save_markdown_report()`: Saves the report in Markdown format and calls other export functions
- `save_json_report()`: Parses markdown content into structured JSON with metadata
- `markdown_to_word()`: Converts markdown to a formatted Word document with proper styling

## Dependencies

- pandas, numpy: Data processing and Excel file handling
- open-deep-research, langgraph, langchain: LLM orchestration and research capabilities
- tavily-python: Web search API integration
- python-docx, markdown, docx: Document generation and conversion
- aiohttp, asyncio: Asynchronous processing
- regex: Advanced text processing
- google-cloud-storage: Cloud storage integration
- openai: OpenAI API client
- python-dotenv: Environment variable management

## Troubleshooting

If you encounter issues:

1. Check that your API keys are correctly set in the `.env` file
2. Ensure all dependencies are installed with the correct versions
3. Check the log file (`company_url_finder.log` or console output) for detailed error messages
4. For Azure OpenAI connectivity issues, run `test_langchain_azure.py` to diagnose the problem
5. For Google Cloud Storage issues, verify your credentials and bucket access permissions

## License

Proprietary - Cisco Systems 

# Azure OpenAI Test Script

This repository contains a simple test script to verify connectivity to Azure OpenAI's GPT-4o model.

## Prerequisites

- Python 3.8 or higher
- Azure OpenAI account with GPT-4o access
- `.env` file with Azure OpenAI credentials

## Installation

1. Clone this repository or download the files
2. Install the required Python packages:

```bash
pip install python-dotenv openai
```

## Configuration

The script expects a `.env` file in the root directory with the following variables:

```
AZURE_OPENAI_ENDPOINT=https://phx-sales-ai.openai.azure.com/
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
```

These variables should already be set up in your environment.

## Usage

Run the test script to verify that your Azure OpenAI connection is working:

```bash
python test_azure_openai.py
```

If successful, the script will:
1. Initialize the Azure OpenAI client
2. Send a test prompt to GPT-4o
3. Display the model's response along with token usage information

## Troubleshooting

If the test fails, the script will output specific error information. Common issues include:

- Invalid API credentials
- Network connectivity problems
- Rate limiting or quota issues
- Model unavailability

Check the error message and verify your Azure OpenAI configuration. 