# News Research Agent

A research agent that generates reports on companies based on news and web content. This tool is designed to help Cisco sellers by providing up-to-date information about customer companies, their IT priorities, and relevant discovery questions. The agent includes enhanced JSON export functionality that incorporates customer metadata from Excel files and produces professionally formatted Word documents.

## Features

- Automated research on companies using their website URLs
- Integration with Tavily for web search capabilities
- Generation of comprehensive reports in three formats:
  - Markdown: Human-readable format with structured sections
  - Word (.docx): Professionally formatted document with proper styling, tables, and clickable hyperlinks
  - JSON: Structured data format with customer metadata for programmatic processing
- Intelligent section parsing that categorizes content by type (introduction, company news, IT priorities, etc.)
- Customer metadata integration from Excel files
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
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
GOOGLE_APPLICATION_CREDENTIALS=path/to/google-credentials.json  # Optional for cloud storage
USE_GCS_EXCEL=true  # Set to true to use Excel file from Google Cloud Storage
```

The `.env` file is ignored by git to ensure your API keys are not committed to version control.

## Google Cloud Storage Integration

The system can use Google Cloud Storage (GCS) for both storing generated reports and loading the customer Excel file. This provides several benefits:

1. Centralized data storage accessible by all team members
2. Versioned Excel file storage with both timestamped and standard filenames
3. Secure access via signed URLs
4. Automatic fallback to local files if GCS access fails

### Uploading Excel Database to GCS

To upload the customer Excel file to Google Cloud Storage, run:

```bash
python upload_excel_to_gcs.py
```

This will:
1. Upload the Excel file to GCS with two filenames:
   - Standard filename (always the same for easy reference)
   - Versioned filename with timestamp (for historical records)
2. Generate signed URLs for accessing the files
3. Report success with file paths and access URLs

The script will handle file verification, error handling, and provide detailed logs. Once uploaded, set `USE_GCS_EXCEL=true` in your `.env` file to make the research scripts use the Excel file from GCS instead of the local copy.

## Usage

### Company URL Finder Tool

You can use our new interactive tool to quickly search for a specific company and generate a report:

```bash
python company_url_finder.py
```

This script will:
1. Prompt you to enter a company name
2. Search for the company's official website using Tavily
3. Ask you to verify the URL (or provide the correct one)
4. Search for the URL in the database to find the matching SAVM ID
5. Check if an existing report is available (within the last 30 days)
6. Either display links to existing reports or generate a new one

This is the recommended way to research a single company quickly, as it provides:
- Interactive verification to ensure the correct URL is used
- Intelligent URL matching between search results and database entries
- Access to existing reports without unnecessary regeneration
- Direct links to all report formats (Markdown, Word, JSON)

### Running the Full Research Agent

Run the research agent to process all customers from the Excel file:

```bash
python research_agent.py
```

The script will:
1. Load customer data from the Excel file, filtering for valid websites and the Heartland-Gulf region
2. Extract customer metadata from all columns in the Excel file
3. Research each company's website with a focus on company news and IT priorities
4. Generate structured reports in Markdown, Word, and JSON formats
5. Save reports to the local `reports` directory (or Google Cloud Storage if configured)

### JSON Export Structure

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
- `upload_excel_to_gcs.py`: Script to upload the Excel database to Google Cloud Storage
- `requirements.txt`: Python dependencies
- `pyproject.toml`: Project configuration
- `.gitignore`: Git ignore file for excluding unnecessary files
- `.env`: Environment variables configuration (see .env.example for template)
- `reports/`: Local directory for generated reports
- `temp/`: Temporary directory for file processing
- `secrets/`: Directory containing Google Cloud credentials
- `Customer Parquet top 80 select hierarchy for test.xlsx`: Customer database

## Key Functions

- `research_topic()`: Core function for researching a topic using LLMs
- `process_url()`: Processes a single URL to generate a research report
- `process_all_urls()`: Processes all URLs from the customer data
- `save_markdown_report()`: Saves the report in Markdown format and calls other export functions
- `save_json_report()`: Parses markdown content into structured JSON with metadata
- `markdown_to_word()`: Converts markdown to a formatted Word document with proper styling
- `standardize_sources_in_markdown()`: Ensures consistent source formatting in markdown
- `add_hyperlink()`: Adds clickable hyperlinks to Word documents
- `load_customer_data()`: Loads and filters customer data from Excel

## Dependencies

- pandas, numpy: Data processing and Excel file handling
- open-deep-research, langgraph, langchain: LLM orchestration and research capabilities
- tavily-python: Web search API integration
- python-docx, markdown, docx: Document generation and conversion
- aiohttp, asyncio: Asynchronous processing
- regex: Advanced text processing
- tqdm: Progress bars for better user experience
- google-cloud-storage: Cloud storage integration (optional)

## Troubleshooting

If you encounter issues:

1. Check that your API keys are correctly set in the `.env` file
2. Ensure all dependencies are installed with the correct versions
3. Check the log file (`research_agent.log`) for detailed error messages
4. For Word document formatting issues, ensure the `python-docx` and `docx` packages are properly installed

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