#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities Package

This package provides utility functions and classes for the application.
It includes caching, storage, file operations, and report generation utilities.

Usage:
    from utils.cache import cache_with_ttl
    from utils.storage import save_file
    from utils.file_utils import markdown_to_json
    from utils.report_utils import save_markdown_report
"""

# Import key functions and classes for easy access
from utils.cache import (
    cache_tavily_search,
    cache_llm_call,
    cache_with_ttl,
    async_cache_with_ttl,
    clear_cache,
    get_cache_stats,
    invalidate_cache_entry
)

from utils.storage import (
    save_file,
    read_file,
    list_files_in_gcs,
    upload_to_gcs,
    download_from_gcs,
    get_temp_dir,
    ensure_directory_exists,
    extract_company_name
)

from utils.file_utils import (
    markdown_to_json,
    markdown_to_word,
    standardize_sources_in_markdown,
    create_markdown_report,
    create_error_report,
    create_placeholder_report
)

from utils.report_utils import (
    save_markdown_report,
    save_error_report,
    save_placeholder_report,
    check_existing_reports,
    get_report_urls,
    load_report_json
)
