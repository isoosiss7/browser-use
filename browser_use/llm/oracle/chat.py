"""
Oracle Code Assist (OCA) compatibility wrapper for ChatOpenAI.

This provides sensible defaults for Browser-Use when calling Oracle's
OpenAI-compatible gateway that backs Codex integrations.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Mapping

import httpx

from browser_use.llm.openai.chat import ChatOpenAI

# Default Oracle gateway endpoint used by Codex CLI.
DEFAULT_OCA_BASE_URL = 'https://code-internal.aiservice.us-chicago-1.oci.oraclecloud.com/20250206/app/litellm'

# Recommended headers for Browser-Use requests against the Oracle gateway.
DEFAULT_OCA_HEADERS: Mapping[str, str] = {'client': 'codex-cli', 'client-version': '0'}


def _load_headers_from_env() -> Mapping[str, str] | None:
	"""Load default headers from COMPATIBLE_OPENAI_DEFAULT_HEADERS if set."""
	env_headers = os.getenv('COMPATIBLE_OPENAI_DEFAULT_HEADERS')
	if not env_headers:
		return None

	try:
		return json.loads(env_headers)
	except json.JSONDecodeError:
		return None


@dataclass
class ChatOracleCodeAssist(ChatOpenAI):
	"""
	Preconfigured ChatOpenAI variant for Oracle Code Assist.

	Args:
	    model: Oracle model identifier (defaults to ``oca/gpt-5-codex``).
	    api_key: Falls back to ``COMPATIBLE_OPENAI_API_KEY``.
	    base_url: Falls back to ``COMPATIBLE_OPENAI_BASE_URL`` or the standard Oracle gateway.
	    default_headers: Falls back to JSON from ``COMPATIBLE_OPENAI_DEFAULT_HEADERS`` or sensible defaults.
	"""

	model: str = 'oca/gpt-5-codex'
	api_key: str | None = None
	base_url: str | httpx.URL | None = None
	default_headers: Mapping[str, str] | None = field(default=None)
	use_responses_api: bool | str = True

	def __post_init__(self) -> None:
		if self.api_key is None:
			self.api_key = os.getenv('COMPATIBLE_OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY')

		if self.base_url is None:
			self.base_url = os.getenv('COMPATIBLE_OPENAI_BASE_URL', DEFAULT_OCA_BASE_URL)

		if self.default_headers is None:
			self.default_headers = _load_headers_from_env() or DEFAULT_OCA_HEADERS
