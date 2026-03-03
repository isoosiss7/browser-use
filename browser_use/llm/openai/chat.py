import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar, overload

import httpx
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError
from openai.types.chat import ChatCompletionContentPartTextParam
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.shared.chat_model import ChatModel
from openai.types.shared_params.reasoning_effort import ReasoningEffort
from openai.types.shared_params.response_format_json_schema import JSONSchema, ResponseFormatJSONSchema
from openai.types.responses import Response
from pydantic import BaseModel

from browser_use.llm.base import BaseChatModel
from browser_use.llm.exceptions import ModelProviderError, ModelRateLimitError
from browser_use.llm.messages import BaseMessage
from browser_use.llm.openai.responses_serializer import ResponsesAPIMessageSerializer
from browser_use.llm.openai.serializer import OpenAIMessageSerializer
from browser_use.llm.schema import SchemaOptimizer
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage

T = TypeVar('T', bound=BaseModel)


@dataclass
class ChatOpenAI(BaseChatModel):
	"""
	A wrapper around AsyncOpenAI that implements the BaseLLM protocol.

	This class accepts all AsyncOpenAI parameters while adding model
	and temperature parameters for the LLM interface (if temperature it not `None`).
	"""

	# Model configuration
	model: ChatModel | str

	# Model params
	temperature: float | None = 0.2
	frequency_penalty: float | None = 0.3  # this avoids infinite generation of \t for models like 4.1-mini
	reasoning_effort: ReasoningEffort = 'low'
	seed: int | None = None
	service_tier: Literal['auto', 'default', 'flex', 'priority', 'scale'] | None = None
	top_p: float | None = None
	add_schema_to_system_prompt: bool = False  # Add JSON schema to system prompt instead of using response_format
	dont_force_structured_output: bool = False  # If True, the model will not be forced to output a structured output
	remove_min_items_from_schema: bool = (
		False  # If True, remove minItems from JSON schema (for compatibility with some providers)
	)
	remove_defaults_from_schema: bool = (
		False  # If True, remove default values from JSON schema (for compatibility with some providers)
	)

	# Client initialization parameters
	api_key: str | None = None
	organization: str | None = None
	project: str | None = None
	base_url: str | httpx.URL | None = None
	websocket_base_url: str | httpx.URL | None = None
	timeout: float | httpx.Timeout | None = None
	max_retries: int = 5  # Increase default retries for automation reliability
	default_headers: Mapping[str, str] | None = None
	default_query: Mapping[str, object] | None = None
	http_client: httpx.AsyncClient | None = None
	_strict_response_validation: bool = False
	max_completion_tokens: int | None = 4096
	reasoning_models: list[ChatModel | str] | None = field(
		default_factory=lambda: [
			'o4-mini',
			'o3',
			'o3-mini',
			'o1',
			'o1-pro',
			'o3-pro',
			'gpt-5',
			'gpt-5-mini',
			'gpt-5-nano',
		]
	)
	use_responses_api: bool | str = 'auto'

	# Static
	@property
	def provider(self) -> str:
		return 'openai'

	def _get_client_params(self) -> dict[str, Any]:
		"""Prepare client parameters dictionary."""
		# Resolve credentials and endpoints from environment variables shared with compatible providers
		self.api_key = self.api_key or os.getenv('OPENAI_API_KEY') or os.getenv('COMPATIBLE_OPENAI_API_KEY')
		self.base_url = self.base_url or os.getenv('OPENAI_BASE_URL') or os.getenv('COMPATIBLE_OPENAI_BASE_URL')

		# Allow override of default headers via JSON environment variable for compatibility layers
		if self.default_headers is None:
			env_headers = os.getenv('COMPATIBLE_OPENAI_DEFAULT_HEADERS')
			if env_headers:
				try:
					self.default_headers = json.loads(env_headers)
				except json.JSONDecodeError:
					pass

		# Define base client params
		base_params = {
			'api_key': self.api_key,
			'organization': self.organization,
			'project': self.project,
			'base_url': self.base_url,
			'websocket_base_url': self.websocket_base_url,
			'timeout': self.timeout,
			'max_retries': self.max_retries,
			'default_headers': self.default_headers,
			'default_query': self.default_query,
			'_strict_response_validation': self._strict_response_validation,
		}

		# Create client_params dict with non-None values
		client_params = {k: v for k, v in base_params.items() if v is not None}

		# Add http_client if provided
		if self.http_client is not None:
			client_params['http_client'] = self.http_client

		return client_params

	def get_client(self) -> AsyncOpenAI:
		"""
		Returns an AsyncOpenAI client.

		Returns:
			AsyncOpenAI: An instance of the AsyncOpenAI client.
		"""
		client_params = self._get_client_params()
		return AsyncOpenAI(**client_params)

	@property
	def name(self) -> str:
		return str(self.model)

	def _get_usage(self, response: ChatCompletion) -> ChatInvokeUsage | None:
		if response.usage is not None:
			# Note: completion_tokens already includes reasoning_tokens per OpenAI API docs.
			# Unlike Google Gemini where thinking_tokens are reported separately,
			# OpenAI's reasoning_tokens are a subset of completion_tokens.
			usage = ChatInvokeUsage(
				prompt_tokens=response.usage.prompt_tokens,
				prompt_cached_tokens=response.usage.prompt_tokens_details.cached_tokens
				if response.usage.prompt_tokens_details is not None
				else None,
				prompt_cache_creation_tokens=None,
				prompt_image_tokens=None,
				# Completion
				completion_tokens=response.usage.completion_tokens,
				total_tokens=response.usage.total_tokens,
			)
		else:
			usage = None

		return usage

	@overload
	async def ainvoke(
		self, messages: list[BaseMessage], output_format: None = None, **kwargs: Any
	) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: type[T], **kwargs: Any) -> ChatInvokeCompletion[T]: ...

	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None, **kwargs: Any
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		"""
		Invoke the model with the given messages.

		Args:
			messages: List of chat messages
			output_format: Optional Pydantic model class for structured output

		Returns:
			Either a string response or an instance of output_format
		"""

		if self._should_use_responses_api():
			return await self._ainvoke_responses_api(messages, output_format)

		openai_messages = OpenAIMessageSerializer.serialize_messages(messages)

		try:
			model_params: dict[str, Any] = {}

			if self.temperature is not None:
				model_params['temperature'] = self.temperature

			if self.frequency_penalty is not None:
				model_params['frequency_penalty'] = self.frequency_penalty

			if self.max_completion_tokens is not None:
				model_params['max_completion_tokens'] = self.max_completion_tokens

			if self.top_p is not None:
				model_params['top_p'] = self.top_p

			if self.seed is not None:
				model_params['seed'] = self.seed

			if self.service_tier is not None:
				model_params['service_tier'] = self.service_tier

			if self.reasoning_models and any(str(m).lower() in str(self.model).lower() for m in self.reasoning_models):
				model_params['reasoning_effort'] = self.reasoning_effort
				model_params.pop('temperature', None)
				model_params.pop('frequency_penalty', None)

			if output_format is None:
				# Return string response
				response = await self.get_client().chat.completions.create(
					model=self.model,
					messages=openai_messages,
					**model_params,
				)

				choice = response.choices[0] if response.choices else None
				if choice is None:
					base_url = str(self.base_url) if self.base_url is not None else None
					hint = f' (base_url={base_url})' if base_url is not None else ''
					raise ModelProviderError(
						message=(
							'Invalid OpenAI chat completion response: missing or empty `choices`.'
							' If you are using a proxy via `base_url`, ensure it implements the OpenAI'
							' `/v1/chat/completions` schema and returns `choices` as a non-empty list.'
							f'{hint}'
						),
						status_code=502,
						model=self.name,
					)

				usage = self._get_usage(response)
				return ChatInvokeCompletion(
					completion=choice.message.content or '',
					usage=usage,
					stop_reason=choice.finish_reason,
				)

			else:
				response_format: JSONSchema = {
					'name': 'agent_output',
					'strict': True,
					'schema': SchemaOptimizer.create_optimized_json_schema(
						output_format,
						remove_min_items=self.remove_min_items_from_schema,
						remove_defaults=self.remove_defaults_from_schema,
					),
				}

				# Add JSON schema to system prompt if requested
				if self.add_schema_to_system_prompt and openai_messages and openai_messages[0]['role'] == 'system':
					schema_text = f'\n<json_schema>\n{response_format}\n</json_schema>'
					if isinstance(openai_messages[0]['content'], str):
						openai_messages[0]['content'] += schema_text
					elif isinstance(openai_messages[0]['content'], Iterable):
						openai_messages[0]['content'] = list(openai_messages[0]['content']) + [
							ChatCompletionContentPartTextParam(text=schema_text, type='text')
						]

				if self.dont_force_structured_output:
					response = await self.get_client().chat.completions.create(
						model=self.model,
						messages=openai_messages,
						**model_params,
					)
				else:
					# Return structured response
					response = await self.get_client().chat.completions.create(
						model=self.model,
						messages=openai_messages,
						response_format=ResponseFormatJSONSchema(json_schema=response_format, type='json_schema'),
						**model_params,
					)

				choice = response.choices[0] if response.choices else None
				if choice is None:
					base_url = str(self.base_url) if self.base_url is not None else None
					hint = f' (base_url={base_url})' if base_url is not None else ''
					raise ModelProviderError(
						message=(
							'Invalid OpenAI chat completion response: missing or empty `choices`.'
							' If you are using a proxy via `base_url`, ensure it implements the OpenAI'
							' `/v1/chat/completions` schema and returns `choices` as a non-empty list.'
							f'{hint}'
						),
						status_code=502,
						model=self.name,
					)

				if choice.message.content is None:
					raise ModelProviderError(
						message='Failed to parse structured output from model response',
						status_code=500,
						model=self.name,
					)

				usage = self._get_usage(response)

				parsed = output_format.model_validate_json(choice.message.content)

				return ChatInvokeCompletion(
					completion=parsed,
					usage=usage,
					stop_reason=choice.finish_reason,
				)

		except ModelProviderError:
			# Preserve status_code and message from validation errors
			raise

		except RateLimitError as e:
			raise ModelRateLimitError(message=e.message, model=self.name) from e

		except APIConnectionError as e:
			raise ModelProviderError(message=str(e), model=self.name) from e

		except APIStatusError as e:
			raise ModelProviderError(message=e.message, status_code=e.status_code, model=self.name) from e

		except Exception as e:
			raise ModelProviderError(message=str(e), model=self.name) from e

	def _should_use_responses_api(self) -> bool:
		"""Determine whether to call the Responses API instead of Chat Completions."""
		if isinstance(self.use_responses_api, bool):
			return self.use_responses_api

		wire_api = os.getenv('COMPATIBLE_OPENAI_WIRE_API')
		if wire_api and wire_api.lower().startswith('responses'):
			return True

		if os.getenv('COMPATIBLE_OPENAI_API_KEY'):
			return True

		base = str(self.base_url or os.getenv('COMPATIBLE_OPENAI_BASE_URL') or '')
		if 'litellm' in base or 'responses' in base:
			return True

		return False

	def _coerce_responses_result(self, response: Any) -> Response:
		"""Normalize responses API results or raise helpful model provider errors."""
		if isinstance(response, Response):
			return response

		# Handle string SSE payloads (e.g., 'data: {...}')
		if isinstance(response, str):
			payloads = [
				line.split('data:', 1)[1].strip()
				for line in response.splitlines()
				if line.startswith('data:')
			]
			for payload in reversed(payloads):
				if payload == '[DONE]':
					continue
				try:
					data = json.loads(payload)
					if isinstance(data, dict):
						return Response.model_validate(data)
				except json.JSONDecodeError:
					continue
			raise ModelProviderError(message=response, model=self.name)

		# Some proxies may return dict responses instead of SDK objects
		if isinstance(response, dict):
			message = response.get('error') or response.get('message') or str(response)
			raise ModelProviderError(message=message, model=self.name)

		raise ModelProviderError(message=str(response), model=self.name)

	def _get_usage_from_responses(self, response: Response) -> ChatInvokeUsage | None:
		"""Extract token usage information from a Responses API result."""
		if response.usage is None:
			return None

		input_details = getattr(response.usage, 'input_tokens_details', None)
		cached_tokens = getattr(input_details, 'cached_tokens', None) if input_details else None

		return ChatInvokeUsage(
			prompt_tokens=response.usage.input_tokens,
			prompt_cached_tokens=cached_tokens,
			prompt_cache_creation_tokens=None,
			prompt_image_tokens=None,
			completion_tokens=response.usage.output_tokens,
			total_tokens=response.usage.total_tokens,
		)

	async def _ainvoke_responses_api(
		self,
		messages: list[BaseMessage],
		output_format: type[T] | None = None,
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		"""Invoke the model using the Responses API."""
		input_messages = ResponsesAPIMessageSerializer.serialize_messages(messages)

		params: dict[str, Any] = {
			'model': self.model,
			'input': input_messages,
		}

		if self.temperature is not None:
			params['temperature'] = self.temperature

		if self.max_completion_tokens is not None:
			params['max_output_tokens'] = self.max_completion_tokens

		if self.top_p is not None:
			params['top_p'] = self.top_p

		if self.seed is not None:
			params['seed'] = self.seed

		if self.service_tier is not None:
			params['service_tier'] = self.service_tier

		if self.reasoning_models and any(str(m).lower() in str(self.model).lower() for m in self.reasoning_models):
			params['reasoning'] = {'effort': self.reasoning_effort}
			params.pop('temperature', None)

		try:
			if output_format is None or self.dont_force_structured_output:
				response = await self.get_client().responses.create(**params)
				response = self._coerce_responses_result(response)
				usage = self._get_usage_from_responses(response)
				return ChatInvokeCompletion(
					completion=response.output_text or '',
					usage=usage,
					stop_reason=response.status if response.status else None,
				)

			json_schema = SchemaOptimizer.create_optimized_json_schema(
				output_format,
				remove_min_items=self.remove_min_items_from_schema,
				remove_defaults=self.remove_defaults_from_schema,
			)

			# Inject schema in system prompt if requested
			if self.add_schema_to_system_prompt and input_messages and input_messages[0].get('role') == 'system':
				schema_text = f'\n<json_schema>\n{json_schema}\n</json_schema>'
				content = input_messages[0].get('content', '')
				if isinstance(content, str):
					input_messages[0]['content'] = content + schema_text
				elif isinstance(content, list):
					input_messages[0]['content'] = list(content) + [{'type': 'input_text', 'text': schema_text}]
				params['input'] = input_messages

			text_format = {
				'type': 'json_schema',
				'name': 'agent_output',
				'strict': True,
				'schema': json_schema,
			}
			if self.dont_force_structured_output:
				params.pop('text', None)
			else:
				params['text'] = {'format': text_format}

			response = await self.get_client().responses.create(**params)
			response = self._coerce_responses_result(response)

			if not response.output_text:
				raise ModelProviderError(
					message='Failed to parse structured output from model response',
					status_code=500,
					model=self.name,
				)

			usage = self._get_usage_from_responses(response)
			parsed = output_format.model_validate_json(response.output_text)

			return ChatInvokeCompletion(
				completion=parsed,
				usage=usage,
				stop_reason=response.status if response.status else None,
			)

		except RateLimitError as e:
			raise ModelRateLimitError(message=e.message, model=self.name) from e
		except APIConnectionError as e:
			raise ModelProviderError(message=str(e), model=self.name) from e
		except APIStatusError as e:
			raise ModelProviderError(message=e.message, status_code=e.status_code, model=self.name) from e
		except Exception as e:
			raise ModelProviderError(message=str(e), model=self.name) from e
