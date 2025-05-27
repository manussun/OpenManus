import unittest
from unittest.mock import patch, AsyncMock, Mock # Use AsyncMock for async methods

from app.llm import LLM, NON_TOOL_MODELS, REASONING_MODELS, MULTIMODAL_MODELS
from app.config import LLMSettings, default_llm_config # Assuming default_llm_config can be imported or constructed
from app.schema import Message, ToolChoice

# It's good practice to have a way to reset the singleton for testing,
# or ensure tests don't rely on a mutable global state.
# For this test, we'll try to manage by re-assigning attributes on a shared instance,
# or by clearing and re-initializing if the LLM class structure allows easily.

class TestLLMAskTool(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # Clear the LLM singleton instance cache before each test
        # to ensure fresh instances and configurations.
        LLM._instances = {}

        # It's important that the LLM can be initialized in a testable way.
        # We'll use a dummy configuration.
        self.dummy_llm_settings = {
            "default": LLMSettings(model="test-dummy", api_key="test_key", base_url="http://localhost:8080", max_tokens=100, temperature=0.7, api_type="openai"),
            "test_non_tool_model_config": LLMSettings(model=NON_TOOL_MODELS[0], api_key="test_key", base_url="http://localhost:8080", max_tokens=100, api_type="openai"),
            "test_tool_model_config": LLMSettings(model="gpt-4o", api_key="test_key", base_url="http://localhost:8080", max_tokens=100, api_type="openai")
        }
        # Patch the global config object if LLM strictly uses it.
        # For this test, we assume LLM can be instantiated with a specific config dict for testing.
        # This might require either patching `app.config.config.llm` or ensuring `LLM(llm_config=...)` works.
        # Let's assume we can pass a full config structure to LLM for testing.
        # If not, this setup would need to patch `app.config.config.llm`.
        
        # A simple way to ensure LLM uses our test settings for these tests:
        # Patch the global config object that LLM loads.
        self.config_patcher = patch('app.config.config.llm', new=self.dummy_llm_settings)
        self.mock_config = self.config_patcher.start()


    def tearDown(self):
        # Stop any patches started in setUp
        self.config_patcher.stop()
        # Clear the LLM singleton instance cache after each test
        LLM._instances = {}


    @patch('app.llm.AsyncOpenAI') # Patch the client class
    async def test_ask_tool_with_non_tool_model(self, MockAsyncOpenAI):
        # Configure the mock client instance and its methods
        mock_client_instance = MockAsyncOpenAI.return_value
        mock_create_method = AsyncMock() 
        mock_create_method.return_value = Mock( # Simulate the response structure
            choices=[Mock(message=Mock(content="mock_response", tool_calls=None))], # Non-tool models shouldn't return tool_calls
            usage=Mock(prompt_tokens=10, completion_tokens=5)
        )
        mock_client_instance.chat.completions.create = mock_create_method

        # Instantiate LLM with a config for a non-tool model
        llm_instance = LLM(config_name="test_non_tool_model_config", llm_config=self.dummy_llm_settings)
        # Crucial: Ensure the client used by this LLM instance is our mock
        llm_instance.client = mock_client_instance
        # Ensure the model is correctly set (should be handled by config_name if LLM init is robust)
        self.assertEqual(llm_instance.model, NON_TOOL_MODELS[0])


        dummy_messages = [{"role": "user", "content": "Hello"}]
        dummy_tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]

        with patch('app.llm.logger.warning') as mock_logger_warning:
            response_message = await llm_instance.ask_tool(messages=dummy_messages, tools=dummy_tools)

        self.assertIsNotNone(response_message)
        self.assertEqual(response_message.content, "mock_response")
        
        mock_create_method.assert_called_once()
        call_kwargs = mock_create_method.call_args.kwargs
        
        self.assertNotIn("tools", call_kwargs, "Tools parameter should not be passed to non-tool model")
        self.assertNotIn("tool_choice", call_kwargs, "Tool_choice parameter should not be passed to non-tool model")
        mock_logger_warning.assert_called_once()
        self.assertIn(f"Model {llm_instance.model} does not support tool use", mock_logger_warning.call_args[0][0])


    @patch('app.llm.AsyncOpenAI')
    async def test_ask_tool_with_tool_model(self, MockAsyncOpenAI):
        mock_client_instance = MockAsyncOpenAI.return_value
        mock_create_method = AsyncMock()
        # Simulate a response that might include a tool call
        mock_tool_call_response = Mock()
        mock_tool_call_response.id="call_123"
        mock_tool_call_response.type="function"
        mock_tool_call_response.function=Mock(name="get_weather", arguments='{"location":"Paris"}')

        mock_create_method.return_value = Mock(
            choices=[Mock(message=Mock(content=None, tool_calls=[mock_tool_call_response]))], 
            usage=Mock(prompt_tokens=10, completion_tokens=5)
        )
        mock_client_instance.chat.completions.create = mock_create_method

        llm_instance = LLM(config_name="test_tool_model_config", llm_config=self.dummy_llm_settings)
        llm_instance.client = mock_client_instance # Assign the mock client
        self.assertEqual(llm_instance.model, "gpt-4o") # Check model from config

        dummy_messages = [{"role": "user", "content": "What's the weather in Paris?"}]
        dummy_tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {}}}]

        with patch('app.llm.logger.warning') as mock_logger_warning: # To ensure no warning is logged
            response_message = await llm_instance.ask_tool(messages=dummy_messages, tools=dummy_tools, tool_choice=ToolChoice.AUTO)

        self.assertIsNotNone(response_message)
        self.assertIsNotNone(response_message.tool_calls)
        self.assertEqual(len(response_message.tool_calls), 1)
        self.assertEqual(response_message.tool_calls[0].function.name, "get_weather")

        mock_create_method.assert_called_once()
        call_kwargs = mock_create_method.call_args.kwargs
        
        self.assertIn("tools", call_kwargs)
        self.assertEqual(call_kwargs["tools"], dummy_tools)
        self.assertIn("tool_choice", call_kwargs)
        self.assertEqual(call_kwargs["tool_choice"], ToolChoice.AUTO)
        mock_logger_warning.assert_not_called() # No warning should be logged for tool-supporting models

    @patch('app.llm.AsyncOpenAI')
    async def test_ask_tool_with_tool_model_no_tools_provided(self, MockAsyncOpenAI):
        """Test tool model when no tools are passed to ask_tool."""
        mock_client_instance = MockAsyncOpenAI.return_value
        mock_create_method = AsyncMock()
        mock_create_method.return_value = Mock(
            choices=[Mock(message=Mock(content="mock_response_no_tools_needed", tool_calls=None))],
            usage=Mock(prompt_tokens=10, completion_tokens=5)
        )
        mock_client_instance.chat.completions.create = mock_create_method

        llm_instance = LLM(config_name="test_tool_model_config", llm_config=self.dummy_llm_settings)
        llm_instance.client = mock_client_instance
        self.assertEqual(llm_instance.model, "gpt-4o")

        dummy_messages = [{"role": "user", "content": "Just chat with me."}]

        response_message = await llm_instance.ask_tool(messages=dummy_messages, tools=None) # Explicitly pass tools=None

        mock_create_method.assert_called_once()
        call_kwargs = mock_create_method.call_args.kwargs

        # Even for tool-supporting models, if `tools` is None or empty,
        # 'tools' and 'tool_choice' should not be sent.
        # The original code has `if self.model not in NON_TOOL_MODELS and tools:`,
        # so if `tools` is None or empty, it won't pass them.
        self.assertNotIn("tools", call_kwargs)
        self.assertNotIn("tool_choice", call_kwargs)
        self.assertEqual(response_message.content, "mock_response_no_tools_needed")


if __name__ == '__main__':
    unittest.main()
