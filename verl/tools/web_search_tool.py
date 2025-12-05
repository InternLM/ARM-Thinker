# -*- coding: utf-8 -*-
import os
import logging
import requests
from uuid import uuid4
from typing import Dict, Any, Optional
from verl.utils.rollout_trace import rollout_trace_op
from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse


# ============== Logging Configuration ==============
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


# ============== Tool Definition ==============
class WebSearchTool(BaseTool):
    """
    WebSearchTool: Search internet content (via Serper API).
    Params:
      - query: str, natural language search query.
    Returns:
      - A text string containing formatted search results.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return OpenAI tool schema"""
        return self.tool_schema

    async def create(
        self,
        instance_id: Optional[str] = None,
        response_store: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> tuple[str, ToolResponse]:
        """Create a tool instance"""
        if instance_id is None:
            instance_id = str(uuid4())
        new_response_store = kwargs.get("create_kwargs", {}).get(
            "response_store", response_store or {}
        )
        self._instance_dict[instance_id] = {"response_store": new_response_store}
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: Dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """Execute a web search"""
        query = parameters.get("query")
        if not query:
            return (
                ToolResponse(text="Error: missing required parameter 'query'."),
                -0.05,
                {"success": False},
            )

        try:
            results = await self._search(query)
            formatted = self._format_results(results)
            return ToolResponse(text=formatted), 0.0, {"success": True}
        except Exception as e:
            logger.error(f"WebSearchTool error: {e}")
            return ToolResponse(text=f"Error: {e}"), -0.05, {"success": False}

    async def _search(self, query: str) -> dict:
        """Search using the Serper API"""
        api_key = os.getenv("SERPER_API_KEY")
        api_url = os.getenv("SERPER_URL", "https://google.serper.dev/search")
        print(f"api_key: {api_key}")
        print(f"api_url: {api_url}")

        if not api_key:
            raise ValueError(
                "SERPER_API_KEY is missing. Please set it as an environment variable."
            )

        headers = {"Content-Type": "application/json", "X-API-KEY": api_key}
        payload = {"q": query, "num": 5}
        response = requests.post(api_url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()

    def _format_results(self, data: dict) -> str:
        """Format Serper search results"""
        lines = []

        # answerBox
        answer = data.get("answerBox", {})
        if "snippet" in answer:
            lines.append(
                f"[AnswerBox] {answer.get('title', 'AnswerBox')}\n"
                f"{answer['snippet']}\n{answer.get('link', '')}"
            )

        # knowledgeGraph
        kg = data.get("knowledgeGraph", {})
        if kg:
            lines.append(
                f"[KnowledgeGraph] {kg.get('title', '')}\n"
                f"{kg.get('description', '')}\n{kg.get('descriptionLink', '')}"
            )
            for key, value in kg.get("attributes", {}).items():
                lines.append(f"- {key}: {value}")

        # organic results
        for i, result in enumerate(data.get("organic", []), 1):
            lines.append(
                f"[{i}] {result.get('title', '')}\n"
                f"{result.get('snippet', '')}\n{result.get('link', '')}\n"
                f"Date: {result.get('date', 'Unknown')}"
            )

        if not lines:
            lines.append("No search results found.")

        return "```\n" + "\n\n".join(lines) + "\n```"

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release instance"""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
