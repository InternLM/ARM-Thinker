import ast
import base64
import json
import logging
import os
import re
import traceback
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI
from arm_agent.utils import decode_base64_image, generate_identifier_path_name
from arm_agent.system_template import TemplateDict

# Import verl tools
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.experimental.agent_loop.tool_parser import HermesToolParser

from dotenv import load_dotenv
load_dotenv()

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

def run_once_with_prompt_single_turn(
    model_client: OpenAI,
    model_name: str,
    messages,
    retry=3,
    temperature=0.0,
    max_tokens=2048,
    n=1,
    top_p=1.0,
    repetition_penalty=1.0,
    top_k=50,
    **kwargs,
):
    """Call model to generate response"""
    num_retries = 0

    while num_retries < retry:
        try:
            chat_response = model_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
                top_p=top_p,
                extra_body={
                    "repetition_penalty": repetition_penalty,
                    "top_k": top_k,
                },
                **kwargs,
            )
            return chat_response
        except Exception as e:
            logger.error(
                f"[Retry {num_retries+1}/{retry}] Exception type: {type(e).__name__}"
            )
            logger.error(f"Error message: {e}")
            logger.error("Traceback:\n" + traceback.format_exc())
            num_retries += 1
    raise RuntimeError(f"Calling OpenAI API failed after {retry} retries.")


NOT_ALLOWED_TOOL_CALL_MESSAGE = "**Important Requirement:**\nYou have reached the maximum number of tool calls and are not allowed to call any more tools. Please reason on the current situation and **must** give your final answer or judgment within `<answer>...</answer>`."


class VerlAgent:
    """Agent based on verl tool system"""

    def __init__(
        self,
        api_base: str,
        api_key: str,
        tool_config_path: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        model_name: str = "gpt-3.5-turbo",
        max_round: int = 30,
        max_tool_response_length: int = 2048,
        tool_response_truncate_side: str = "middle",
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        top_k: int = -1, # -1 means no top_k
        system_template_type: str = None,
        use_role_tool: bool = True,
        **kwargs,
    ):
        self.kwargs = kwargs
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            http_client=httpx.Client(verify=False),
        )
        self.model = model_name
        self.max_round = max_round
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_tool_response_length = max_tool_response_length
        self.tool_response_truncate_side = tool_response_truncate_side
        self.max_parallel_calls = 1  # Default to sequential tool calls
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.top_k = top_k
        self.use_role_tool = use_role_tool
        # Set system template
        # If system_template is a string key in TemplateDict, use the template from dict
        # Otherwise, use it directly as template content
        if system_template_type:
            self.system_template = TemplateDict.get(system_template_type, None)
        else:
            self.system_template = TemplateDict["CommonSystemTemplate"]
        # Initialize verl tools
        self.tools = {}
        self.tool_schemas = []
        if tool_config_path:
            tool_list = initialize_tools_from_config(tool_config_path)
            self.tools = {tool.name: tool for tool in tool_list}
            self.tool_schemas = [
                tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)
                for tool in tool_list
            ]

        # Initialize tool parser (using verl's HermesToolParser)
        # Note: here we need a tokenizer, but we use OpenAI API, so create a simple mock tokenizer
        self.tool_parser = HermesToolParser(self._create_mock_tokenizer())

        # Build system prompt
        self.system_prompt = self._build_system_prompt()

    def _create_mock_tokenizer(self):
        """Create a simple mock tokenizer for tool parser

        Note: in training, HermesToolParser.extract_tool_calls() receives token_ids,
        but we use OpenAI API directly to get text, so this mock tokenizer needs to handle text input.
        """

        class MockTokenizer:
            def decode(self, token_ids):
                if token_ids is None:
                    return ""
                # If input is a string (our case: OpenAI API returns text), return it directly
                if isinstance(token_ids, str):
                    return token_ids
                # If input is a token_ids list (training case), handle simply
                return " ".join(map(str, token_ids))

        return MockTokenizer()

    def _build_system_prompt(self) -> str:
        """Build system prompt, including tool information"""
        if not self.tool_schemas:
            return "You are a helpful assistant."

        tools_text = "\n".join(
            [json.dumps(schema, ensure_ascii=False) for schema in self.tool_schemas]
        )

        return self.system_template.format(tools_text=tools_text)

    def _truncate_tool_response(self, text: str) -> str:
        """Truncate tool response text"""
        if not text or len(text) <= self.max_tool_response_length:
            return text

        if self.tool_response_truncate_side == "left":
            return text[: self.max_tool_response_length] + "...(truncated)"
        elif self.tool_response_truncate_side == "right":
            return "(truncated)..." + text[-self.max_tool_response_length :]
        else:  # middle
            length = self.max_tool_response_length // 2
            return text[:length] + "...(truncated)..." + text[-length:]

    async def _call_verl_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        imgs_map: Dict[str, str] = None,
        texts_map: Dict[str, str] = None,
    ) -> ToolResponse:
        """Call verl tool"""
        # logger.info(f"=== Calling verl tool: {tool_name} ===")
        if tool_name not in self.tools:
            return ToolResponse(text=f"Error: Unknown tool '{tool_name}'")

        tool = self.tools[tool_name]
        instance_id = None

        try:
            # Build response_store
            response_store = {}
            if imgs_map:
                # response_store.update(imgs_map)
                response_store["imgs_map"] = imgs_map
            if texts_map:
                # response_store.update(texts_map)
                response_store["texts_map"] = texts_map

            # Create tool instance, pass response_store
            # logger.info(f"Creating tool instance with response_store: {response_store}")
            instance_id, _ = await tool.create(
                create_kwargs={"response_store": response_store}
            )
            # logger.info(f"Tool instance created with ID: {instance_id}")

            # Execute tool
            # logger.info(f"Executing tool with args: {tool_args}")
            tool_execution_response, _, _ = await tool.execute(instance_id, tool_args)
            # logger.info(f"Tool execution completed")

            # Process text response
            tool_response_text = tool_execution_response.text
            if tool_response_text:
                # logger.info(f"Original tool response length: {len(tool_response_text)}")
                tool_response_text = self._truncate_tool_response(tool_response_text)
                # logger.info(
                #     f"Truncated tool response length: {len(tool_response_text)}"
                # )

            # Create ToolResponse
            tool_response_kwargs = {"text": tool_response_text}

            # Add multimedia data
            for attr_name in ["image", "video"]:
                if hasattr(tool_execution_response, attr_name):
                    attr_value = getattr(tool_execution_response, attr_name)
                    if attr_value is not None:
                        tool_response_kwargs[attr_name] = attr_value

            return ToolResponse(**tool_response_kwargs)

        except Exception as e:
            logger.error(f"Error when executing tool {tool_name}: {e}")
            logger.error(traceback.format_exc())
            return ToolResponse(text=f"Error when executing tool: {e}")
        finally:
            if tool and instance_id:
                await tool.release(instance_id)
                # logger.info(f"Tool instance {instance_id} released")

    def run(
        self,
        user_messages: List[Dict],
        text_0: Optional[str] = None,
    ):
        """Run agent verl, align with training logic"""
        TOOL_CALL_IMG_TEMP_DIR = os.getenv("TOOL_CALL_IMG_TEMP", None)
        if not TOOL_CALL_IMG_TEMP_DIR:
            raise ValueError("Missing `TOOL_CALL_IMG_TEMP` in env.")

        texts_map = {}
        if text_0:
            texts_map["text_0"] = text_0

        imgs_map = {}
        observation_counter = 0  # for generating observation_N naming (e.g., observation_1, observation_2, ...)
        image_type = None

        messages = [{"role": "system", "content": self.system_prompt}] + user_messages

        # logger.info(f"=== Agent Verl Start ===")
        # logger.info(f"System prompt length: {len(self.system_prompt)}")
        # logger.info(f"Available tools: {list(self.tools.keys())}")
        # logger.info(f"User messages count: {len(user_messages)}")
        # logger.info(f"TOOL_CALL_IMG_TEMP_DIR: {TOOL_CALL_IMG_TEMP_DIR}")

        # Step 1: extract and save base64 image, get image type from data:image/{image_type};base64, (currently only support single image as original_image)
        # currently only support single image as original_image
        for message in user_messages:
            if message["role"] != "user":
                continue
            content = message.get("content", [])
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    img_url = item["image_url"]["url"]
                    try:
                        # Check if base64 encoded
                        if not img_url.startswith("data:image/"):
                            logger.warning(
                                f"Image URL is not base64 encoded: {img_url}"
                            )
                            raise ValueError(
                                f"Image URL is not base64 encoded: {img_url}"
                            )

                        # Extract image type
                        match = re.search(r"data:image/(.*);base64,", img_url)
                        if not match:
                            logger.warning(f"Invalid base64 image format: {img_url}")
                            raise ValueError(f"Invalid base64 image format: {img_url}")
                        image_type = match.group(1)

                        # Decode image
                        image = decode_base64_image(img_url)

                        # Save image
                        image_name = f"original_image"
                        path_name = generate_identifier_path_name()
                        image_path = os.path.join(
                            TOOL_CALL_IMG_TEMP_DIR, f"{path_name}.{image_type}"
                        )
                        image.save(image_path)
                        imgs_map[image_name] = image_path
                        # logger.info(
                        #     f"Saved original image: {image_name} -> {image_path}"
                        # )

                    except Exception as e:
                        logger.warning(f"Failed to process image: {e}")
                        raise

        tool_call_count = 0

        for round_num in range(self.max_round):
            # generate model reply
            reply = run_once_with_prompt_single_turn(
                model_client=self.client,
                model_name=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                n=1,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                top_k=self.top_k,
                **self.kwargs,
            )
            content = reply.choices[0].message.content
            
            if content is None:
                # save messages to file
                # with open("./messages.json", "a") as f:
                #     f.write(json.dumps(messages) + "\n")
                return "Failed to obtain Answer from model!", -1

            # [TAG-1011]
            # find first </tool_call> and truncate the content
            if "</tool_call>" in content:
                tool_call_end_index = content.find("</tool_call>")
                if tool_call_end_index != -1:
                    content = content[: tool_call_end_index + len("</tool_call>")]


            # Use verl's HermesToolParser to parse tool calls
            import asyncio

            # Note: in training, token_ids are passed, but we have text, so pass text directly
            _, tool_calls = asyncio.run(self.tool_parser.extract_tool_calls(content))
            # print(f"content: {content}")
            # print(f"tool_calls: {tool_calls}")

            # add assistant message
            messages.append({"role": "assistant", "content": content})

            # if no tool calls, end conversation
            if not tool_calls:
                break

            if round_num >= self.max_round - 1:
                if self.use_role_tool:
                    role = "tool"
                else:
                    role = "user"
                # add the warning message and let model generate final response
                messages.append(
                    {
                        # "role": "user",
                        "role": role,
                        "content": NOT_ALLOWED_TOOL_CALL_MESSAGE,
                    }
                )
                # Generate final response after warning
                final_reply = run_once_with_prompt_single_turn(
                    model_client=self.client,
                    model_name=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    n=1,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    top_k=self.top_k,
                    **self.kwargs,
                )
                final_content = final_reply.choices[0].message.content
                messages.append({"role": "assistant", "content": final_content})
                break

            # process tool calls
            # [TAG-1011]
            # Only process the first tool call
            tool_calls = tool_calls[:1]
            # [TAG-1011]
            for i, tool_call in enumerate(tool_calls):
                tool_name = tool_call.name
                tool_args = json.loads(tool_call.arguments)

                # call verl tool
                tool_result = asyncio.run(
                    self._call_verl_tool(tool_name, tool_args, imgs_map, texts_map)
                )

                # Remove all <image>\n and <image>
                tool_result.text = tool_result.text.replace("<image>\n", "").replace("<image>", "")  

                # process tool response - align with training logic

                # process tool returned image, save to imgs_map
                if tool_result.image:
                    for img in tool_result.image:
                        # generate new image key name
                        observation_counter += 1
                        new_image_name = f"observation_{observation_counter}"
                        path_name = generate_identifier_path_name()
                        # use the same image type as original image, fallback to png
                        img_type = image_type if image_type else "png"
                        image_path = os.path.join(
                            TOOL_CALL_IMG_TEMP_DIR, f"{path_name}.{img_type}"
                        )
                        img.save(image_path)
                        imgs_map[new_image_name] = image_path

                        # logger.info(
                        #     f"Saved tool result image: {new_image_name} -> {image_path}"
                        # )

                if tool_result.image or tool_result.video:
                    # multimodal content
                    content_list = []
                    # best to add image first because in prompt we said "this is..."
                    if tool_result.image:
                        # convert PIL images to base64 for OpenAI API
                        for img in tool_result.image:
                            import io
                            import base64

                            buffer = io.BytesIO()
                            # convert file extension to PIL format name
                            format_map = {
                                "jpg": "JPEG",
                                "jpeg": "JPEG",
                                "png": "PNG",
                                "gif": "GIF",
                                "bmp": "BMP",
                                "webp": "WEBP",
                            }
                            pil_format = format_map.get(img_type.lower(), "PNG")
                            img.save(buffer, format=pil_format)
                            img_base64 = base64.b64encode(buffer.getvalue()).decode()
                            content_list.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/{img_type};base64,{img_base64}"
                                    },
                                }
                            )
                    if tool_result.text:
                        content_list.append({"type": "text", "text": tool_result.text})
                    model_name = self.model.lower()
                    if self.use_role_tool:
                        role = "tool"
                    else:
                        role = "user"
                    tool_message = {
                        "role": role,
                        "content": content_list,
                    }
                    messages.append(tool_message)
                    # logger.info(f"Added tool message (multimodal): {tool_message}")
                else:
                    model_name = self.model.lower()
                    if self.use_role_tool:
                        role = "tool"
                    else:
                        role = "user"
                    tool_message = {
                        "role": role,
                        "content": tool_result.text or "",
                    }
                    messages.append(tool_message)
                    # logger.info(f"Added tool message (text): {tool_message}")

                tool_call_count += 1

        # logger.info(f"=== Agent Verl End ===")
        # logger.info(f"Total tool calls: {tool_call_count}")
        # logger.info(f"Final messages count: {len(messages)}")
        return messages, tool_call_count
