import asyncio
import copy
import json
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

NOT_ALLOWED_TOOL_CALL_MESSAGE = "\n\n**Important Requirement:**\nYou have reached the maximum number of tool calls and are not allowed to call any more tools. Please reason on the current situation and **must** give your final answer or judgment within `<answer>...</answer>`."

@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = (
            config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        )
        cls.max_parallel_calls = (
            config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        )
        cls.max_tool_response_length = (
            config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        )
        cls.tool_response_truncate_side = (
            config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        )
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = (
            initialize_tools_from_config(tool_config_path) if tool_config_path else []
        )
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [
            tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)
            for tool in tool_list
        ]
        cls.tool_parser = ToolParser.get_tool_parser(
            config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer
        )

        cls.apply_chat_template_kwargs = config.data.get(
            "apply_chat_template_kwargs", {}
        )
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}],
            add_generation_prompt=False,
            tokenize=True,
            **cls.apply_chat_template_kwargs,
        )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # [TAG-Chris]
        # --- [ADD] Stop at </tool_call> and </answer>, and KEEP them in the output ---
        # if (
        #     "stop_token_ids" not in sampling_params
        #     or not sampling_params["stop_token_ids"]
        # ):
        #     # Convert these stop strings into token IDs
        #     stop_token_ids = []
        #     for s in ["</tool_call>"]:  # </answer> has no corresponding id in the tokenizer
        #         ids = self.tokenizer.encode(s, add_special_tokens=False)
        #         stop_token_ids.extend(ids)
        #     sampling_params["stop_token_ids"] = stop_token_ids
        # ---------------------------------------------------------------------------

        print(f"sampling_params: {sampling_params}")

        # [NEW]
        # logger.info(
        #     f"verl/experimental/agent_loop/tool_agent_loop.py:async def run():messages:\n{messages}"
        # )
        # logger.info(f"verl/experimental/agent_loop/tool_agent_loop.py:async def run():kwargs:\n{kwargs}")

        image_data = copy.deepcopy(
            kwargs.get("multi_modal_data", {}).get("image", None)
        )

        # Debug: Print image_data content (simplified)
        # print(
        #     f"DEBUG: image_data type: {type(image_data)}, length: {len(image_data) if isinstance(image_data, list) else 'N/A'}"
        # )

        # Initialize imgs_map and resp_map for tool calling (align with agent_verl.py)
        imgs_map = {}
        resp_map = {}
        observation_counter = 0  # for generating observation_N naming
        image_type = None

        # Process original images and save to imgs_map (align with agent_verl.py)
        if image_data:
            # Handle the case where image_data might be a list or single image
            images_to_process = (
                image_data if isinstance(image_data, list) else [image_data]
            )
            for i, img in enumerate(images_to_process):
                # Skip None images
                if img is None:
                    continue

                # Generate path for original image
                from arm_agent.utils import generate_identifier_path_name
                import os

                # actually the img is a PIL.Image.Image object, so actually just "image_type = 'png'" is enough
                if (
                    img is not None
                    and hasattr(img, "format")
                    and img.format is not None
                ):
                    image_type = img.format.lower()
                    if image_type == "jpeg":
                        image_type = "jpg"
                else:
                    image_type = "png"

                # Create temp directory if not exists
                TOOL_CALL_IMG_TEMP_DIR = os.getenv(
                    "TOOL_CALL_IMG_TEMP", "/tmp/tool_call_images"
                )
                os.makedirs(TOOL_CALL_IMG_TEMP_DIR, exist_ok=True)

                # Save original image
                image_name = f"original_image" if i == 0 else f"original_image_{i}"
                path_name = generate_identifier_path_name()
                image_path = os.path.join(
                    TOOL_CALL_IMG_TEMP_DIR, f"{path_name}.{image_type}"
                )
                img.save(image_path)
                imgs_map[image_name] = image_path

        metrics = {}
        request_id = uuid4().hex
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            print(f"raw_prompt: {raw_prompt}")
            print(f"self.apply_chat_template_kwargs: {self.apply_chat_template_kwargs}")

            model_inputs = self.processor(
                text=[raw_prompt], images=image_data, return_tensors="pt"
            )
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:

            # # First get the raw prompt text
            # raw_prompt = await self.loop.run_in_executor(
            #     None,
            #     lambda: self.tokenizer.apply_chat_template(
            #         messages,
            #         tools=self.tool_schemas,
            #         add_generation_prompt=True,
            #         tokenize=False,  # Get text, not token IDs
            #         **self.apply_chat_template_kwargs,
            #     ),
            # )

            # # Then tokenize the raw prompt
            # prompt_ids = await self.loop.run_in_executor(
            #     None,
            #     lambda: self.tokenizer.encode(raw_prompt, add_special_tokens=False),
            # )
            raise NotImplementedError("Not implemented, processor should not be None!")

        # [NEW]
        # logger.info(
        #     f"verl/experimental/agent_loop/tool_agent_loop.py:async def run():raw_prompt:\n{raw_prompt}"
        # )

        response_mask, response_logprobs = [], []
        tools_kwargs = kwargs.get("tools_kwargs", {})

        user_turns, assistant_turns = 0, 0

        # Initialize metrics for this rollout
        from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics

        metrics = AgentLoopMetrics()
        metrics.tool_calls_count = 0
        metrics.successful_tool_calls = 0
        metrics.failed_tool_calls = 0

        # Initialize timing dict for simple_timer
        timing_dict = {}

        # Initialize response_text
        response_text = ""

        # Simple timing log
        import time

        logger.info(f"[ToolAgent] START - max_turns={self.max_assistant_turns}")

        while True:
            logger.info(f"[ToolAgent] Turn {assistant_turns + 1}: Generation START")
            gen_t0 = time.time()
            with simple_timer("generate_sequences", timing_dict):
                output = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=image_data,
                )
            response_ids = output.token_ids
            prompt_ids += response_ids
            response_mask += [1] * len(response_ids)
            if output.log_probs:
                response_logprobs += output.log_probs
            logger.info(
                f"[ToolAgent] Turn {assistant_turns + 1}: Generation DONE ({time.time()-gen_t0:.1f}s)"
            )
            assistant_turns += 1

            # # reach max response length
            # if len(response_mask) >= self.response_length:
            #     logger.info(f"[ToolAgent] Max response length reached, EXIT")
            #     break
            # logger.info(f"len(response_mask): {len(response_mask)}, self.response_length: {self.response_length}, EXIT")

            _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)

            # Check for parsing failures by counting tool call tags vs successful parses
            response_text = await self.loop.run_in_executor(
                None, self.tokenizer.decode, response_ids
            )

            # Count total tool call tags in the response
            import re

            tool_call_tags = len(re.findall(r"<tool_call>", response_text))

            if tool_call_tags > 0:
                # Some tool calls were attempted
                if tool_call_tags > len(tool_calls):
                    # Some tool calls failed to parse
                    parsing_failures = tool_call_tags - len(tool_calls)
                    metrics.failed_tool_calls += parsing_failures
                    # logger.warning(
                    #     f"Tool call parsing failed: {parsing_failures} out of {tool_call_tags} tool calls failed to parse"
                    # )

            if not tool_calls:
                logger.info(f"[ToolAgent] No tool calls detected, EXIT")
                break

            # reach max user turns
            if self.max_user_turns and user_turns >= self.max_user_turns:
                logger.info(f"[ToolAgent] Max user turns reached, EXIT, self.max_user_turns: {self.max_user_turns}, user_turns: {user_turns}")
                break

            # reach max assistant turns
            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                logger.info(f"[ToolAgent] Max turns reached, EXIT, self.max_assistant_turns: {self.max_assistant_turns}, assistant_turns: {assistant_turns}")
                # break
                tool_responses = [ToolResponse(text=NOT_ALLOWED_TOOL_CALL_MESSAGE)]
                # break
            else:
                # call tools
                tasks = []
                for tool_call in tool_calls[: self.max_parallel_calls]:
                    tasks.append(
                        self._call_tool(tool_call, tools_kwargs, imgs_map, resp_map)
                    )

                # Update tool call metrics for this rollout
                current_tool_calls = len(tool_calls[: self.max_parallel_calls])
                logger.info(
                    f"[ToolAgent] Turn {assistant_turns}: Executing {current_tool_calls} tool(s)..."
                )

                tool_t0 = time.time()
                with simple_timer("tool_calls", timing_dict):
                    tool_responses = await asyncio.gather(*tasks)
                logger.info(
                    f"[ToolAgent] Turn {assistant_turns}: Tool execution DONE ({time.time()-tool_t0:.1f}s)"
                )

                # Debug: Print tool responses (simplified)
                # print(f"DEBUG: Number of tool responses: {len(tool_responses)}")
                # for i, resp in enumerate(tool_responses):
                #     if hasattr(resp, "image") and resp.image:
                #         print(
                #             f"DEBUG: tool_responses[{i}] has image data: {type(resp.image)}"
                #         )

                # Count successful and failed tool calls for this rollout
                # TODO: need some check here
                successful_calls = sum(
                    1
                    for resp in tool_responses
                    if not isinstance(resp, Exception)
                    and not (
                        hasattr(resp, "text")
                        and resp.text
                        and resp.text.startswith("Error when executing tool:")
                    )
                )
                failed_calls = len(tool_responses) - successful_calls

                # Update metrics
                metrics.tool_calls_count += len(tool_responses)  # Actual number of tools executed
                metrics.successful_tool_calls += successful_calls
                metrics.failed_tool_calls += failed_calls

                logger.info(
                    f"[TOOL_CALLS] Results: {successful_calls} successful, {failed_calls} failed"
                )

                # if any(isinstance(item, Exception) for item in tool_responses):
                #     break

            # Extract messages and update multi_modal_data
            tool_messages = []
            new_images_this_turn = []
            for tool_response in tool_responses:
                # Create message from tool response
                if tool_response.image or tool_response.video:
                    # Multi-modal content with structured format
                    content = []
                    if tool_response.image:
                        content.append({"type": "image"})
                    if tool_response.video:
                        content.append({"type": "video"})
                    if tool_response.text:
                        content.append({"type": "text", "text": tool_response.text})
                    message = {"role": "tool", "content": content}
                else:
                    # Text-only content
                    message = {"role": "tool", "content": tool_response.text or ""}

                tool_messages.append(message)

                # Handle image data
                # if tool_response.image:
                #     print(f"DEBUG: Adding tool response images to image_data")

                if tool_response.image:
                    # if image_data is None:
                    #     image_data = []
                    # elif not isinstance(image_data, list):
                    #     image_data = [image_data]

                    # Add new image data
                    if isinstance(tool_response.image, list):
                        # image_data.extend(tool_response.image)
                        new_images_this_turn.extend(tool_response.image)
                    else:
                        # image_data.append(tool_response.image)
                        new_images_this_turn.append(tool_response.image)

                # Handle video data
                if tool_response.video:
                    # Currently not supported, raise informative error
                    # logger.warning(
                    #     "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                    # )
                    raise NotImplementedError(
                        "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                    )

            # append tool_response_ids
            if self.processor is not None:
                raw_tool_response = await self.loop.run_in_executor(
                    None,
                    lambda messages=tool_messages: self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False,
                        **self.apply_chat_template_kwargs,
                    ),
                )
                # Use only the new images from this turn for processing tool responses
                current_images = new_images_this_turn if new_images_this_turn else None
                model_inputs = self.processor(
                    text=[raw_tool_response], images=current_images, return_tensors="pt"
                )
                tool_response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
            else:
                # tool_response_ids = await self.loop.run_in_executor(
                #     None,
                #     lambda messages=tool_messages: self.tokenizer.apply_chat_template(
                #         messages,
                #         add_generation_prompt=True,
                #         tokenize=True,
                #         **self.apply_chat_template_kwargs,
                #     ),
                # )
                raise NotImplementedError("Not implemented, processor should not be None!")
            # DEBUG: decode whole ids
            # tool_response_ids_decoded = self.tokenizer.decode(
            #     tool_response_ids, skip_special_tokens=False
            # )
            # print(f"DEBUG: tool_response_ids_decoded: {tool_response_ids_decoded}")
            # tool_response_ids = tool_response_ids[len(self.system_prompt) :]
            # print(f"self.system_prompt: {self.system_prompt}")
            # print(f"len(self.system_prompt): {len(self.system_prompt)}")
            tool_response_ids = tool_response_ids[len(self.system_prompt) :]
            # Decode separately
            # tool_response_ids_decoded = self.tokenizer.decode(
            #     tool_response_ids, skip_special_tokens=False
            # )
            # print(f"tool_response_ids_decoded: {tool_response_ids_decoded}")
            # Count <|image_pad|>
            # image_pad_count = tool_response_ids_decoded.count("<|image_pad|>")
            # print(f"image_pad_count: {image_pad_count}")
            # Decode tool_response_ids again
            # tool_response_ids_decoded = self.tokenizer.decode(tool_response_ids, skip_special_tokens=False)
            # print(f"tool_response_ids_decoded: {tool_response_ids_decoded}")
            # print(f"self.system_prompt: {self.system_prompt}")
            # print(f"len(self.system_prompt): {len(self.system_prompt)}")
            # print(f"DEBUG: tool_response_ids after: {tool_response_ids}")

            # NOTE: last turn should not be user turn, or the EOS token reward
            # can't be propagated to previous token in GAE.
            # [TAG-Chris] 2025.10.17
            # TODO: need to change to prompt_length here?
            # self.response_length = self.prompt_length
            print(f"self.response_length: {self.response_length}")
            if len(response_mask) + len(tool_response_ids) >= self.response_length:
                logger.info(f"len(response_mask): {len(response_mask)}, len(tool_response_ids): {len(tool_response_ids)}, self.response_length: {self.response_length}, EXIT")
                break

            prompt_ids += tool_response_ids
            # Image data should be added same time to text
            for tool_response in tool_responses:
                # first save for image context
                if tool_response.image:
                    for img in tool_response.image:
                        # Skip None images
                        if img is None:
                            continue

                        # Generate new image key name
                        observation_counter += 1
                        new_image_name = f"observation_{observation_counter}"
                        from arm_agent.utils import generate_identifier_path_name
                        import os

                        path_name = generate_identifier_path_name()
                        # Use the same image type as original image, fallback to png
                        img_type = image_type if image_type else "png"
                        TOOL_CALL_IMG_TEMP_DIR = os.getenv(
                            "TOOL_CALL_IMG_TEMP", "/tmp/tool_call_images"
                        )
                        image_path = os.path.join(
                            TOOL_CALL_IMG_TEMP_DIR, f"{path_name}.{img_type}"
                        )
                        img.save(image_path)
                        imgs_map[new_image_name] = image_path
                # save for image data
                if tool_response.image:
                    if image_data is None:
                        image_data = []
                    elif not isinstance(image_data, list):
                        image_data = [image_data]
                    # Add new image data
                    if isinstance(tool_response.image, list):
                        image_data.extend(tool_response.image)
                    else:
                        image_data.append(tool_response.image)
            
            # DEBUG: decode whole prompt_ids
            # prompt_ids_decoded = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
            # print(f"DEBUG: prompt_ids_decoded: {prompt_ids_decoded}")
            # print(f"DEBUG: prompt_ids: {prompt_ids}")
            response_mask += [0] * len(tool_response_ids)
            if response_logprobs:
                response_logprobs += [0.0] * len(tool_response_ids)
            user_turns += 1

        response_ids = prompt_ids[-len(response_mask) :]
        prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]

        multi_modal_data = {"image": image_data} if image_data is not None else {}

        # Update timing metrics
        metrics.generate_sequences = timing_dict.get("generate_sequences", 0.0)
        metrics.tool_calls = timing_dict.get("tool_calls", 0.0)

        logger.info(
            f"[ToolAgent] COMPLETE - turns={assistant_turns}, tools={metrics.tool_calls_count}, success={metrics.successful_tool_calls}"
        )

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=(
                response_logprobs[: self.response_length] if response_logprobs else None
            ),
            num_turns=user_turns + assistant_turns + 1,
            metrics=metrics,
        )
        return output

    async def _call_tool(
        self,
        tool_call: FunctionCall,
        tools_kwargs: dict[str, Any],
        imgs_map: dict = None,
        resp_map: dict = None,
    ) -> ToolResponse:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})

            # Build response_store with imgs_map and resp_map (align with agent_verl.py)
            response_store = {}
            if imgs_map:
                response_store["imgs_map"] = imgs_map
            if resp_map:
                response_store["resp_map"] = resp_map

            # Merge with existing create_kwargs
            create_kwargs = kwargs.get("create_kwargs", {})
            create_kwargs["response_store"] = response_store

            # logger.info(f"verl/experimental/agent_loop/tool_agent_loop.py:async def _call_tool():kwargs:\n{kwargs}")
            instance_id, _ = await tool.create(create_kwargs=create_kwargs)
            tool_execution_response, _, _ = await tool.execute(instance_id, tool_args)
        except Exception as e:
            # logger.warning(f"Error when executing tool: {e}")
            return ToolResponse(
                text=f"Error when executing tool: {e}",
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if (
            tool_response_text
            and len(tool_response_text) > self.max_tool_response_length
        ):
            if self.tool_response_truncate_side == "left":
                tool_response_text = (
                    tool_response_text[: self.max_tool_response_length]
                    + "...(truncated)"
                )
            elif self.tool_response_truncate_side == "right":
                tool_response_text = (
                    "(truncated)..."
                    + tool_response_text[-self.max_tool_response_length :]
                )
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = (
                    tool_response_text[:length]
                    + "...(truncated)..."
                    + tool_response_text[-length:]
                )

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs)
