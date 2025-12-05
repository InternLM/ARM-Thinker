import os
from arm_agent.agent_verl import VerlAgent
from arm_agent.utils import encode_image_file
from dotenv import load_dotenv
import json
import requests
from eval_utils.utils import FIXED_COT_PROMPT
import base64
import traceback

load_dotenv()

def message_format_with_image_file(user_prompt, image_file, image_type="png"):
    base64_image = encode_image_file(image_file)
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_type};base64,{base64_image}"
                    },
                },
                {"type": "text", "text": user_prompt},
            ],
        }
    ]


def main():
    # set environment variable for tool call image temp
    os.environ["TOOL_CALL_IMG_TEMP"] = "/tmp/agent_images"
    os.makedirs("/tmp/agent_images", exist_ok=True)

    # api config
    api_base = "http://10.102.198.50:36001/v1"
    api_key = "EMPTY"
    model_name = "Qwen/Qwen3-VL-8B-Instruct"

    # test if the api is accessible
    print(f"try to access {api_base}/models")
    response = requests.get(f"{api_base}/models")
    print(response.json())
    print(f"access {api_base}/models success")

    # tool config path
    tool_config_path = "arm_agent/tests/config/test_image_zoom_in_tool_config.yaml"

    # initialize VerlAgent
    agent = VerlAgent(
        api_base=api_base,
        api_key=api_key,
        tool_config_path=tool_config_path,
        temperature=0.7,
        max_tokens=2048,
        model_name=model_name,
        max_round=4,
        max_tool_response_length=2048,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_k=20,
        top_p=0.8,
        system_template_type="Qwen3VLSystemTemplateWithTools",
    )

    # test image file
    image_file = "arm_agent/tests/test_assert/wukang_road.png"

    # user prompt
    # copy from https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/think_with_images.ipynb
    user_prompt = f"""\
Your role is that of a research assistant specializing in visual information. Answer questions about images by looking at them closely and then using research tools (Only the `crop_image` tool is available). Please follow this structured thinking process and show your work.

Start an iterative loop for each question:

- **First, look closely:** Begin with a detailed description of the image, paying attention to the user's question. List what you can tell just by looking, and what you'll need to look up.
- **Next, find information:** Use a tool to research the things you need to find out.
- **Then, review the findings:** Carefully analyze what the tool tells you and decide on your next action.

Continue this loop until your research is complete.

To finish, bring everything together in a clear, synthesized answer that fully responds to the user's question.

---
Now, here is the question: Where was the picture taken?
"""

    # format user messages
    user_messages = message_format_with_image_file(user_prompt, image_file)

    # run agent
    result, tool_calls = agent.run(user_messages)

    # Option1: Not process the result, then the image in result is base64 encoded, the result.json will be large.
    option_for_process_image_in_result = "save_to_path" # "base64" or "save_to_path" or "hidden", default is "base64"
    if option_for_process_image_in_result == "hidden":
        for msg in result:
            role = msg.get("role", "")
            content = msg.get("content", "")
            # print image part, base64 is too long to print
            if role == "user" and isinstance(content, list):
                for item in content:
                    if "type" in item and item["type"] == "image_url":
                        item["image_url"]["url"] = "hidden"
            print(f"{role.upper()}:\n{content}\n")
    elif option_for_process_image_in_result == "base64":
        pass
    elif option_for_process_image_in_result == "save_to_path": 
        save_img_root = "arm_agent/tests/result_images/test_agent_image_zoom_in"
        os.makedirs(save_img_root, exist_ok=True)
        # clear the old images
        for file in os.listdir(save_img_root):
            os.remove(os.path.join(save_img_root, file))
        img_counter = 0
        for msg in result:
            content = msg.get("content", "")
            if not isinstance(content, list):
                continue
            for itemm in content:
                if not (isinstance(itemm, dict) and itemm.get("type") == "image_url"):
                    continue
                image_url = itemm.get("image_url", {}).get("url", "")
                if not image_url:
                    continue
                try:
                    # judge whether it is base64 or url
                    if image_url.startswith("data:image"):
                        header, b64_data = image_url.split(",", 1)
                        ext = header.split("/")[1].split(";")[0]  # extract extension
                        img_bytes = base64.b64decode(b64_data)
                        img_name = f"img_{img_counter:04d}.{ext}"
                    else:
                        raise ValueError(
                            f"Invalid image url: {image_url}, can only process base64 image url"
                        )
                    # save image
                    img_path = os.path.join(save_img_root, img_name)
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)

                    # replace image_url["url"] with relative path
                    itemm["image_url"]["url"] = f"{img_name}"
                    img_counter += 1

                    print(f"[Saved] {img_path}, {itemm['image_url']['url']}")
                except Exception as e:
                    print(f"Error: {e}")
                    print("Traceback:\n" + traceback.format_exc())
                    raise ValueError(f"Error: {e}")
    else:
        raise ValueError(f"Invalid option: {option_for_process_image_in_result}")
    # save result
    with open("arm_agent/tests/result.json", "w", encoding="utf-8") as f:
        json.dump(result, f)
    print("result saved to result.json")
    print(f"tool calls: {tool_calls}")
    print(result[-1]["content"])


if __name__ == "__main__":
    main()
