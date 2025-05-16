import os
import base64
import asyncio
import aiohttp
import csv
import json
import time
from openai import AsyncOpenAI, RateLimitError

def encode_image_to_base64(image_path):
    """
    encode an image file to a base64 string.
    """
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_image

async def send_image_to_vlm_api(semaphore, 
                                client: AsyncOpenAI, 
                                image_path, 
                                text_prompt):
    """
    Sending an image and a text prompt to the VLM API asynchronously,
    return the response content.
    """
    base64_image = encode_image_to_base64(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "auto"
                    }
                },
                {
                    "type": "text",
                    "text": text_prompt
                }
            ]
        }
    ]

    while True:
        async with semaphore:
            try:
                response_list = []
                for _ in range(3):
                    response = await client.chat.completions.create(
                        model="Qwen/Qwen2-VL-72B-Instruct",
                        messages=messages,
                        stream=False
                    )
                    response_list.append(response.choices[0].message.content)
                print(f"Finish requesting for image {image_path}")
                return image_path, response_list
            except RateLimitError:
                print(f"Rate limit exceeded for image {image_path}. Retrying in 40 secs...")
                await asyncio.sleep(40)
            except Exception as e:
                print(f"An error occurred for image {image_path}: {e}")
                return image_path, []

async def process_images_in_folder(folder_path, output_csv, text_prompt, max_concurrent_requests=5):
    """
    Traverse all images in a folder,
    generate a three-class label for each image and save to .csv
    """
    # Using the AsyncOpenAI client
    client = AsyncOpenAI(
        api_key="sk-oqoekrdbaghvlsjjohahlxdtvlrpfeplqbxmbkhdibwzonyu",  # Replace with your actual API key
        base_url="https://api.siliconflow.cn/v1"
    )
    
    # Define the maximum number of concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    
    tasks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            tasks.append(asyncio.create_task(send_image_to_vlm_api(semaphore, client, image_path, text_prompt)))
            # No need to sleep here as concurrency is controlled by semaphore

    # Do all tasks and collect results in `responses`
    responses = await asyncio.gather(*tasks)
    with open(output_csv, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Image ID", "Urban", "Rural", "Mountain", "Highland"])
        for (filename, response_list) in responses:
            image_id = os.path.splitext(os.path.basename(filename))[0]
            
            aggr_response_list = []
            for response in response_list:
                if response.startswith('[') and response.endswith(']'):
                    cleaned_response = response.strip("[]")
                    try:
                        prob = [float(prob) for prob in cleaned_response.split(",")]
                        total = sum(prob)
                        if total > 0:
                            normalized_prob = [score / total for score in prob]
                        else:
                            normalized_prob = [0, 0, 0, 0]
                        aggr_response_list.append(normalized_prob)
                    except ValueError:
                        continue
                else:
                    continue
            if aggr_response_list:
                # Get averaged probability for each class
                avg_prob = [sum(x)/len(aggr_response_list) for x in zip(*aggr_response_list)]
            else:
                avg_prob = [0, 0, 0, 0]
            csv_writer.writerow([image_id, *avg_prob])
        print(f"Finished processing {len(responses)} images.")

if __name__ == "__main__":
    folder_path = "mountains"
    output_csv = folder_path+"/pseudo_labels_" + folder_path.split("/")[-1] + ".csv"
    text_prompt = "你是一个卫星图像专家和经济学专家。我将给你一张中国四川省山区的卫星图片，你需要关于它属于城市地区、农村地区、山区无人区、高原无人区打分，分数越高表明越属于这一类。打分标准是城市、农村、山区或高原占整个卫星图像的占比。你需要输出一个列表，列表的第一、二、三、四个元素分别为属于城市地区、农村地区、山区无人区、高原无人区的分数，每一项都是0到100的值。对于这张图片而言，山区的分数应该最高。输出时不要附带任何解释信息，只输出形如[a,b,c,d]的分数列表。"
    max_concurrent_requests = 10
    asyncio.run(process_images_in_folder(folder_path, output_csv, text_prompt, max_concurrent_requests))
