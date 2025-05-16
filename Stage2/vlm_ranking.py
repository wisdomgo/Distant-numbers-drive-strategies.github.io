import os
import base64
import asyncio
import csv
import json
import time
import random
from openai import AsyncOpenAI, RateLimitError

def encode_image_to_base64(image_path):
    """
    将图像文件编码为Base64字符串。
    """
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_image

async def send_image_to_vlm_api(semaphore, 
                                client: AsyncOpenAI, 
                                image_path, 
                                text_prompt,
                                folder):
    """
    发送一张图片及提示词到VLM API，获取响应内容。
    返回 (run_id, folder_id, score)。
    """
    folder_id = folder.split('/')[-1]  # 提取文件夹编号
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
                for _ in range(3):  # 重试3次
                    response = await client.chat.completions.create(
                        model="Qwen/Qwen2-VL-72B-Instruct",
                        messages=messages,
                        stream=False
                    )
                    response_content = response.choices[0].message.content.strip()
                    response_list.append(response_content)
                print(f"Finish request for {image_path}, The score of cluster {folder_id} by VLM is: {response_list}")
                
                # 解析响应，假设每个响应是一个数值字符串
                scores = []
                for response_content in response_list:
                    try:
                        # 尝试将响应内容直接转换为浮点数
                        score = float(response_content)
                        scores.append(score)
                    except ValueError:
                        print(f"Error in resolving the content: {response_content}")
                        continue

                # 计算平均分数
                if scores:
                    avg_score = sum(scores) / len(scores)
                else:
                    avg_score = 0.0

                return folder_id, round(avg_score, 4)

            except RateLimitError:
                print(f"Image {image_path} reach rate limit. Retry in 40 seconds...")
                await asyncio.sleep(40)
            except Exception as e:
                print(f"Error when requesting for {image_path}: {e}")
                return folder_id, 0.0

async def process_images_in_folder(folder_path, run_id, text_prompt, semaphore, client, max_concurrent_requests=5):
    """
    遍历指定文件夹中的23个子文件夹，发送API请求，收集分数并排序。
    返回 (run_id, final_sorted_order)。
    """
    tasks = []
    selected_images = {}
    for folder_id in range(23):
        folder = os.path.join(folder_path, str(folder_id))
        if os.path.isdir(folder):
            image_files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if image_files:
                # 随机选择一张图片
                chosen_file = random.choice(image_files)
                image_path = os.path.join(folder, chosen_file)
                selected_images[str(folder_id)] = image_path
                print(f"[Run {run_id}] choose {chosen_file} from cluster {folder_id}.")
                tasks.append(asyncio.create_task(send_image_to_vlm_api(semaphore, client, image_path, text_prompt, folder)))
            else:
                print(f"[Run {run_id}] fail to find PNG/JPG/JPEG files from cluster {folder_id}.")
                # 如果未找到图片，分数默认为0
                selected_images[str(folder_id)] = None
                tasks.append(asyncio.create_task(asyncio.sleep(0, result=(str(folder_id), 0.0))))
        else:
            print(f"[Run {run_id}] cluster {folder_id} doesn't exist.")
            selected_images[str(folder_id)] = None
            tasks.append(asyncio.create_task(asyncio.sleep(0, result=(str(folder_id), 0.0))))
    
    # 等待所有任务完成
    responses = await asyncio.gather(*tasks)
    
    # 生成字典：folder_id -> score
    image_scores = {folder_id: score for folder_id, score in responses}
    
    # 排序逻辑：根据分数降序排序，分数相差在10分之内的归为同一档次
    sorted_items = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"[Run {run_id}] Ranked score list: {sorted_items}")

    sorted_order = []
    current_group = []
    previous_score = None
    
    for folder_id, score in sorted_items:
        if previous_score is None:
            current_group = [folder_id]
            previous_score = score
        else:
            if abs(previous_score - score) <= 10:
                current_group.append(folder_id)
            else:
                if len(current_group) > 1:
                    sorted_order.append("=".join(current_group))
                else:
                    sorted_order.append(current_group[0])
                current_group = [folder_id]
                previous_score = score
    
    # 添加最后一个分组
    if current_group:
        if len(current_group) > 1:
            sorted_order.append("=".join(current_group))
        else:
            sorted_order.append(current_group[0])
    
    # 生成最终排序字符串，例如："0>1>2>4=3=6>9=12>17>11>13>10>7>20=19=21>5>14>15>16>18>8>22"
    final_sorted_order = ">".join(sorted_order)
    print(f"[Run {run_id}] Final sorted order: {final_sorted_order}")
    
    return run_id, final_sorted_order

async def main(runs):
    folder_path = "/mnt/Final/dev-measure/data/cluster_sichuan"
    output_path = "/mnt/Final/dev-measure/Stage2"
    output_csv = os.path.join(output_path, "sorted_images_all_runs.csv")
    text_prompt = (
        "你是一个遥感图像领域的经济学专家，我要求你根据每张图片所代表地区的经济发展水平，对这张图片进行打分，分数范围为0到100，"
        "分数越高表示经济发展水平越高。一般而言，城市地区应打分80到100分，农村地区应打分20到79分，山地、高原和无人区应打分0到19分。"
        "介于这几种情况之间的地形，你需要根据你的判断进行打分。输出的时候必须只输出一个得分值，不要有任何其他的信息。"
    )
    max_concurrent_requests = 10
    
    # 初始化AsyncOpenAI客户端
    client = AsyncOpenAI(
        api_key="sk-jmgaqbgwgcnkgpkvqjdjogcupxhmbijuwcmzzsyokqttsqpu",  # 请替换为您的实际API密钥
        base_url="https://api.siliconflow.cn/v1"
    )
    
    # 定义信号量以控制并发请求的数量
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    
    # 创建3个运行任务
    tasks = []
    for run_id in range(1, runs + 1):
        tasks.append(asyncio.create_task(process_images_in_folder(folder_path, run_id, text_prompt, semaphore, client, max_concurrent_requests)))
    
    # 等待所有运行完成
    run_results = await asyncio.gather(*tasks)
    
    # 将所有运行结果写入同一个CSV文件
    try:
        with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            # 写入表头
            csv_writer.writerow(["Run ID", "Sorted Order"])
            # 写入每次运行的结果
            for run_id, sorted_order in run_results:
                csv_writer.writerow([run_id, sorted_order])
        print(f"All POGs have been written into {output_csv}.")
    except Exception as e:
        print(f"Error when writing into csv file: {e}.")

if __name__ == "__main__":
    runs = 1
    begin = time.time()
    asyncio.run(main(runs=runs))
    print(f'Successfully run {runs} rounds in {time.time()-begin} seconds.')