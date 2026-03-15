#!/usr/bin/env python3
import aiohttp
import asyncio
import json
import sys
import time
import os

# TEST_PLANNING 模式下的输入文件
TEST_PLANNING = True
PLANNING_FILTER = False
if TEST_PLANNING:
    if PLANNING_FILTER:
        IN_FILE = "/data/temp/planning_filtered_t5.json"
    else:
        IN_FILE = "/data/temp/planning.json"
else:
    IN_FILE = None

# 并发控制
MAX_CONCURRENT = int(sys.argv[1])

# 对齐 curl 中使用的模型路径
MODEL_NAME = (
    "/mnt/tidal-alsh01/dataset/redone/heshien/red_mg_dllm/"
    "Red-dLLM_30BA3B_pt_s27000_sft_s5100_sft_v2_compress_text_20260306002212/hf_800"
)


def load_all_planning_samples(file_path):
    """
    从 JSON 文件加载所有 TEST_PLANNING 模式下的样本
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


async def query_sglang_async(session, system, user, idx, total, port=20000):
    """
    异步查询 SGLang /v1/chat/completions endpoint
    """
    url = f"http://localhost:{port}/v1/chat/completions"
    content = None
    
    messages = [{"role": "user", "content": user}]
    if system:
        messages.insert(0, {"role": "system", "content": system})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 256,
        "stream": False
    }
    
    start_time = time.time()
    
    try:
        async with session.post(url, json=payload) as response:
            response.raise_for_status()
            result = await response.json()
            
            end_time = time.time()
            total_latency = (end_time - start_time) * 1000  # 转换为毫秒
            
            # 只打印 EXTRACTED CONTENT
            print(f"\n[Sample {idx + 1}/{total}] EXTRACTED CONTENT:")
            print("-" * 60)
            
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                content = choice["message"]["content"]
                print(content)
            else:
                print("No content in response")
            
            return {
                "idx": idx,
                "success": True,
                "latency_ms": total_latency,
                "content": content,
            }
            
    except aiohttp.ClientConnectionError:
        print(f"\n[Sample {idx + 1}/{total}] Error: Cannot connect to localhost:{port}. Is SGLang server running?")
        return {"idx": idx, "success": False, "error": "Connection error"}
    except aiohttp.ClientResponseError as e:
        print(f"\n[Sample {idx + 1}/{total}] HTTP Error: {e.status} - {e.message}")
        return {"idx": idx, "success": False, "error": f"HTTP {e.status}"}
    except Exception as e:
        print(f"\n[Sample {idx + 1}/{total}] Error: {e}")
        return {"idx": idx, "success": False, "error": str(e)}


async def process_samples_with_limit(session, samples, semaphore, port=20000):
    """
    使用信号量限制并发数处理样本
    """
    total = len(samples)
    
    async def process_one(idx, sample):
        async with semaphore:
            if PLANNING_FILTER:
                messages = sample.get('messages', '')
                system = messages[0].get('content')
                user = messages[1].get('content')
            else:
                system = sample.get('planning_system', '')
                user = sample.get('planning_user', '')
            return await query_sglang_async(session, system, user, idx, total, port)
    
    # 创建所有任务
    start = 0
    tasks = [process_one(idx, sample) for idx, sample in enumerate(samples[start:start+20])]
    
    # 等待所有任务完成
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results


async def main():
    # 默认使用 TEST_PLANNING 模式
    if TEST_PLANNING and IN_FILE and os.path.exists(IN_FILE):
        print(f"TEST_PLANNING mode: Loading all samples from {IN_FILE}")
        samples = load_all_planning_samples(IN_FILE)
        total_samples = len(samples)
        print(f"Total samples to process: {total_samples}")
        print(f"Max concurrent requests: {MAX_CONCURRENT}")
        print("=" * 60)
    else:
        print("Error: IN_FILE not set or file not found")
        sys.exit(1)
    
    # samples = [samples[14]] * 100
    # xxx = {}
    # xxx['planning_system'] = "说中文"
    # xxx['planning_user'] = "iphone 18 值得购买吗"
    # samples = [xxx] * 100

    # 创建信号量限制并发
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    # 创建 aiohttp session
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        start_time = time.time()
        
        # 处理所有样本
        results = await process_samples_with_limit(session, samples, semaphore)

        end_time = time.time()
        total_time = end_time - start_time

    # 统计结果
    success_results = [r for r in results if isinstance(r, dict) and r.get("success")]
    success_count = len(success_results)
    failed_count = total_samples - success_count
    
    # 计算成功请求的单条耗时统计
    latencies = [r["latency_ms"] for r in success_results]
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        # 计算中位数
        sorted_latencies = sorted(latencies)
        mid = len(sorted_latencies) // 2
        if len(sorted_latencies) % 2 == 0:
            median_latency = (sorted_latencies[mid - 1] + sorted_latencies[mid]) / 2
        else:
            median_latency = sorted_latencies[mid]
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    print(f"Total samples: {total_samples}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Total time: {total_time:.2f} seconds")
    if total_samples > 0:
        print(f"Overall throughput: {total_samples/total_time:.2f} samples/second")
    print("-" * 60)
    print("LATENCY STATISTICS (successful requests):")
    if latencies:
        print(f"  Average: {avg_latency:.2f} ms")
        print(f"  Median:  {median_latency:.2f} ms")
        print(f"  Min:     {min_latency:.2f} ms")
        print(f"  Max:     {max_latency:.2f} ms")
    else:
        print("  No successful requests to calculate statistics")

    ordered_results = sorted(
        [r for r in results if isinstance(r, dict)],
        key=lambda item: item["idx"],
    )
    print("\n" + "=" * 60)
    print("CONTENT IN ORDER:")
    print("=" * 60)
    for result in ordered_results:
        print(f"\n[Sample {result['idx'] + 1}]")
        print("-" * 60)
        if result.get("success"):
            print(result.get("content") or "")
        else:
            print(f"Request failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    asyncio.run(main())
