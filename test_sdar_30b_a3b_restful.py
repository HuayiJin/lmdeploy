"""
SDAR-30B-A3B (JetLM/SDAR-30B-A3B-Sci) RESTful API 测试脚本

服务启动方式（端口 20000，pytorch 后端，tp=2）：
    lmdeploy serve api_server <model_path>/JetLM/SDAR-30B-A3B-Sci \
        --backend pytorch \
        --tp 2 \
        --server-port 20000 \
        --dllm-block-length 4 \
        --dllm-denoising-steps 4 \
        --dllm-confidence-threshold 0.9

运行测试：
    pytest autotest/tools/restful/test_sdar_30b_a3b_restful.py \
        --config autotest/config_h.yml -v
"""

import os
import subprocess
import time

import psutil
import pytest
import requests
from openai import OpenAI

from lmdeploy.serve.openai.api_client import APIClient

# ── 服务配置 ──────────────────────────────────────────────────────────
MODEL_REPO = '/mnt/tidal-alsh01/dataset/redone/heshien/red_mg_dllm/Red-dLLM_30BA3B_pt_s27000_sft_s5100_sft_v2_compress_text_20260306002212/hf_800'
BACKEND = 'pytorch'
TP = 1
SERVER_PORT = 20000
BASE_URL = f'http://127.0.0.1:{SERVER_PORT}'
SERVER_TIMEOUT = 1200  # 等待服务启动最长时间（秒）

# SDAR 特有的服务端启动参数（来自 config_utils.py sdar 模型逻辑）
DLLM_EXTRA_PARAMS = {
    'dllm-block-length': 4,
    'dllm-denoising-steps': 4,
    'dllm-confidence-threshold': 0.9,
}

# SDAR 推理采样参数（temperature=1.0, top_p=1.0, top_k=-1 表示禁用 top_k 过滤）
SDAR_SAMPLING = {
    'temperature': 1.0,
    'top_p': 1.0,
    'extra_body': {'top_k': -1},
}

# 基础功能测试用例
BASIC_TEST_CASES = [
    {
        'prompt': '你好，请介绍一下你自己。',
        'keywords': [],  # 不做关键词断言，仅验证有输出
    },
    {
        'prompt': '1+1等于几？',
        'keywords': ['2'],
    },
    {
        'prompt': '请用一句话解释什么是大语言模型。',
        'keywords': [],
    },
]

# 多轮对话测试
MULTI_TURN_CASES = [
    '你叫什么名字？',
    '你刚才说的名字是什么？',
]


# ── 工具函数 ──────────────────────────────────────────────────────────

def _health_check(url: str, model_name: str) -> bool:
    """检查服务是否已就绪。"""
    try:
        api_client = APIClient(url)
        available = api_client.available_models
        if not available:
            return False
        messages = [{'role': 'user', 'content': '你好'}]
        for output in api_client.chat_completions_v1(
            model=available[0], messages=messages, top_k=1, max_tokens=16
        ):
            if output.get('code') is not None and output.get('code') != 0:
                return False
            return True
        return False
    except Exception:
        return False


def _start_server(model_path: str, log_file: str) -> int:
    """启动 lmdeploy api_server，返回进程 PID。失败返回 0。"""
    extra_args = ' '.join(
        f'--{k} {v}' for k, v in DLLM_EXTRA_PARAMS.items()
    )
    cmd = (
        f'lmdeploy serve api_server {model_path} '
        f'--backend {BACKEND} '
        f'--tp {TP} '
        f'--server-port {SERVER_PORT} '
        f'--communicator nccl '
        f'--allow-terminate-by-client '
        f'{extra_args}'
    )
    print(f'\n[server] 启动命令:\n  {cmd}\n')

    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    env = os.environ.copy()
    with open(log_file, 'w') as f:
        f.write(f'# 启动命令: {cmd}\n\n')
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=f,
            shell=True,
            text=True,
            env=env,
            encoding='utf-8',
            errors='replace',
            start_new_session=True,
        )

    pid = proc.pid
    start_time = time.time()
    time.sleep(5)

    for _ in range(SERVER_TIMEOUT):
        time.sleep(1)
        if time.time() - start_time >= SERVER_TIMEOUT:
            break
        if _health_check(BASE_URL, MODEL_REPO):
            print(f'[server] 服务已就绪，PID={pid}，耗时 {int(time.time()-start_time)}s')
            return pid
        try:
            rc = proc.wait(timeout=0.5)
            if rc != 0:
                with open(log_file) as lf:
                    print(f'[server] 进程异常退出，日志:\n{lf.read()[-3000:]}')
                return 0
        except subprocess.TimeoutExpired:
            continue

    print(f'[server] 等待超时（{SERVER_TIMEOUT}s）')
    return 0


def _stop_server(pid: int) -> None:
    """通过 /terminate 端点或 SIGTERM 停止服务。"""
    try:
        requests.get(f'{BASE_URL}/terminate', timeout=10)
        time.sleep(3)
    except Exception:
        pass

    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.terminate()
        parent.terminate()
    except psutil.NoSuchProcess:
        pass


# ── Pytest Fixtures ───────────────────────────────────────────────────

@pytest.fixture(scope='module')
def server(config):
    """模块级 fixture：启动服务（模块结束后关闭）。"""
    model_path = os.path.join(config.get('model_path', '/model'), MODEL_REPO)
    log_path = config.get('server_log_path', '/tmp/lmdeploy_test')
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_path, f'sdar_30b_a3b_{SERVER_PORT}_{timestamp}.log')

    pid = _start_server(model_path, log_file)
    assert pid > 0, (
        f'SDAR-30B-A3B 服务启动失败，请检查日志: {log_file}'
    )

    yield {'pid': pid, 'url': BASE_URL, 'log': log_file}

    _stop_server(pid)


@pytest.fixture(scope='module')
def openai_client(server):
    """OpenAI 兼容客户端。"""
    client = OpenAI(api_key='EMPTY', base_url=f'{server["url"]}/v1')
    model_id = client.models.list().data[0].id
    return client, model_id


# ── 测试用例 ──────────────────────────────────────────────────────────

class TestSDARHealthAndModel:
    """服务健康状态与模型信息。"""

    def test_health_endpoint(self, server):
        resp = requests.get(f'{server["url"]}/health', timeout=30)
        assert resp.status_code == 200, f'health 接口异常: {resp.status_code}'

    def test_model_available(self, openai_client):
        client, model_id = openai_client
        assert SDAR_NAME_FRAGMENT in model_id, (
            f'模型名称应包含 SDAR-30B-A3B，实际: {model_id}'
        )

    def test_list_models(self, openai_client):
        client, _ = openai_client
        models = client.models.list()
        assert len(models.data) >= 1, '模型列表为空'


# 辅助常量
SDAR_NAME_FRAGMENT = 'SDAR-30B-A3B'


class TestSDARBasicChat:
    """基础单轮对话功能。"""

    @pytest.mark.parametrize('case', BASIC_TEST_CASES)
    def test_single_turn(self, openai_client, case):
        client, model_id = openai_client
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{'role': 'user', 'content': case['prompt']}],
            max_tokens=512,
            **{k: v for k, v in SDAR_SAMPLING.items() if k != 'extra_body'},
            extra_body=SDAR_SAMPLING['extra_body'],
        )
        content = resp.choices[0].message.content
        assert content and len(content.strip()) > 0, (
            f'空回复，prompt={case["prompt"]!r}'
        )
        for kw in case['keywords']:
            assert kw in content, (
                f'期望关键词 {kw!r} 不在回复中，回复: {content!r}'
            )

    def test_stream_output(self, openai_client):
        client, model_id = openai_client
        stream = client.chat.completions.create(
            model=model_id,
            messages=[{'role': 'user', 'content': '请数数：1到5'}],
            max_tokens=128,
            stream=True,
            **{k: v for k, v in SDAR_SAMPLING.items() if k != 'extra_body'},
            extra_body=SDAR_SAMPLING['extra_body'],
        )
        chunks = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                chunks.append(delta)
        full_text = ''.join(chunks)
        assert len(full_text.strip()) > 0, '流式输出为空'


class TestSDARMultiTurn:
    """多轮对话测试。"""

    def test_multi_turn_context(self, openai_client):
        client, model_id = openai_client
        messages = []
        for prompt in MULTI_TURN_CASES:
            messages.append({'role': 'user', 'content': prompt})
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=256,
                **{k: v for k, v in SDAR_SAMPLING.items() if k != 'extra_body'},
                extra_body=SDAR_SAMPLING['extra_body'],
            )
            reply = resp.choices[0].message.content
            assert reply and len(reply.strip()) > 0, (
                f'多轮对话第 {len(messages)} 轮回复为空'
            )
            messages.append({'role': 'assistant', 'content': reply})


class TestSDARSamplingParams:
    """SDAR 特有采样参数验证。"""

    def test_top_k_disabled(self, openai_client):
        """top_k=-1（禁用）时服务不报错（SDAR 默认配置）。"""
        client, model_id = openai_client
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{'role': 'user', 'content': '简单说一下你的功能'}],
            max_tokens=128,
            temperature=1.0,
            top_p=1.0,
            extra_body={'top_k': -1},
        )
        assert resp.choices[0].message.content, '使用 top_k=-1 时回复为空'

    def test_max_tokens_limit(self, openai_client):
        """验证 max_tokens 参数被正确遵守。"""
        client, model_id = openai_client
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{'role': 'user', 'content': '请尽可能长地自我介绍'}],
            max_tokens=32,
            temperature=1.0,
            top_p=1.0,
            extra_body={'top_k': -1},
        )
        usage = resp.usage
        assert usage.completion_tokens <= 32, (
            f'completion_tokens({usage.completion_tokens}) 超过 max_tokens(32)'
        )


# ── 独立启动入口（非 pytest 调用时直接运行服务）──────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='启动 SDAR-30B-A3B 服务并执行简单冒烟测试'
    )
    parser.add_argument(
        '--model-path',
        default='/mnt/shared-storage-user/llmrazor-share/qa-llm-cicd/'
                'cicd-autotest/eval_resource/model/JetLM/SDAR-30B-A3B-Sci',
        help='模型本地路径',
    )
    parser.add_argument(
        '--log-dir', default='/tmp/lmdeploy_sdar_test', help='日志目录'
    )
    parser.add_argument(
        '--only-start', action='store_true', help='仅启动服务，不运行测试'
    )
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.log_dir, f'sdar_server_{ts}.log')

    print(f'正在启动 SDAR-30B-A3B 服务（port={SERVER_PORT}）...')
    pid = _start_server(args.model_path, log_file)
    if pid == 0:
        print('服务启动失败，请检查日志：', log_file)
        raise SystemExit(1)

    print(f'服务已就绪，PID={pid}，日志：{log_file}')
    print(f'API 地址：{BASE_URL}/v1')

    if args.only_start:
        print('--only-start 模式，服务保持运行中。按 Ctrl+C 停止。')
        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            pass
    else:
        # 冒烟测试
        client = OpenAI(api_key='EMPTY', base_url=f'{BASE_URL}/v1')
        model_id = client.models.list().data[0].id
        print(f'\n冒烟测试，模型：{model_id}')
        for case in BASIC_TEST_CASES:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{'role': 'user', 'content': case['prompt']}],
                max_tokens=256,
                temperature=1.0,
                top_p=1.0,
                extra_body={'top_k': -1},
            )
            content = resp.choices[0].message.content
            print(f'  Q: {case["prompt"]}')
            print(f'  A: {content[:120]}...\n' if len(content) > 120 else f'  A: {content}\n')

    _stop_server(pid)
    print('服务已停止。')
