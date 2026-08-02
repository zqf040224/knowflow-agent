#!/usr/bin/env python3
"""系统接口烟测。

启动当前目录下的 app.py，登录后检查知识库搜索接口。
"""

import json
import os
import subprocess
import time
from pathlib import Path

import requests


APP_DIR = Path(__file__).resolve().parent
PORT = int(os.getenv("TEST_PORT", "5003"))
BASE_URL = f"http://localhost:{PORT}"


def wait_for_server(timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            requests.get(f"{BASE_URL}/login", timeout=2)
            return
        except requests.RequestException:
            time.sleep(1)
    raise TimeoutError(f"服务未在 {timeout} 秒内启动: {BASE_URL}")


def main():
    print("===== 系统功能测试 =====")
    app_process = None
    admin_password = os.getenv("TEST_ADMIN_PASSWORD")
    if not admin_password:
        raise RuntimeError("运行系统烟测前必须设置 TEST_ADMIN_PASSWORD")
    child_env = os.environ.copy()
    child_env["ADMIN_DEFAULT_PASSWORD"] = admin_password

    try:
        print("\n1. 启动应用程序...")
        app_process = subprocess.Popen(
            ["python3", "app.py", "--port", str(PORT)],
            cwd=str(APP_DIR),
            env=child_env,
        )
        wait_for_server()
        print(f"   ✓ 应用程序已启动: {BASE_URL}")

        session = requests.Session()

        print("\n2. 测试登录功能...")
        response = session.post(
            f"{BASE_URL}/api/auth/login",
            headers={"Content-Type": "application/json"},
            data=json.dumps({
                "username": os.getenv("TEST_ADMIN_USERNAME", "admin"),
                "password": admin_password,
            }),
            timeout=15,
        )
        assert response.status_code == 200, f"登录失败，状态码: {response.status_code}"
        login_result = response.json()
        assert login_result.get("success"), f"登录失败: {login_result.get('message')}"
        token = login_result["token"]
        session.headers.update({"Authorization": f"Bearer {token}"})
        print("   ✓ 登录成功")

        print("\n3. 测试知识库搜索...")
        response = session.post(
            f"{BASE_URL}/api/search",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"query": "公文格式"}),
            timeout=30,
        )
        assert response.status_code == 200, f"搜索失败，状态码: {response.status_code}"
        results = response.json()
        assert isinstance(results, list), "搜索结果不是列表"
        print(f"   ✓ 搜索成功，返回 {len(results)} 条结果")
        if results:
            print(f"   第一条结果: {results[0].get('source', 'N/A')}")

    finally:
        if app_process:
            print("\n4. 停止应用程序...")
            app_process.terminate()
            try:
                app_process.wait(timeout=5)
                print("   ✓ 应用程序已停止")
            except subprocess.TimeoutExpired:
                app_process.kill()
                print("   ✓ 应用程序已强制停止")

    print("\n===== 测试完成 =====")


if __name__ == "__main__":
    main()
