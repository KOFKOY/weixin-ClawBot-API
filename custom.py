import asyncio
import base64
import builtins
import io
import json
import os
import random
import re
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import time
import urllib.request
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAIAPI, OpenAIConfig

executor = ThreadPoolExecutor(max_workers=4)
ai = None  # 启动时从配置文件加载后初始化


def _silent_print(*args, **kwargs):
    return


print = _silent_print

# ========== 自动重连配置（可调参数） ==========
# 测试时将数值改小，例如：
#   "session_duration": 300, "warning_before": 60, "reminder_interval": 30,
#   "force_before": 60, "qrcode_scan_timeout": 120
RECONNECT_CONFIG = {
    "session_duration":    24 * 3600,  # 会话总时长（秒）
    "warning_before":       2 * 3600,  # 提前多久开始自动下发重连地址（秒）
    "reminder_interval":      30 * 60, # 重连失败后的重试间隔（秒）
    "force_before":           30 * 60, # 最后多久强制重连（秒）
    "qrcode_scan_timeout":       600,  # 等待用户扫码最长时间（秒）
}
# =============================================

# ========== 配置文件 ==========
CONFIG_FILE = "config.json"
_DEFAULT_PROMPT = ""
CHANNEL_VERSION = "2.4.3"
ILINK_APP_ID = "bot"
ILINK_APP_CLIENT_VERSION = str((2 << 16) | (4 << 8) | 3)
BOT_AGENT = "weixin-ClawBot-API/1.0.1 (python)"
WEBHOOK_NOTIFY_URL = "https://wx.kofkoy.de5.net/webhook"

PROVIDERS = {
    "dusapi": {
        "label": "DusAPI",
        "base_url": "https://api.dusapi.com",
        "model": "gpt-5",
        "prompt": _DEFAULT_PROMPT,
    },
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
        "prompt": _DEFAULT_PROMPT,
    },
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-5.4",
        "prompt": _DEFAULT_PROMPT,
    },
}


def mask_key(key: str) -> str:
    """保留前5位和后5位，中间用星号替换。"""
    if len(key) <= 10:
        return key
    return key[:5] + "*" * (len(key) - 10) + key[-5:]


def load_config_file() -> dict:
    """加载本地配置文件，并兼容旧版扁平配置结构。"""
    if not os.path.exists(CONFIG_FILE):
        return {"provider": "dusapi", "providers": {}}

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 兼容旧版扁平配置：{api_key, base_url, model, prompt}
    if "providers" not in cfg:
        old_provider_cfg = {
            "api_key": cfg.get("api_key", ""),
            "base_url": cfg.get("base_url", PROVIDERS["dusapi"]["base_url"]),
            "model": cfg.get("model", PROVIDERS["dusapi"]["model"]),
            "prompt": cfg.get("prompt", _DEFAULT_PROMPT),
        }
        cfg = {
            "provider": "dusapi",
            "providers": {"dusapi": old_provider_cfg},
        }
    cfg.setdefault("provider", "dusapi")
    cfg.setdefault("providers", {})
    return cfg


def save_config_file(cfg: dict):
    """将配置写入本地 config.json 文件。"""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def choose_provider(default_provider: str) -> str:
    """在命令行交互选择 AI 提供商并返回 provider key。"""
    print("\n请选择 AI 提供商：")
    keys = list(PROVIDERS.keys())
    for index, key in enumerate(keys, 1):
        default_mark = "（默认）" if key == default_provider else ""
        print(f"  {index}. {PROVIDERS[key]['label']} {default_mark}")

    while True:
        choice = input("输入序号或名称后回车: ").strip().lower()
        if not choice:
            return default_provider if default_provider in PROVIDERS else "dusapi"
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(keys):
                return keys[idx]
        if choice in PROVIDERS:
            return choice
        print("输入无效，请重新选择。")


def prompt_provider_config(provider: str, old_cfg: dict | None = None) -> dict:
    """交互式读取指定提供商配置，支持沿用历史值。"""
    defaults = PROVIDERS[provider]
    old_cfg = old_cfg or {}
    print(f"\n配置 {defaults['label']}：")

    old_key = old_cfg.get("api_key", "")
    key_prompt = f"请输入 API Key（当前 {mask_key(old_key)}，留空沿用）: " if old_key else "请输入 API Key: "
    api_key = input(key_prompt).strip() or old_key

    old_base_url = old_cfg.get("base_url", defaults["base_url"])
    base_url = input(f"请输入 API 地址（留空默认/沿用 {old_base_url}）: ").strip() or old_base_url

    old_model = old_cfg.get("model", defaults["model"])
    model = input(f"请输入模型名称（留空默认/沿用 {old_model}）: ").strip() or old_model

    old_prompt = old_cfg.get("prompt", defaults["prompt"])
    prompt = input("请输入系统提示词（留空默认/沿用当前值）: ").strip() or old_prompt

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "prompt": prompt,
    }


def load_or_create_config() -> dict:
    """先选择 AI 提供商，再确认或创建对应配置。"""
    sep = "=" * 60
    dash = "-" * 60
    cfg = load_config_file()

    while True:
        # provider = choose_provider(cfg.get("provider", "dusapi"))
        provider = 'openai'
        cfg["provider"] = provider
        provider_cfg = cfg["providers"].get(provider)
        label = PROVIDERS[provider]["label"]

        if not provider_cfg:
            print(f"\n未找到 {label} 配置，需要创建。")
            provider_cfg = prompt_provider_config(provider)
            cfg["providers"][provider] = provider_cfg
            save_config_file(cfg)
            print(f"\n配置已保存到 {CONFIG_FILE}\n")
            return {"provider": provider, **provider_cfg}

        print(f"\n{sep}")
        print(f"  当前选择：{label}")
        print("  当前配置如下：")
        print(sep)
        print(f"  API Key  : {mask_key(provider_cfg.get('api_key', ''))}")
        print(f"  API 地址 : {provider_cfg.get('base_url', '')}")
        print(f"  模型     : {provider_cfg.get('model', '')}")
        prompt_preview = provider_cfg.get("prompt", "")[:50]
        print(f"  提示词   : {prompt_preview}{'...' if len(provider_cfg.get('prompt','')) > 50 else ''}")
        print(dash)

        # choice = input("\n使用此配置继续？(直接回车或输入 Y 继续 / 输入 N 重新配置 / 输入 S 切换提供商): ").strip().upper()
        choice = 'Y'
        if choice == "N":
            provider_cfg = prompt_provider_config(provider, provider_cfg)
            cfg["providers"][provider] = provider_cfg
            save_config_file(cfg)
            print(f"\n配置已保存到 {CONFIG_FILE}\n")
            return {"provider": provider, **provider_cfg}
        if choice == "S":
            continue
        else:
            save_config_file(cfg)
            return {"provider": provider, **provider_cfg}
# ==============================

BASE_URL = "https://ilinkai.weixin.qq.com"
COMMANDS_MSG = (
    "连接成功！\n"
    "可用指令：\n"
    "/help  /指令   - 查看全部指令列表\n"
    "/time          - 查询当前连接剩余时间\n"
    "/重新连接       - 立即触发重新连接（直接下发重连地址）\n"
    "\n非指令输入即为 AI 对话"
)


def make_headers(token=None):
    """构造调用 iLink 接口所需请求头。"""
    uin = str(random.randint(0, 0xFFFFFFFF))
    headers = {
        "Content-Type": "application/json",
        "AuthorizationType": "ilink_bot_token",
        "X-WECHAT-UIN": base64.b64encode(uin.encode()).decode(),
        "iLink-App-Id": ILINK_APP_ID,
        "iLink-App-ClientVersion": ILINK_APP_CLIENT_VERSION,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def base_info():
    """返回公共基础信息字段。"""
    return {
        "channel_version": CHANNEL_VERSION,
        "bot_agent": BOT_AGENT,
    }


class SessionState(str, Enum):
    """连接状态机状态。"""
    ACTIVE = "ACTIVE"
    RECONNECTING = "RECONNECTING"


@dataclass
class RuntimeState:
    """统一管理运行期可变状态，避免大量 list/dict 引用透传。"""
    bot_token: str
    bot_base_url: str = ""
    login_time: float = field(default_factory=time.time)
    state: SessionState = SessionState.ACTIVE
    last_contact_from_id: str | None = None
    last_contact_context_token: str | None = None
    typing_ticket_cache: dict = field(default_factory=dict)
    context_token_cache: dict = field(default_factory=dict)
    welcomed_users: set = field(default_factory=set)

    def remaining_seconds(self, cfg: dict) -> float:
        return max(0.0, self.login_time + cfg["session_duration"] - time.time())

    def current_base_url(self) -> str:
        return self.bot_base_url or BASE_URL


async def api_get(session, path, token=None, base_url=None):
    """统一封装 GET 请求，打印响应并尽量解析为 JSON。"""
    url = f"{base_url or BASE_URL}/{path}"
    async with session.get(url, headers=make_headers(token)) as res:
        text = await res.text()
        print(f"  [GET {path}] HTTP {res.status} → {text[:200]}")
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                data.setdefault("_http_status", res.status)
            return data
        except Exception:
            return {"_http_status": res.status, "_raw_text": text}


async def api_post(session, path, body, token=None, base_url=None):
    """统一封装 POST 请求，打印响应并尽量解析为 JSON。"""
    url = f"{base_url or BASE_URL}/{path}"
    async with session.post(url, json=body, headers=make_headers(token)) as res:
        text = await res.text()
        print(f"  [{path}] HTTP {res.status} → {text[:200]}")
        try:
            import json
            data = json.loads(text)
            if isinstance(data, dict):
                data.setdefault("_http_status", res.status)
            return data
        except Exception:
            return {"_http_status": res.status, "_raw_text": text}


def _is_send_result_ok(result: dict) -> bool:
    """兼容多种返回结构，尽量判断 sendmessage 是否成功。"""
    if not isinstance(result, dict):
        return False
    status = result.get("_http_status")
    if status is not None and int(status) >= 400:
        return False
    for key in ("errcode", "code", "retcode", "ret"):
        value = result.get(key)
        if value not in (None, 0, "0", 200, "200", "ok", "OK"):
            return False
    return True


async def send_msg_safe(session, to_id, context_token, text, bot_token, bot_base_url) -> bool:
    """发送微信消息，失败时返回 False，不抛异常。"""
    if not to_id or not context_token:
        print(f"[发送失败] 缺少 to_id/context_token，消息内容: {text}")
        return False
    try:
        client_id = f"openclaw-weixin-{random.randint(0, 0xFFFFFFFF):08x}"
        result = await api_post(
            session,
            "ilink/bot/sendmessage",
            {
                "msg": {
                    "from_user_id": "",
                    "to_user_id": to_id,
                    "client_id": client_id,
                    "message_type": 2,
                    "message_state": 2,
                    "context_token": context_token,
                    "item_list": [{"type": 1, "text_item": {"text": text}}],
                },
                "base_info": base_info(),
            },
            bot_token,
            bot_base_url or None,
        )
        return _is_send_result_ok(result)
    except Exception as e:
        print(f"[发送失败] sendmessage 异常({e})，消息内容: {text}")
        return False


async def notify_webhook(session, final_msg: str) -> bool:
    """发送兜底 webhook 通知。"""
    url = f"{WEBHOOK_NOTIFY_URL}?msg={quote(final_msg, safe='')}"
    try:
        async with session.get(url) as res:
            await res.text()
            return 200 <= res.status < 300
    except Exception as e:
        print(f"[Webhook] 通知失败: {e}")
        return False


def extract_qrcode_address(data: dict) -> str:
    """优先提取可直接访问的二维码地址。"""
    qrcode = str(data.get("qrcode", ""))
    qrcode_img_content = str(data.get("qrcode_img_content", ""))
    for candidate in (qrcode, qrcode_img_content):
        if candidate.startswith("http"):
            return candidate
    return qrcode or qrcode_img_content


class ReconnectStateMachine:
    """连接重连状态机：ACTIVE -> RECONNECTING -> ACTIVE。"""

    def __init__(self, runtime: RuntimeState, cfg: dict):
        self.runtime = runtime
        self.cfg = cfg
        self._lock = asyncio.Lock()
        self.next_retry_at = 0.0

    def remaining_seconds(self) -> float:
        return self.runtime.remaining_seconds(self.cfg)

    def should_auto_reconnect(self) -> bool:
        if self.runtime.state != SessionState.ACTIVE:
            return False
        if time.time() < self.next_retry_at:
            return False
        return self.remaining_seconds() <= self.cfg["warning_before"]

    def _schedule_retry(self, delay_seconds: float):
        self.next_retry_at = time.time() + max(5.0, float(delay_seconds))

    async def trigger(self, session, reason: str) -> bool:
        if self._lock.locked():
            return False
        async with self._lock:
            self.runtime.state = SessionState.RECONNECTING
            success = await self._run_reconnect_flow(session, reason)
            self.runtime.state = SessionState.ACTIVE
            if success:
                self.next_retry_at = 0.0
            return success

    async def _run_reconnect_flow(self, session, reason: str) -> bool:
        current_base = self.runtime.current_base_url()
        local_token_list = [self.runtime.bot_token] if self.runtime.bot_token else []

        try:
            data = await fetch_login_qrcode(session, current_base, local_token_list)
        except Exception as e:
            print(f"[重连] 获取二维码失败: {e}")
            self._schedule_retry(self.cfg["reminder_interval"])
            return False

        qrcode = data.get("qrcode")
        if not qrcode:
            print("[重连] 未获取到 qrcode 字段")
            self._schedule_retry(self.cfg["reminder_interval"])
            return False

        address = extract_qrcode_address(data)
        remaining_m = self.remaining_seconds() / 60
        final_msg = (
            f"[重连-{reason}] 连接剩余约 {remaining_m:.0f} 分钟，"
            f"请直接打开地址完成重连：{address}"
        )

        sent = await send_msg_safe(
            session,
            self.runtime.last_contact_from_id,
            self.runtime.last_contact_context_token,
            final_msg,
            self.runtime.bot_token,
            self.runtime.bot_base_url,
        )
        if not sent:
            await notify_webhook(session, final_msg)

        login_result = await wait_login_confirmation(
            session,
            qrcode,
            current_base,
            timeout_seconds=self.cfg["qrcode_scan_timeout"],
            allow_already_connected=True,
        )

        if login_result.get("already_connected"):
            self.runtime.login_time = time.time()
            await send_msg_safe(
                session,
                self.runtime.last_contact_from_id,
                self.runtime.last_contact_context_token,
                "[重连] 服务端提示连接已有效，继续使用当前连接。",
                self.runtime.bot_token,
                self.runtime.bot_base_url,
            )
            return True

        new_token = login_result.get("bot_token")
        if not new_token:
            fail_msg = "[重连失败] 扫码超时或未确认，稍后会再次发送重连地址。"
            sent_fail = await send_msg_safe(
                session,
                self.runtime.last_contact_from_id,
                self.runtime.last_contact_context_token,
                fail_msg,
                self.runtime.bot_token,
                self.runtime.bot_base_url,
            )
            if not sent_fail:
                await notify_webhook(session, fail_msg)
            self._schedule_retry(self.cfg["reminder_interval"])
            return False

        self.runtime.bot_token = new_token
        self.runtime.bot_base_url = login_result.get("baseurl", self.runtime.bot_base_url)
        self.runtime.login_time = time.time()
        self.runtime.typing_ticket_cache.clear()
        await send_msg_safe(
            session,
            self.runtime.last_contact_from_id,
            self.runtime.last_contact_context_token,
            "[重连完成] 新连接已建立，已自动切换。",
            self.runtime.bot_token,
            self.runtime.bot_base_url,
        )
        return True


async def reconnect_timer_task(session, machine: ReconnectStateMachine):
    """独立定时器：接近到期时自动触发重连地址下发。"""
    while True:
        if machine.should_auto_reconnect():
            reason = "自动强制" if machine.remaining_seconds() <= machine.cfg["force_before"] else "自动提醒"
            await machine.trigger(session, reason)
        await asyncio.sleep(5)


def render_terminal_qr(content: str):
    """在终端渲染二维码，优先使用远程图片内容。"""
    if not content:
        return
    builtins.print("\n扫码地址:", content)
    if content.startswith("http") and render_terminal_image_from_url(content):
        return
    render_generated_qr(content)


def render_terminal_image_from_url(url: str) -> bool:
    """下载二维码图片并转为字符画输出到终端。"""
    try:
        from PIL import Image
    except ImportError:
        return False

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        image = Image.open(io.BytesIO(data)).convert("L")
        max_width = 72
        scale = max(1, int(image.width / max_width))
        width = max(1, int(image.width / scale))
        height = max(1, int(image.height / scale))
        image = image.resize((width, height))
        builtins.print()
        for y in range(height):
            builtins.print("".join("██" if image.getpixel((x, y)) < 128 else "  " for x in range(width)))
        builtins.print()
        return True
    except Exception as e:
        builtins.print(f"二维码图片渲染失败，改用本地二维码生成方式: {e}")
        return False


def render_generated_qr(content: str):
    """使用本地 qrcode 库直接生成二维码字符画。"""
    try:
        import qrcode
    except ImportError:
        builtins.print("未安装 qrcode/Pillow，无法在终端渲染二维码；安装 `pip install qrcode pillow` 后会自动显示。")
        return

    qr = qrcode.QRCode(version=1,border=0)
    qr.add_data(content)
    qr.make(fit=True)
    matrix = qr.get_matrix()
    builtins.print()
    for row in matrix:
        builtins.print("".join("██" if cell else "  " for cell in row))
    builtins.print()


def save_qrcode_content(content: str):
    """根据二维码内容类型保存文件或直接在终端渲染。"""
    if not content:
        return
    if content.startswith("data:image/"):
        header, b64 = content.split(",", 1)
        m = re.search(r"data:image/(\w+)", header)
        ext = m.group(1) if m else "png"
        with open(f"qrcode.{ext}", "wb") as f:
            f.write(base64.b64decode(b64))
        builtins.print(f"二维码已保存到 qrcode.{ext}")
    elif content.startswith("<svg"):
        with open("qrcode.svg", "w", encoding="utf-8") as f:
            f.write(content)
        builtins.print("二维码已保存到 qrcode.svg，用浏览器打开")
    elif content.startswith("http"):
        render_terminal_qr(content)
    else:
        try:
            with open("qrcode.png", "wb") as f:
                f.write(base64.b64decode(content))
            builtins.print("二维码已保存到 qrcode.png")
        except Exception:
            render_terminal_qr(content)


async def fetch_login_qrcode(session, base_url=BASE_URL, local_token_list=None):
    """获取登录二维码，POST 失败时兼容回退到 GET 流程。"""
    body = {"local_token_list": local_token_list or []}
    data = await api_post(session, "ilink/bot/get_bot_qrcode?bot_type=3", body, None, base_url)
    if data.get("qrcode"):
        return data
    print("POST 获取二维码未返回 qrcode，尝试兼容旧版 GET 流程。")
    return await api_get(session, "ilink/bot/get_bot_qrcode?bot_type=3", None, base_url)


async def poll_login_status(session, qrcode, base_url=BASE_URL, verify_code=None):
    """轮询二维码登录状态并归一化返回关键信息。"""
    endpoint = f"ilink/bot/get_qrcode_status?qrcode={quote(qrcode, safe='')}"
    if verify_code:
        endpoint += f"&verify_code={quote(verify_code, safe='')}"
    status = await api_get(session, endpoint, None, base_url)
    state = status.get("status", "")

    if state == "confirmed" or status.get("bot_token"):
        return {
            "bot_token": status.get("bot_token"),
            "baseurl": status.get("baseurl") or status.get("base_url") or base_url,
            "ilink_bot_id": status.get("ilink_bot_id"),
            "ilink_user_id": status.get("ilink_user_id"),
        }
    if state == "binded_redirect" or status.get("binded_redirect"):
        return {"already_connected": True}
    if state == "expired":
        return {"expired": True}
    if state == "scaned_but_redirect":
        redirect_host = status.get("redirect_host")
        if redirect_host:
            return {"redirect_base": f"https://{redirect_host}"}
        print("服务端要求切换扫码轮询节点，但未返回 redirect_host，继续使用当前节点。")
        return {}
    if state == "scaned":
        return {"scanned": True, "verify_code_accepted": bool(verify_code)}
    elif state in ("need_verifycode", "verify_code_blocked") or status.get("need_verifycode"):
        if state == "verify_code_blocked":
            return {"verify_code_blocked": True}
        return {"need_verifycode": True, "retry_verifycode": bool(verify_code)}
    elif state and state != "wait":
        print(f"登录状态: {state}，原始响应: {status}")

    if status.get("local_token_list"):
        print("服务端返回 local_token_list 信息，继续等待扫码确认。")
    return {}


async def wait_login_confirmation(session, qrcode, base_url=BASE_URL, timeout_seconds=None,
                                  allow_already_connected=False):
    """持续等待扫码确认，处理超时、跳转节点与配对码场景。"""
    deadline = time.time() + timeout_seconds if timeout_seconds else None
    current_base_url = base_url
    pending_verify_code = None
    scanned_printed = False

    while True:
        if deadline and time.time() >= deadline:
            return {"timeout": True}

        try:
            result = await poll_login_status(session, qrcode, current_base_url, pending_verify_code)
        except Exception as e:
            print(f"轮询扫码状态失败，稍后重试: {e}")
            await asyncio.sleep(1)
            continue

        if result.get("bot_token"):
            return result
        if result.get("already_connected"):
            return result if allow_already_connected else {"already_connected": True}
        if result.get("expired"):
            return result
        if result.get("verify_code_blocked"):
            return result
        if result.get("redirect_base"):
            current_base_url = result["redirect_base"]
            print(f"扫码轮询切换到新节点: {current_base_url}")
            continue
        if result.get("scanned"):
            if pending_verify_code and result.get("verify_code_accepted"):
                pending_verify_code = None
            if not scanned_printed:
                print("已扫码，等待手机端确认...")
                scanned_printed = True
        if result.get("need_verifycode"):
            prompt = "你输入的数字不匹配，请重新输入: " if result.get("retry_verifycode") else "请输入手机微信显示的数字配对码: "
            pending_verify_code = input(prompt).strip()
            continue

        await asyncio.sleep(1)


async def login_with_qrcode(session, base_url=BASE_URL, show_qrcode=True):
    """执行扫码登录主流程，必要时刷新二维码重试。"""
    refresh_count = 0
    max_refresh_count = 3
    while True:
        data = await fetch_login_qrcode(session, base_url)
        qrcode = data["qrcode"]
        qrcode_img_content = data.get("qrcode_img_content", "")

        if show_qrcode:
            builtins.print("qrcode:", qrcode)
            save_qrcode_content(str(qrcode_img_content or qrcode))
        builtins.print("等待扫码...")

        login_result = await wait_login_confirmation(session, qrcode, base_url)
        if login_result.get("bot_token"):
            return login_result
        if login_result.get("already_connected"):
            builtins.print("服务端提示此端已连接过，但当前独立程序没有可复用 token，将重新生成二维码。")
        elif login_result.get("expired"):
            builtins.print("二维码已过期，正在重新生成...")
        elif login_result.get("verify_code_blocked"):
            builtins.print("多次输入配对码错误，正在刷新二维码...")
        elif login_result.get("timeout"):
            builtins.print("登录等待超时，正在重新生成二维码...")

        refresh_count += 1
        if refresh_count >= max_refresh_count:
            raise RuntimeError("二维码多次失效或登录失败，请稍后重试。")


async def main():
    """程序主入口：登录、启动定时重连并循环收发消息。"""
    async with aiohttp.ClientSession() as session:
        login_result = await login_with_qrcode(session, show_qrcode=True)
        runtime = RuntimeState(
            bot_token=login_result["bot_token"],
            bot_base_url=login_result.get("baseurl", ""),
            login_time=time.time(),
        )
        machine = ReconnectStateMachine(runtime, RECONNECT_CONFIG)
        asyncio.create_task(reconnect_timer_task(session, machine))

        get_updates_buf = ""
        print("开始监听消息...")
        while True:
            result = await api_post(
                session,
                "ilink/bot/getupdates",
                {"get_updates_buf": get_updates_buf, "base_info": base_info()},
                runtime.bot_token,
                runtime.bot_base_url or None,
            )
            get_updates_buf = result.get("get_updates_buf") or get_updates_buf

            # OpenAI 智能体定时任务通知：从队列取出并尝试发送微信消息。
            if hasattr(ai, "drain_notifications"):
                for notice in ai.drain_notifications(limit=20):
                    target_user_id = notice.get("user_id") or runtime.last_contact_from_id
                    target_context_token = runtime.context_token_cache.get(target_user_id) or runtime.last_contact_context_token
                    notice_text = notice.get("message", "")
                    if not target_user_id or not target_context_token:
                        print(f"[定时通知] 无可用会话，跳过发送: {notice_text}")
                        continue
                    await send_msg_safe(
                        session,
                        target_user_id,
                        target_context_token,
                        f"[定时提醒] {notice_text}",
                        runtime.bot_token,
                        runtime.bot_base_url,
                    )

            for msg in result.get("msgs") or []:
                if msg.get("message_type") != 1:
                    continue
                text = msg.get("item_list", [{}])[0].get("text_item", {}).get("text", "")
                from_id = msg["from_user_id"]
                context_token = msg["context_token"]
                print(f"收到消息: {text}")
                text_clean = text.strip()

                runtime.last_contact_from_id = from_id
                runtime.last_contact_context_token = context_token
                runtime.context_token_cache[from_id] = context_token

                if from_id not in runtime.welcomed_users:
                    runtime.welcomed_users.add(from_id)
                    # await send_msg_safe(session, from_id, context_token,
                    #                     COMMANDS_MSG, runtime.bot_token, runtime.bot_base_url)
                    # continue

                # /help  /指令 — 返回指令列表
                if text_clean in ("/help", "/指令"):
                    await send_msg_safe(session, from_id, context_token,
                                        COMMANDS_MSG, runtime.bot_token, runtime.bot_base_url)
                    continue

                # /time 指令
                if text_clean == "/time":
                    _rem = runtime.remaining_seconds(RECONNECT_CONFIG)
                    _h, _m, _s = int(_rem // 3600), int((_rem % 3600) // 60), int(_rem % 60)
                    _ts = f"{_h} 小时 {_m} 分钟" if _h > 0 else f"{_m} 分钟 {_s} 秒"
                    _state = "重连中" if runtime.state == SessionState.RECONNECTING else "运行中"
                    await send_msg_safe(session, from_id, context_token,
                                        f"当前连接状态：{_state}，剩余时间：{_ts}",
                                        runtime.bot_token, runtime.bot_base_url)
                    continue

                if text_clean == "/重新连接":
                    if runtime.state == SessionState.RECONNECTING:
                        await send_msg_safe(session, from_id, context_token,
                                            "重连正在进行中，请稍候...",
                                            runtime.bot_token, runtime.bot_base_url)
                    else:
                        await send_msg_safe(session, from_id, context_token,
                                            "好的，正在下发重连地址...",
                                            runtime.bot_token, runtime.bot_base_url)
                        await machine.trigger(session, "手动")
                    continue

                # getconfig 获取 typing_ticket（每个用户缓存一次）
                if from_id not in runtime.typing_ticket_cache:
                    cfg = await api_post(
                        session,
                        "ilink/bot/getconfig",
                        {"ilink_user_id": from_id, "context_token": context_token,
                         "base_info": base_info()},
                        runtime.bot_token,
                        runtime.bot_base_url or None,
                    )
                    runtime.typing_ticket_cache[from_id] = cfg.get("typing_ticket", "")
                typing_ticket = runtime.typing_ticket_cache[from_id]

                # sendtyping status=1 表示"正在输入"
                if typing_ticket:
                    await api_post(
                        session,
                        "ilink/bot/sendtyping",
                        {"ilink_user_id": from_id, "typing_ticket": typing_ticket,
                         "status": 1, "base_info": base_info()},
                        runtime.bot_token,
                        runtime.bot_base_url or None,
                    )

                # 调用 AI
                loop = asyncio.get_event_loop()
                # 或者替换为你自已要用的接口
                reply = await loop.run_in_executor(executor, ai.chat, text)

                # sendmessage（补全 SDK 所需字段）
                client_id = f"openclaw-weixin-{random.randint(0, 0xFFFFFFFF):08x}"
                send_result = await api_post(
                    session,
                    "ilink/bot/sendmessage",
                    {
                        "msg": {
                            "from_user_id": "",
                            "to_user_id": from_id,
                            "client_id": client_id,
                            "message_type": 2,
                            "message_state": 2,
                            "context_token": context_token,
                            "item_list": [{"type": 1, "text_item": {"text": reply}}],
                        },
                        "base_info": base_info(),
                    },
                    runtime.bot_token,
                    runtime.bot_base_url or None,
                )
                print(f"sendmessage 返回: {send_result}")
                print(f"已回复: {reply[:50]}...")

                # sendtyping status=2 取消"正在输入"
                if typing_ticket:
                    await api_post(
                        session,
                        "ilink/bot/sendtyping",
                        {"ilink_user_id": from_id, "typing_ticket": typing_ticket,
                         "status": 2, "base_info": base_info()},
                        runtime.bot_token,
                        runtime.bot_base_url or None,
                    )


print(
    "\n"
    "╔══════════════════════════════════════════════════════════╗\n"
    "║          微信 ClawBot  ·  WeChat iLink Bot               ║\n"
    "║  Copyright (c) 2026 SiverKing. All rights reserved.     ║\n"
    "║  GitHub : https://github.com/SiverKing/weixin-ClawBot-API║\n"
    "╚══════════════════════════════════════════════════════════╝"
)

_raw_cfg = load_or_create_config()
ai = OpenAIAPI(OpenAIConfig(
        api_key=_raw_cfg["api_key"],
        base_url=_raw_cfg["base_url"],
        model=_raw_cfg["model"],
        prompt=_raw_cfg["prompt"],
        type=_raw_cfg.get("type", "chat"),
    ))

asyncio.run(main())
