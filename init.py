"""
================================================================================
智能客服Agent前置数据初始化模块
================================================================================

【功能说明】
本模块负责智能客服系统的初始化工作，包括：
1. 环境配置加载：从 .env 文件加载环境变量
2. FAQ数据同步：将 faq_data.json 同步到 MySQL 数据库
3. 意图树更新：根据 FAQ 数据自动更新 config/intents.yaml
4. 知识库初始化：连接 Milvus/Elasticsearch 并导入向量数据
5. 健康检查：验证系统组件是否正常工作

【使用方式】
- 完整初始化：python init.py --mode all
- 仅同步MySQL：python init.py --mode mysql
- 仅导入向量：python init.py --mode vector
- 仅检查状态：python init.py --mode check

【环境变量】
- MYSQL_HOST/MYSQL_PORT/MYSQL_USER/MYSQL_PASSWORD/MYSQL_DATABASE
- MILVUS_URI: Milvus 连接地址
- ES_URL: Elasticsearch 连接地址
- INIT_BATCH_SIZE: 批处理大小
- INIT_FORCE_VECTOR_REIMPORT: 强制重导入向量数据
- INIT_SKIP_VECTOR_IMPORT: 跳过向量导入

================================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

import pymysql
import yaml
from dotenv import load_dotenv
from langchain_core.documents import Document

from data.knowledge_base import KnowledgeBase

# ================================================================================
# 第一节：路径与常量定义
# ================================================================================
# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent
# FAQ数据文件路径
FAQ_DATA_PATH = ROOT_DIR / "faq_data.json"
# 意图树配置文件路径
INTENTS_PATH = ROOT_DIR / "config" / "intents.yaml"
# 日志目录
LOG_DIR = ROOT_DIR / "logs"

T = TypeVar("T")
logger = logging.getLogger("init")


# ================================================================================
# 第二节：配置数据类
# ================================================================================
@dataclass(frozen=True)
class InitSettings:
    """
    初始化配置数据类

    【设计说明】
    使用 frozen dataclass 确保配置不可变，所有配置从环境变量读取。
    提供便捷的属性访问方式，支持链式调用（如 mysql_server_config()）

    【字段说明】
    - mysql_*: MySQL数据库连接配置
    - milvus_*: Milvus向量数据库配置
    - elasticsearch_*: Elasticsearch配置
    - batch_size: 批处理大小
    - retry_*: 重试配置
    - force_vector_reimport: 是否强制重导入向量数据
    - skip_vector_import: 是否跳过向量导入
    - sample_queries: 样例检索查询列表
    """
    mysql_host: str
    mysql_port: int
    mysql_user: str
    mysql_password: str
    mysql_database: str
    milvus_uri: str
    milvus_collection: str
    elasticsearch_url: str
    elasticsearch_index: str
    batch_size: int
    retry_attempts: int
    retry_interval_seconds: float
    force_vector_reimport: bool
    skip_vector_import: bool
    sample_queries: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "InitSettings":
        """
        从环境变量构建配置对象

        【实现逻辑】
        1. 读取所有环境变量，提供默认值
        2. 对数值型配置进行类型转换
        3. 处理布尔型和字符串元组配置
        """
        return cls(
            mysql_host=os.getenv("MYSQL_HOST", "127.0.0.1"),
            mysql_port=int(os.getenv("MYSQL_PORT", "3306")),
            mysql_user=os.getenv("MYSQL_USER", "root"),
            mysql_password=os.getenv("MYSQL_PASSWORD", ""),
            mysql_database=os.getenv("MYSQL_DATABASE", "customer_service_db"),
            milvus_uri=normalize_milvus_uri(os.getenv("MILVUS_URI", "http://localhost:19530")),
            milvus_collection=os.getenv("MILVUS_COLLECTION_NAME", "customer_service"),
            elasticsearch_url=os.getenv("ES_URL", "http://localhost:9200"),
            elasticsearch_index=os.getenv("ES_INDEX_NAME", "customer_service"),
            batch_size=int(os.getenv("INIT_BATCH_SIZE", "100")),
            retry_attempts=int(os.getenv("INIT_RETRY_ATTEMPTS", "3")),
            retry_interval_seconds=float(os.getenv("INIT_RETRY_INTERVAL_SECONDS", "2")),
            force_vector_reimport=env_bool("INIT_FORCE_VECTOR_REIMPORT", False),
            skip_vector_import=env_bool("INIT_SKIP_VECTOR_IMPORT", False),
            sample_queries=tuple(
                query.strip()
                for query in os.getenv("INIT_SAMPLE_QUERIES", "你好,退货,会员积分").split(",")
                if query.strip()
            ),
        )

    def mysql_server_config(self) -> dict[str, Any]:
        """
        获取MySQL服务器连接配置（不含数据库名）

        【用途】
        用于连接MySQL服务器时创建数据库
        """
        return {
            "host": self.mysql_host,
            "port": self.mysql_port,
            "user": self.mysql_user,
            "password": self.mysql_password,
            "charset": "utf8mb4",
            "cursorclass": pymysql.cursors.DictCursor,
            "autocommit": False,
        }

    def mysql_db_config(self) -> dict[str, Any]:
        """
        获取MySQL数据库连接配置（包含数据库名）

        【用途】
        用于连接具体数据库后的所有操作
        """
        config = self.mysql_server_config()
        config["database"] = self.mysql_database
        return config


# ================================================================================
# 第三节：数据传输对象（DTO）
# ================================================================================
@dataclass(frozen=True)
class FAQItem:
    """
    FAQ数据项

    【设计说明】
    使用 frozen dataclass 作为不可变数据结构，
    content_hash 属性用于增量同步判断

    【字段说明】
    - intent_id: 意图ID（如 A1、B2）
    - question: 用户问题
    - answer: 标准答案
    - domain: 领域（可选）
    - action: 动作类型（可选）
    """
    intent_id: str
    question: str
    answer: str
    domain: str = ""
    action: str = ""

    @property
    def content_hash(self) -> str:
        """
        计算内容哈希值

        【实现逻辑】
        对所有字段JSON序列化后计算SHA-256哈希，
        用于判断内容是否发生变化

        【用途】
        增量同步时判断FAQ是否需要更新
        """
        payload = {
            "intent_id": self.intent_id,
            "question": self.question,
            "answer": self.answer,
            "domain": self.domain,
            "action": self.action,
        }
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass
class FAQSyncResult:
    """
    MySQL FAQ同步结果

    【字段说明】
    - mysql_total_before/after: 同步前后MySQL中FAQ数量
    - inserted: 新插入数量
    - updated: 更新数量
    - unchanged: 未变化数量
    - missing_in_source: MySQL中有但源数据没有的数量
    """
    mysql_total_before: int
    mysql_total_after: int
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    missing_in_source: int = 0


@dataclass
class VectorImportResult:
    """
    向量导入结果

    【字段说明】
    - milvus_imported: Milvus导入数量
    - elasticsearch_imported: Elasticsearch导入数量
    - skipped: 是否跳过
    - reason: 跳过原因
    """
    milvus_imported: int = 0
    elasticsearch_imported: int = 0
    skipped: bool = False
    reason: str = ""


# ================================================================================
# 第四节：工具函数
# ================================================================================
def configure_logging() -> None:
    """
    配置日志系统

    【实现细节】
    1. 创建日志目录
    2. 使用RotatingFileHandler实现日志轮转（10MB/文件，保留5个备份）
    3. 同时输出到控制台和文件
    4. 格式：时间 - 模块名 - 级别 - 消息
    """
    LOG_DIR.mkdir(exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handlers: list[logging.Handler] = [
        RotatingFileHandler(LOG_DIR / "init.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"),
        logging.StreamHandler(),
    ]

    root_logger.handlers.clear()
    for handler in handlers:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)


def env_bool(name: str, default: bool) -> bool:
    """
    解析布尔型环境变量

    【实现逻辑】
    支持多种格式：1/true/yes/y/on（不区分大小写）
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def normalize_milvus_uri(raw_uri: str) -> str:
    """
    标准化Milvus URI格式

    【实现逻辑】
    将 tcp:// 前缀转换为 http://
    Milvus 2.x 版本推荐使用 http:// 协议
    """
    milvus_uri = raw_uri.strip()
    if milvus_uri.startswith("tcp://"):
        return f"http://{milvus_uri[len('tcp://'):]}"
    return milvus_uri


def retry(operation_name: str, attempts: int, interval_seconds: float) -> Callable[[Callable[[], T]], T]:
    """
    重试装饰器工厂

    【设计说明】
    通用重试机制，用于处理网络波动等临时性失败

    【参数说明】
    - operation_name: 操作名称（用于日志）
    - attempts: 重试次数
    - interval_seconds: 重试间隔（秒）

    【实现逻辑】
    1. 执行目标函数
    2. 失败时记录异常并等待后重试
    3. 达到最大重试次数后抛出最后一次异常
    """
    def decorator(func: Callable[[], T]) -> T:
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                logger.info("开始执行: %s (尝试 %s/%s)", operation_name, attempt, attempts)
                result = func()
                logger.info("执行成功: %s", operation_name)
                return result
            except Exception as exc:
                last_error = exc
                logger.exception("执行失败: %s (尝试 %s/%s)", operation_name, attempt, attempts)
                if attempt < attempts:
                    time.sleep(interval_seconds)
        assert last_error is not None
        raise last_error

    return decorator


# ================================================================================
# 第五节：环境初始化
# ================================================================================
def setup_environment() -> None:
    """
    初始化环境配置

    【实现逻辑】
    1. 加载 .env 文件
    2. 配置 LangSmith API Key（如果提供）

    【注意】
    LangSmith 用于调试和追踪 LLM 调用，非必需
    """
    load_dotenv(ROOT_DIR / ".env")
    langsmith_key = os.getenv("LANGCHAIN_SMITH_API_KEY")
    if langsmith_key:
        os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
        os.environ["LANGCHAIN_API_KEY"] = langsmith_key
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "智能客服Agent")


# ================================================================================
# 第六节：FAQ 数据加载与校验
# ================================================================================
def load_faq_data(path: Path = FAQ_DATA_PATH) -> list[FAQItem]:
    """
    读取并校验FAQ数据文件

    【功能】
    1. 从JSON文件读取FAQ数据
    2. 校验数据格式和完整性
    3. 检测重复数据
    4. 构建FAQItem列表

    【校验规则】
    - 顶层必须是数组
    - 每条必须包含 intent_id、question、answer
    - 检测并拒绝重复的 (intent_id, question) 组合

    【参数】
    - path: FAQ数据文件路径

    【返回】
    - FAQItem列表

    【异常】
    - ValueError: 数据格式错误或缺少必填字段
    """
    logger.info("读取FAQ数据文件: %s", path)
    with path.open("r", encoding="utf-8") as file:
        raw_items = json.load(file)

    if not isinstance(raw_items, list):
        raise ValueError("faq_data.json 顶层结构必须是列表")

    faq_items: list[FAQItem] = []
    seen_keys: set[tuple[str, str]] = set()
    for index, item in enumerate(raw_items, 1):
        if not isinstance(item, dict):
            raise ValueError(f"FAQ第 {index} 条不是对象")

        intent_id = str(item.get("intent_id", "")).strip()
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not intent_id or not question or not answer:
            raise ValueError(f"FAQ第 {index} 条缺少 intent_id/question/answer")

        key = (intent_id, question)
        if key in seen_keys:
            raise ValueError(f"FAQ存在重复数据: intent_id={intent_id}, question={question}")
        seen_keys.add(key)

        faq_items.append(
            FAQItem(
                intent_id=intent_id,
                question=question,
                answer=answer,
                domain=str(item.get("domain", "") or "").strip(),
                action=str(item.get("action", "") or "").strip(),
            )
        )

    logger.info("FAQ数据校验完成，共 %s 条", len(faq_items))
    return faq_items


# ================================================================================
# 第七节：意图树构建与同步
# ================================================================================
def build_intents_from_faq(faq_items: Iterable[FAQItem], existing_intents: dict[str, Any]) -> dict[str, Any]:
    """
    根据FAQ数据构建意图树

    【功能】
    1. 从FAQ数据提取所有意图ID
    2. 生成意图名称和描述
    3. 保留D1/D2等特殊意图

    【保留策略】
    - D1（寒暄）和 D2（人工介入）始终保留，不受FAQ影响
    - 其他意图根据FAQ数据自动生成或更新

    【参数】
    - faq_items: FAQ数据列表
    - existing_intents: 现有的意图树配置

    【返回】
    - 意图树字典
    """
    ids_from_faq = {item.intent_id for item in faq_items}
    managed_ids = sorted(intent_id for intent_id in ids_from_faq if intent_id not in {"D1", "D2"})

    # 意图目录定义，包含标准名称和描述
    # 用于自动生成意图树条目
    default_catalog = {
        "A1": ("退货/退款", "覆盖破损、尺码拍错、质量问题、描述不符、运费承担、退货地址、退款到账等退货退款问题"),
        "A2": ("换货/维修", "覆盖换尺码换颜色、寄修到店维修、维修点查询、维修费用、备用机、换货物流等售后处理"),
        "A3": ("物流状态", "覆盖物流查询、修改地址、催发货、加急发货、驿站投放、物流停滞、丢件退回等配送问题"),
        "B1": ("规格参数", "覆盖尺寸重量、材质说明、颜色款式、兼容性、保修信息、包装清单等商品参数咨询"),
        "B2": ("价格活动", "覆盖优惠券、满减、保价、限时活动、会员价、拼团秒杀、运费券等价格促销问题"),
        "B3": ("使用教程", "覆盖安装步骤、绑定账号、功能设置、故障排除、清洁保养、升级重置等使用指导"),
        "C1": ("登录注册", "覆盖注册登录、密码修改、账号注销、手机号绑定、验证码、异常登录和账号安全问题"),
        "C2": ("会员权益", "覆盖积分查询、会员等级、权益领取、生日礼包、专属客服、会员活动和积分兑换问题"),
    }

    intents: dict[str, Any] = {}
    # 处理从FAQ中提取的意图ID
    for intent_id in managed_ids:
        existing = existing_intents.get(intent_id, {}) if isinstance(existing_intents.get(intent_id), dict) else {}
        name, description = default_catalog.get(
            intent_id,
            (existing.get("name", intent_id), existing.get("description", f"由 faq_data.json 自动同步的 {intent_id} 意图")),
        )
        intents[intent_id] = {
            "name": existing.get("name", name),
            "description": existing.get("description", description) if intent_id not in default_catalog else description,
            "intent_id": intent_id,
        }

    # 保留D1/D2等特殊意图
    for reserved_id in ("D1", "D2"):
        if reserved_id in existing_intents:
            intents[reserved_id] = existing_intents[reserved_id]
        elif reserved_id in ids_from_faq:
            name = "寒暄" if reserved_id == "D1" else "人工介入"
            description = "包含你好、你是谁、谢谢等问候与结束语" if reserved_id == "D1" else "包含投诉、转人工、电话联系等人工服务请求"
            intents[reserved_id] = {"name": name, "description": description, "intent_id": reserved_id}

    return intents


def sync_intents_yaml(faq_items: list[FAQItem], path: Path = INTENTS_PATH) -> dict[str, Any]:
    """
    同步意图树配置到YAML文件

    【功能】
    1. 读取现有的意图树配置
    2. 根据FAQ数据构建新的意图树
    3. 写入更新后的YAML文件

    【参数】
    - faq_items: FAQ数据列表
    - path: 意图树YAML文件路径

    【返回】
    - 新的意图树数据
    """
    logger.info("同步意图树配置: %s", path)
    existing_data: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            existing_data = yaml.safe_load(file) or {}

    new_data = {"intents": build_intents_from_faq(faq_items, existing_data.get("intents", {}))}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("# 二级分类意图树\n")
        yaml.safe_dump(new_data, file, allow_unicode=True, sort_keys=False)

    logger.info("意图树同步完成，共 %s 个意图", len(new_data["intents"]))
    return new_data


# ================================================================================
# 第八节：MySQL FAQ 仓库
# ================================================================================
class FAQRepository:
    """
    FAQ数据仓库

    【职责】
    1. 管理MySQL连接
    2. 创建/迁移数据库表结构
    3. 实现FAQ数据的CRUD操作
    4. 提供分页查询能力

    【设计模式】
    Repository模式，封装数据访问逻辑

    【注意】
    当前版本未启用事务，建议在高并发场景下添加
    """

    def __init__(self, settings: InitSettings):
        """初始化FAQ仓库"""
        self.settings = settings

    def connect_server(self):
        """
        连接到MySQL服务器（不含数据库）

        【用途】
        用于创建数据库
        """
        return pymysql.connect(**self.settings.mysql_server_config())

    def connect_database(self):
        """
        连接到MySQL数据库

        【用途】
        用于执行数据库操作
        """
        return pymysql.connect(**self.settings.mysql_db_config())

    def ensure_database_and_table(self) -> None:
        """
        确保数据库和FAQ表存在

        【实现逻辑】
        1. 创建数据库（如果不存在）
        2. 创建FAQ表（如果不存在）
        3. 执行字段迁移（兼容旧表结构）
        """
        with self.connect_server() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{self.settings.mysql_database}` "
                    "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
            connection.commit()

        with self.connect_database() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS faq (
                        id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
                        domain VARCHAR(128) NOT NULL DEFAULT '',
                        intent VARCHAR(64) NOT NULL,
                        action VARCHAR(128) NOT NULL DEFAULT '',
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        content_hash CHAR(64) NOT NULL DEFAULT '',
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        INDEX idx_intent (intent),
                        INDEX idx_content_hash (content_hash)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                    """
                )
                self._ensure_faq_columns(cursor)
            connection.commit()

    def _ensure_faq_columns(self, cursor: pymysql.cursors.Cursor) -> None:
        """
        兼容迁移：确保faq表包含必要字段

        【迁移字段】
        - domain: 领域
        - intent: 意图ID
        - action: 动作类型
        - content_hash: 内容哈希
        """
        cursor.execute("SHOW COLUMNS FROM faq")
        columns = {row["Field"] for row in cursor.fetchall()}

        if "domain" not in columns:
            logger.info("faq表缺少domain字段，执行兼容迁移")
            cursor.execute("ALTER TABLE faq ADD COLUMN domain VARCHAR(128) NOT NULL DEFAULT '' AFTER id")
            columns.add("domain")

        if "intent" not in columns and "intent_id" in columns:
            logger.info("faq表使用旧字段intent_id，新增intent字段并迁移数据")
            cursor.execute("ALTER TABLE faq ADD COLUMN intent VARCHAR(64) NOT NULL DEFAULT '' AFTER domain")
            cursor.execute("UPDATE faq SET intent = intent_id WHERE intent = ''")
            columns.add("intent")

        required_columns = {"intent", "question", "answer"}
        missing_required = required_columns - columns
        if missing_required:
            raise RuntimeError(f"faq表缺少必要字段: {', '.join(sorted(missing_required))}")

        migrations = {
            "action": "ALTER TABLE faq ADD COLUMN action VARCHAR(128) NOT NULL DEFAULT '' AFTER intent",
            "content_hash": "ALTER TABLE faq ADD COLUMN content_hash CHAR(64) NOT NULL DEFAULT '' AFTER answer",
        }
        for column, statement in migrations.items():
            if column not in columns:
                logger.info("faq表缺少%s字段，执行兼容迁移", column)
                cursor.execute(statement)

    def count_faq(self) -> int:
        """
        获取FAQ总数

        【返回】
        FAQ记录数量
        """
        with self.connect_database() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) AS total FROM faq")
                return int(cursor.fetchone()["total"])

    def sync_faq_items(self, faq_items: list[FAQItem]) -> FAQSyncResult:
        """
        同步FAQ数据到MySQL

        【实现逻辑】
        1. 读取现有FAQ记录
        2. 比较content_hash判断是否需要更新
        3. 执行INSERT或UPDATE

        【返回】
        同步结果统计
        """
        before = self.count_faq()
        existing_by_key: dict[tuple[str, str], dict[str, Any]] = {}

        with self.connect_database() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT id, domain, intent, action, question, answer, content_hash FROM faq")
                for row in cursor.fetchall():
                    existing_by_key[(row["intent"], row["question"])] = row

                result = FAQSyncResult(mysql_total_before=before, mysql_total_after=before)
                source_keys = {(item.intent_id, item.question) for item in faq_items}
                result.missing_in_source = len(set(existing_by_key) - source_keys)

                for item in faq_items:
                    existing = existing_by_key.get((item.intent_id, item.question))
                    if existing is None:
                        cursor.execute(
                            """
                            INSERT INTO faq (domain, intent, action, question, answer, content_hash)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            """,
                            (item.domain, item.intent_id, item.action, item.question, item.answer, item.content_hash),
                        )
                        result.inserted += 1
                        continue

                    if existing.get("content_hash") != item.content_hash or existing.get("answer") != item.answer:
                        cursor.execute(
                            """
                            UPDATE faq
                            SET domain=%s, action=%s, answer=%s, content_hash=%s
                            WHERE id=%s
                            """,
                            (item.domain, item.action, item.answer, item.content_hash, existing["id"]),
                        )
                        result.updated += 1
                    else:
                        result.unchanged += 1

            connection.commit()

        result.mysql_total_after = self.count_faq()
        logger.info(
            "MySQL FAQ同步完成: before=%s after=%s inserted=%s updated=%s unchanged=%s extra_in_mysql=%s",
            result.mysql_total_before,
            result.mysql_total_after,
            result.inserted,
            result.updated,
            result.unchanged,
            result.missing_in_source,
        )
        return result

    def iter_faq_records(self, batch_size: int) -> Iterable[list[dict[str, Any]]]:
        """
        分批迭代FAQ记录

        【参数】
        - batch_size: 每批记录数

        【返回】
        分批的FAQ记录列表生成器
        """
        offset = 0
        while True:
            with self.connect_database() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT id, domain, intent, action, question, answer
                        FROM faq
                        ORDER BY id
                        LIMIT %s OFFSET %s
                        """,
                        (batch_size, offset),
                    )
                    records = cursor.fetchall()
            if not records:
                break
            yield records
            offset += batch_size

    def get_distinct_intents(self) -> set[str]:
        """
        获取FAQ中所有不重复的意图ID

        【返回】
        意图ID集合
        """
        with self.connect_database() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT DISTINCT intent FROM faq")
                return {row["intent"] for row in cursor.fetchall()}


# ================================================================================
# 第九节：知识库初始化器
# ================================================================================
class KnowledgeBaseInitializer:
    """
    知识库初始化器

    【职责】
    1. 连接Milvus和Elasticsearch
    2. 执行向量数据导入
    3. 验证导入结果

    【导入策略】
    - 如果已有数据且数量足够，跳过导入
    - 支持强制重导入模式

    【健壮性】
    - 向量库不可用时优雅降级
    - 单批失败不影响其他批次
    """

    def __init__(self, settings: InitSettings, repository: FAQRepository):
        """初始化知识库初始化器"""
        self.settings = settings
        self.repository = repository
        self.knowledge_base = KnowledgeBase()

    def ensure_connected(self) -> None:
        """
        确保知识库已连接

        【异常】
        RuntimeError: Milvus和Elasticsearch均未连接
        """
        self.knowledge_base.ensure_connected()
        if self.knowledge_base.vector_store is None and self.knowledge_base.elasticsearch_store is None:
            raise RuntimeError("Milvus和Elasticsearch均未连接，知识库初始化无法继续")

    def check_knowledge_base(self) -> dict[str, Any]:
        """
        检查知识库状态

        【检查项】
        - Milvus集合是否存在
        - Milvus实体数量
        - Milvus集合字段
        - Elasticsearch可用性

        【返回】
        知识库状态字典
        """
        summary: dict[str, Any] = {
            "milvus_collection_exists": False,
            "milvus_entity_count": 0,
            "milvus_fields": [],
            "elasticsearch_available": self.knowledge_base.elasticsearch_store is not None,
        }

        milvus_stats = self.knowledge_base.get_milvus_collection_stats(self.settings.milvus_collection)
        summary["milvus_collection_exists"] = milvus_stats["collection_exists"]
        summary["milvus_entity_count"] = milvus_stats["entity_count"]
        summary["milvus_fields"] = milvus_stats["fields"]

        logger.info("知识库检查结果: %s", json.dumps(summary, ensure_ascii=False))
        return summary

    def import_mysql_faq_to_knowledge_base(self) -> VectorImportResult:
        """
        将MySQL中的FAQ导入到向量知识库

        【导入策略】
        1. 如果设置跳过向量导入，直接返回
        2. 如果Milvus已有足够数据且未强制重导入，跳过
        3. 分批从MySQL读取并导入到向量库

        【返回】
        导入结果统计
        """
        if self.settings.skip_vector_import:
            return VectorImportResult(skipped=True, reason="INIT_SKIP_VECTOR_IMPORT=true")

        kb_summary = self.check_knowledge_base()
        mysql_total = self.repository.count_faq()
        if (
            not self.settings.force_vector_reimport
            and kb_summary["milvus_entity_count"] >= mysql_total
            and mysql_total > 0
        ):
            reason = "Milvus已有数据且数量不少于MySQL FAQ数量，跳过向量重导入"
            logger.info(reason)
            return VectorImportResult(skipped=True, reason=reason)

        result = VectorImportResult()
        for records in self.repository.iter_faq_records(self.settings.batch_size):
            documents = [self._record_to_document(record) for record in records]
            if not documents:
                continue

            if self.knowledge_base.elasticsearch_store is not None:
                try:
                    self.knowledge_base.elasticsearch_store.add_documents(documents)
                    result.elasticsearch_imported += len(documents)
                    logger.info("成功添加 %s 条FAQ到Elasticsearch", len(documents))
                except Exception:
                    logger.exception("添加FAQ到Elasticsearch失败")

            if self.knowledge_base.vector_store is not None:
                try:
                    self.knowledge_base.vector_store.add_documents(documents)
                    result.milvus_imported += len(documents)
                    logger.info("成功添加 %s 条FAQ到Milvus", len(documents))
                except Exception:
                    logger.exception("添加FAQ到Milvus失败")

        self.knowledge_base.cache.clear()
        logger.info(
            "MySQL FAQ导入知识库完成: milvus=%s elasticsearch=%s",
            result.milvus_imported,
            result.elasticsearch_imported,
        )
        return result

    def run_sample_searches(self) -> None:
        """
        执行样例检索验证

        【用途】
        验证知识库初始化后检索功能是否正常
        """
        for query in self.settings.sample_queries:
            try:
                results = self.knowledge_base.search(query, k=3)
                logger.info("样例检索 query=%s result_count=%s", query, len(results))
            except Exception:
                logger.exception("样例检索失败: %s", query)

    def _record_to_document(self, record: dict[str, Any]) -> Document:
        """
        将数据库记录转换为LangChain Document

        【参数】
        - record: 数据库记录字典

        【返回】
        LangChain Document对象

        【文档结构】
        - page_content: "问题: 意图:xxx 动作:xxx 问题:xxx\n答案:xxx"
        - metadata: 包含type、mysql_id、intent_id等
        """
        action = record.get("action") or ""
        domain = record.get("domain") or ""
        intent = record.get("intent") or ""
        question = record.get("question") or ""
        answer = record.get("answer") or ""
        enhanced_question = f"意图: {intent} 动作: {action} 问题: {question}"
        doc = Document(
            page_content=f"问题: {enhanced_question}\n答案: {answer}",
            metadata={
                "type": "faq",
                "mysql_id": str(record.get("id", "")),
                "intent_id": intent,
                "domain": domain,
                "original_question": question,
                "action": action,
            },
        )
        doc.metadata["doc_id"] = self.knowledge_base._generate_doc_id(doc)
        return doc


# ================================================================================
# 第十节：孤儿节点检测
# ================================================================================
def detect_orphan_intents(repository: FAQRepository, intents_data: dict[str, Any]) -> dict[str, Any]:
    """
    检测孤儿节点

    【功能】
    检测意图树与实际FAQ数据的一致性

    【检测类型】
    1. 孤儿节点：FAQ中有但intents.yaml中无
    2. 未使用节点：intents.yaml中有但FAQ中无

    【用途】
    - 辅助维护意图树完整性
    - 发现配置遗漏或数据不一致
    """
    faq_intents = repository.get_distinct_intents()
    yaml_intents = set((intents_data.get("intents") or {}).keys())
    orphan_nodes = sorted(faq_intents - yaml_intents)
    unused_yaml_nodes = sorted(yaml_intents - faq_intents)
    result = {
        "total_faq_intents": len(faq_intents),
        "total_yaml_intents": len(yaml_intents),
        "orphan_nodes": orphan_nodes,
        "unused_yaml_nodes": unused_yaml_nodes,
        "has_orphans": bool(orphan_nodes),
    }
    logger.info("意图一致性检查结果: %s", json.dumps(result, ensure_ascii=False))
    return result


# ================================================================================
# 第十一节：主流程编排
# ================================================================================
def run_initialization() -> dict[str, Any]:
    """
    执行完整初始化流程

    【流程】
    1. 加载环境配置
    2. 校验FAQ数据
    3. 同步意图树
    4. 初始化MySQL表
    5. 同步FAQ到MySQL
    6. 检测孤儿意图
    7. 连接知识库
    8. 导入向量数据
    9. 执行样例检索验证

    【错误处理】
    各阶段独立重试，失败不影响已成功阶段

    【返回】
    初始化结果汇总
    """
    setup_environment()
    configure_logging()
    settings = InitSettings.from_env()
    logger.info("开始初始化智能客服Agent前置数据")
    logger.info(
        "初始化配置: mysql=%s:%s/%s milvus=%s/%s elasticsearch=%s/%s batch_size=%s force_vector_reimport=%s",
        settings.mysql_host,
        settings.mysql_port,
        settings.mysql_database,
        settings.milvus_uri,
        settings.milvus_collection,
        settings.elasticsearch_url,
        settings.elasticsearch_index,
        settings.batch_size,
        settings.force_vector_reimport,
    )

    faq_items = load_faq_data()
    intents_data = sync_intents_yaml(faq_items)
    repository = FAQRepository(settings)

    retry("创建MySQL数据库和faq表", settings.retry_attempts, settings.retry_interval_seconds)(
        repository.ensure_database_and_table
    )
    mysql_sync = retry("同步FAQ到MySQL", settings.retry_attempts, settings.retry_interval_seconds)(
        lambda: repository.sync_faq_items(faq_items)
    )
    intent_check = detect_orphan_intents(repository, intents_data)

    kb_initializer = KnowledgeBaseInitializer(settings, repository)
    retry("连接Milvus/Elasticsearch知识库", settings.retry_attempts, settings.retry_interval_seconds)(
        kb_initializer.ensure_connected
    )
    kb_check_before = kb_initializer.check_knowledge_base()
    vector_import = retry("MySQL FAQ导入Milvus/Elasticsearch", settings.retry_attempts, settings.retry_interval_seconds)(
        kb_initializer.import_mysql_faq_to_knowledge_base
    )
    kb_check_after = kb_initializer.check_knowledge_base()
    kb_initializer.run_sample_searches()

    summary = {
        "mysql_sync": mysql_sync.__dict__,
        "intent_check": intent_check,
        "knowledge_base_before": kb_check_before,
        "vector_import": vector_import.__dict__,
        "knowledge_base_after": kb_check_after,
    }
    logger.info("初始化完成: %s", json.dumps(summary, ensure_ascii=False))
    return summary


# ================================================================================
# 第十二节：独立功能入口
# ================================================================================
def import_mysql_faq_to_knowledge_base_only() -> VectorImportResult:
    """
    仅执行向量导入（不包含MySQL同步）

    【用途】
    在MySQL已有数据，仅需更新向量库时使用
    """
    setup_environment()
    configure_logging()
    settings = InitSettings.from_env()
    repository = FAQRepository(settings)
    initializer = KnowledgeBaseInitializer(settings, repository)
    initializer.ensure_connected()
    return initializer.import_mysql_faq_to_knowledge_base()


def check_knowledge_base_only() -> dict[str, Any]:
    """
    仅检查知识库状态

    【用途】
    检查Milvus/Elasticsearch中的数据状态
    """
    setup_environment()
    configure_logging()
    settings = InitSettings.from_env()
    repository = FAQRepository(settings)
    initializer = KnowledgeBaseInitializer(settings, repository)
    initializer.ensure_connected()
    return initializer.check_knowledge_base()


def sync_mysql_faq_only() -> FAQSyncResult:
    """
    仅同步FAQ到MySQL（不执行向量导入）

    【用途】
    更新MySQL数据，暂不更新向量库
    """
    setup_environment()
    configure_logging()
    settings = InitSettings.from_env()
    faq_items = load_faq_data()
    repository = FAQRepository(settings)
    repository.ensure_database_and_table()
    return repository.sync_faq_items(faq_items)


# ================================================================================
# 第十三节：命令行入口
# ================================================================================
def parse_args() -> argparse.Namespace:
    """
    解析命令行参数

    【参数】
    --mode: 初始化模式
    - all: 完整流程
    - mysql: 仅同步FAQ到MySQL
    - vector: 仅MySQL导入知识库
    - check: 仅检查知识库
    """
    parser = argparse.ArgumentParser(description="智能客服Agent前置数据初始化")
    parser.add_argument(
        "--mode",
        choices=("all", "mysql", "vector", "check"),
        default="all",
        help="初始化模式：all=完整流程，mysql=仅同步FAQ到MySQL，vector=仅MySQL导入知识库，check=仅检查知识库",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "all":
        run_initialization()
    elif args.mode == "mysql":
        sync_mysql_faq_only()
    elif args.mode == "vector":
        import_mysql_faq_to_knowledge_base_only()
    elif args.mode == "check":
        check_knowledge_base_only()


if __name__ == "__main__":
    main()
