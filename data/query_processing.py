import re

# 预编译正则模式
SPLIT_PATTERN = re.compile(r'[。？！?!；;,.]+')

DEPENDENT_PATTERNS = [
    re.compile(p) for p in [
        r'^帮我.{0,4}(查|看|处理|确认|一下)',
        r'^(麻烦|劳烦|请帮).{0,6}$',
        r'^(好的|谢谢|知道了|明白|嗯|哦)',
        r'^\d{5,}$',
        r'^(我的|这个|那个).{0,4}$',
        r'^.{1,4}$',
    ]
]


def split_user_input(text: str) -> list[str]:
    """
    对用户原始输入做轻量句子拆分。
    核心逻辑：按标点切句 → 识别并过滤/合并补充句 → 返回独立句子列表。
    如果无法切出多句，返回包含原始输入的单元素列表。
    """
    # 1. 按标点切句
    parts = [p.strip() for p in SPLIT_PATTERN.split(text) if p.strip()]

    # 没切出多句，直接返回原始输入
    if len(parts) <= 1:
        return [text]

    # 2. 识别"补充句"模式（不能独立表达意图的句子）
    result = []
    for part in parts:
        is_dependent = any(pattern.search(part) for pattern in DEPENDENT_PATTERNS)
        if is_dependent:
            # 补充句合并到上一句（极罕见情况下为第一句）
            if result:
                result[-1] = result[-1] + "，" + part
        else:
            result.append(part)

    return result if result else [text]