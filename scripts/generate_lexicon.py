from __future__ import annotations

import itertools
from pathlib import Path

HEADER = """Domain-specific bilingual lexicon for industrial automation.

The entries are synthesized from curated terminology lists covering robotics,
vision systems, production analytics, and safety operations.  Each record
contains an English phrase, a Simplified Chinese translation, and a short
contextual tag describing the subsystem where the phrase appears.  The file is
programmatically generated to ensure consistency and to provide a rich dataset
for translation experiments.
"""

ENG_NOUNS = [
    "actuator",
    "analytics",
    "assembly",
    "calibration",
    "cell",
    "controller",
    "diagnostics",
    "drive",
    "firmware",
    "gantry",
    "inspection",
    "latency",
    "maintenance",
    "manifold",
    "manipulator",
    "material",
    "module",
    "monitor",
    "motor",
    "network",
    "operator",
    "payload",
    "platform",
    "quality",
    "safety",
    "sensor",
    "simulator",
    "telemetry",
    "tooling",
    "uptime",
    "vision",
    "workflow",
]

ENG_ADJECTIVES = [
    "adaptive",
    "aerostatic",
    "autonomous",
    "calibrated",
    "cloud",
    "collaborative",
    "cognitive",
    "composite",
    "digital",
    "edge",
    "fault-tolerant",
    "federated",
    "frictionless",
    "gantry-aligned",
    "graph-based",
    "high-bandwidth",
    "high-precision",
    "immersive",
    "inertial",
    "low-drift",
    "low-latency",
    "modular",
    "predictive",
    "probabilistic",
    "reconfigurable",
    "reinforced",
    "resilient",
    "self-healing",
    "semantic",
    "sensor-fused",
    "stable",
    "synchronized",
    "ultra-fast",
    "vision-guided",
]

ENG_VERBS = [
    "alignment",
    "allocation",
    "analysis",
    "balancing",
    "calibration",
    "commissioning",
    "diagnostics",
    "forecasting",
    "handover",
    "inspection",
    "localization",
    "monitoring",
    "optimization",
    "prediction",
    "profiling",
    "recovery",
    "retuning",
    "scheduling",
    "synchronization",
    "triangulation",
    "verification",
]

CONTEXTS = {
    "vision": "视觉系统",
    "robotics": "机器人控制",
    "translation": "语音与翻译",
    "analytics": "生产分析",
    "safety": "安全联锁",
    "infrastructure": "基础设施",
}

CONTEXT_MAP = {
    "vision": ["calibration", "inspection", "triangulation", "alignment"],
    "robotics": ["actuator", "controller", "manipulator", "handover"],
    "translation": ["monitor", "operator", "network", "communication"],
    "analytics": ["analytics", "forecasting", "prediction", "telemetry"],
    "safety": ["safety", "recovery", "diagnostics", "uptime"],
    "infrastructure": ["network", "platform", "cloud", "workflow"],
}

ZH_ADJECTIVES = {
    "adaptive": "自适应",
    "aerostatic": "静压",
    "autonomous": "自主",
    "calibrated": "校准",
    "cloud": "云化",
    "collaborative": "协作",
    "cognitive": "认知",
    "composite": "复合",
    "digital": "数字化",
    "edge": "边缘",
    "fault-tolerant": "容错",
    "federated": "联邦",
    "frictionless": "无摩擦",
    "gantry-aligned": "龙门对齐",
    "graph-based": "基于图",
    "high-bandwidth": "高带宽",
    "high-precision": "高精度",
    "immersive": "沉浸式",
    "inertial": "惯性",
    "low-drift": "低漂移",
    "low-latency": "低时延",
    "modular": "模块化",
    "predictive": "预测",
    "probabilistic": "概率",
    "reconfigurable": "可重构",
    "reinforced": "强化",
    "resilient": "韧性",
    "self-healing": "自愈",
    "semantic": "语义",
    "sensor-fused": "传感融合",
    "stable": "稳定",
    "synchronized": "同步",
    "ultra-fast": "超高速",
    "vision-guided": "视觉导引",
}

ZH_NOUNS = {
    "actuator": "执行器",
    "analytics": "分析",
    "assembly": "装配",
    "calibration": "标定",
    "cell": "单元",
    "controller": "控制器",
    "diagnostics": "诊断",
    "drive": "驱动",
    "firmware": "固件",
    "gantry": "龙门",
    "inspection": "检测",
    "latency": "时延",
    "maintenance": "维护",
    "manifold": "汇流板",
    "manipulator": "机械臂",
    "material": "物料",
    "module": "模块",
    "monitor": "监控",
    "motor": "电机",
    "network": "网络",
    "operator": "操作员",
    "payload": "负载",
    "platform": "平台",
    "quality": "质量",
    "safety": "安全",
    "sensor": "传感器",
    "simulator": "仿真器",
    "telemetry": "遥测",
    "tooling": "工装",
    "uptime": "运行时间",
    "vision": "视觉",
    "workflow": "工作流",
}

ZH_VERBS = {
    "alignment": "对准",
    "allocation": "分配",
    "analysis": "分析",
    "balancing": "平衡",
    "calibration": "校准",
    "commissioning": "调试",
    "diagnostics": "诊断",
    "forecasting": "预测",
    "handover": "交接",
    "inspection": "检测",
    "localization": "定位",
    "monitoring": "监控",
    "optimization": "优化",
    "prediction": "预测",
    "profiling": "分析",
    "recovery": "恢复",
    "retuning": "重新整定",
    "scheduling": "调度",
    "synchronization": "同步",
    "triangulation": "三角测量",
    "verification": "验证",
}


OUTPUT = Path("echoliner/translation/domain_lexicon.py")


def main() -> None:
    entries: list[tuple[str, str, str]] = []
    for adjective, noun, verb in itertools.product(ENG_ADJECTIVES, ENG_NOUNS, ENG_VERBS):
        phrase = f"{adjective} {noun} {verb}"
        zh_phrase = f"{ZH_ADJECTIVES[adjective]} {ZH_NOUNS[noun]} {ZH_VERBS[verb]}"
        context = "general"
        for ctx, keywords in CONTEXT_MAP.items():
            if noun in keywords or verb in keywords:
                context = ctx
                break
        context_label = CONTEXTS.get(context, "跨领域")
        entries.append((phrase, zh_phrase, context_label))

    with OUTPUT.open("w", encoding="utf8") as fh:
        fh.write('"""' + HEADER + '"""\n\n')
        fh.write("DOMAIN_LEXICON: list[tuple[str, str, str]] = [\n")
        for eng, zh, ctx in entries:
            fh.write(f"    (\"{eng}\", \"{zh}\", \"{ctx}\"),\n")
        fh.write("]\n")


if __name__ == "__main__":
    main()
