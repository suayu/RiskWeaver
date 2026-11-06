"""生成 RiskWeaver 中英文系统总览图，供论文与内部审阅使用。"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent


def box(ax, x, y, w, h, text, color, fontsize=8, linewidth=1.0):
    """绘制圆角模块；坐标统一使用 0--1 画布。"""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=color, edgecolor="#243447", linewidth=linewidth,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color="#17202A", linespacing=1.25)
    return patch


def arrow(ax, start, end, color="#3B4A5A", style="-|>", width=1.25, rad=0.0):
    """绘制有向数据流；弧度用于避免反馈线与主流程重叠。"""
    patch = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=10,
        linewidth=width, color=color,
        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2,
    )
    ax.add_patch(patch)
    return patch


ZH = {
    "title": "RiskWeaver：历史画像驱动的分层动静态对抗场景生成与能力评估",
    "state": "闭环状态与道路\n自车、背景车、道路、历史",
    "profile": "完整历史自车画像\n分位数 · 置信区间 · EWMA\n场景条件化脆弱性",
    "candidates": "结构化异构风险候选\n压迫 · 切入 · 合流 · 施工路段\n风险 / 安全裕度 / 画像匹配",
    "llm": "LLM 高级规划\n候选排序与离散选择\n风险源 · 目标 · 策略 · 锚点 / 模板",
    "diff": "异构风险实例化\n动态：Diffusion 联合轨迹\n静态：参数化施工障碍模板",
    "gate": "联合执行门控\n语义 · 动力学 · 背景 OBB 安全\n静态障碍物 · 正理论可行驶区域",
    "exec": "Safe-Sim 闭环执行\n执行预测前缀并重新规划\n计划—预测—控制全过程追踪",
    "risk": "低反应余量风险与能力评估\nTTC · EA · 可达区域收缩 · 反应时间\n危险度 D · 路线表现 · 能力分数 S",
    "history": "策略—攻击收益表\n执行帧、D、区域变化、失败原因",
    "discrete": "离散语义层",
    "continuous": "动态轨迹与静态几何层",
    "feedback": "行为与攻击结果反馈",
    "reject": "拒绝：记录阶段、原因与指标",
}

EN = {
    "title": "RiskWeaver: Ego-Profile-Aware Hierarchical Adversarial Scenario Generation",
    "state": "Closed-Loop Context\nEgo and traffic states\nRoads and history",
    "profile": "Full-History Ego Profile\nQuantiles · CIs · EWMA\nConditional vulnerability",
    "candidates": "Structured Candidates\nPressure · cut-in · merge\nRisk · safety · profile match",
    "llm": "LLM Planner\nRank candidates\nTarget · strategy · anchors",
    "diff": "Diffusion Refinement\nJoint continuous trajectories\nAnchor and TTC guidance",
    "gate": "Execution Gates\nSemantics · dynamics · OBB\nObstacles · feasible region",
    "exec": "Closed-Loop Execution\nExecute prefix and replan\nFull action tracing",
    "risk": "Risk and Capability\nTTC · EA · reachable area\nResponse · route · D · S",
    "history": "Strategy–Attack Return Memory\nFrames · D · area change · failure",
    "discrete": "Discrete semantic layer",
    "continuous": "Continuous trajectory layer",
    "feedback": "Behavior and outcome feedback",
    "reject": "Reject: record stage, reason, and metric",
}


def draw(labels, stem, font_family, box_fontsize=8.0):
    plt.rcParams.update({
        "font.family": font_family,
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(7.16, 4.45))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(0.5, 0.965, labels["title"], ha="center", va="top",
            fontsize=11, fontweight="bold", color="#17202A")

    # 主流程采用统一的从左到右结构，便于中英文图直接替换。
    box(ax, 0.025, 0.68, 0.18, 0.14, labels["state"], "#D9EAF7", box_fontsize)
    box(ax, 0.235, 0.68, 0.20, 0.14, labels["profile"], "#DDF2E1", box_fontsize)
    box(ax, 0.465, 0.68, 0.23, 0.14, labels["candidates"], "#FFF0C9", box_fontsize - 0.3)
    box(ax, 0.725, 0.68, 0.245, 0.14, labels["llm"], "#F8D7E8", box_fontsize - 0.3)

    box(ax, 0.725, 0.39, 0.245, 0.14, labels["diff"], "#DCD7F7", box_fontsize)
    box(ax, 0.465, 0.39, 0.23, 0.14, labels["gate"], "#FADBD8", box_fontsize - 0.4)
    box(ax, 0.235, 0.39, 0.20, 0.14, labels["exec"], "#D6EAF8", box_fontsize - 0.4)
    box(ax, 0.025, 0.39, 0.18, 0.14, labels["risk"], "#D5F5E3", box_fontsize - 0.7)
    box(ax, 0.305, 0.105, 0.39, 0.12, labels["history"], "#FDEBD0", box_fontsize - 0.1)

    # 前向数据流。
    arrow(ax, (0.205, 0.75), (0.235, 0.75))
    arrow(ax, (0.435, 0.75), (0.465, 0.75))
    arrow(ax, (0.695, 0.75), (0.725, 0.75))
    arrow(ax, (0.847, 0.68), (0.847, 0.53))
    arrow(ax, (0.725, 0.46), (0.695, 0.46))
    arrow(ax, (0.465, 0.46), (0.435, 0.46))
    arrow(ax, (0.235, 0.46), (0.205, 0.46))

    # 执行结果进入历史表，再回到画像与候选排序。
    arrow(ax, (0.115, 0.39), (0.305, 0.165), color="#2874A6", rad=-0.18)
    arrow(ax, (0.50, 0.225), (0.335, 0.68), color="#2874A6", rad=-0.23)
    arrow(ax, (0.695, 0.165), (0.84, 0.68), color="#2874A6", rad=0.28)

    # 门控拒绝作为可观测结果，而非静默丢弃。
    ax.text(0.58, 0.345, labels["reject"], ha="center", va="center",
            fontsize=7.2, color="#922B21", fontweight="bold")
    arrow(ax, (0.58, 0.39), (0.58, 0.36), color="#C0392B")

    ax.text(0.85, 0.86, labels["discrete"], ha="center", va="center",
            fontsize=7.4, color="#7D3C98", fontweight="bold")
    ax.text(0.85, 0.57, labels["continuous"], ha="center", va="center",
            fontsize=7.4, color="#5B2C6F", fontweight="bold")
    ax.text(0.50, 0.055, labels["feedback"], ha="center", va="center",
            fontsize=7.4, color="#1F618D", fontweight="bold")

    fig.tight_layout(pad=0.35)
    for suffix in ("png", "pdf", "svg"):
        path = ROOT / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=("zh", "en", "both"), default="both")
    args = parser.parse_args()
    (ROOT / "zh").mkdir(parents=True, exist_ok=True)
    (ROOT / "en").mkdir(parents=True, exist_ok=True)
    # 服务器使用明确的中文字体文件，避免 Matplotlib 字体缓存漏检 CJK 字体。
    cjk_path = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
    if cjk_path.exists():
        font_manager.fontManager.addfont(cjk_path)
        zh_family = font_manager.FontProperties(fname=cjk_path).get_name()
    else:
        zh_family = "Microsoft YaHei"
    if args.language in ("zh", "both"):
        draw(ZH, "zh/riskweaver_architecture_zh", [zh_family, "DejaVu Sans"])
    if args.language in ("en", "both"):
        draw(EN, "en/riskweaver_architecture_en", ["DejaVu Serif"], box_fontsize=7.0)


if __name__ == "__main__":
    main()
