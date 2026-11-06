"""以最小请求探测指定 LLM 模型是否可用于后续仿真评测。"""

import argparse
import json

from policies.llm_adversarial_planner import LLMAdversarialPlanner


def main():
    """调用一次 Responses 接口，只输出模型可用性和非敏感错误摘要。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    planner = LLMAdversarialPlanner(model_name=args.model)
    try:
        planner.client.responses.create(
            model=planner.model_name,
            input="Reply with OK.",
            extra_body={"enable_thinking": False},
        )
    except Exception as error:
        print(json.dumps({
            "model": planner.model_name,
            "available": False,
            "error_type": type(error).__name__,
            "status_code": getattr(error, "status_code", None),
            "message": str(error),
        }, ensure_ascii=False))
        raise SystemExit(1)
    print(json.dumps({"model": planner.model_name, "available": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
