import time

from openai import OpenAI
from Models.LLM_config import api_key as default_api_key, base_url as default_base_url, model as default_model
import openai

class LLM_agent:
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0

    # 模型价格映射（单位：美元 / 每 1000 tokens）
    model_price_map = {
        "gpt-4.1": {"prompt": 0.002, "completion": 0.008},
        "gpt-4.1-mini": {"prompt": 0.0004, "completion": 0.0016},
    }

    def __init__(self, api_key=None, base_url=None, model=None,
                 system_prompt=None, max_history=10, history=None):
        self.client = OpenAI(
            api_key=api_key or default_api_key,
            base_url=base_url or default_base_url
        )
        self.model = model or default_model
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.history = history or []
        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

    def ask(self, text, max_retry=3, wait=2):
        """向 LLM 提问并记录对话与用量"""
        self.history.append({"role": "user", "content": text})
        # 截断历史
        if self.system_prompt:
            user_assistant = self.history[1:]
            if len(user_assistant) > self.max_history:
                user_assistant = user_assistant[-self.max_history:]
            self.history = [self.history[0]] + user_assistant
        else:
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]

        for attempt in range(max_retry):
            try:
                # 请求
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.history,
                    stream=False
                )
                break
            except openai.OpenAIError as e:  # 捕捉所有 OpenAI SDK 异常
                print("OpenAI API Error:", e)
            time.sleep(wait)
        else:
            raise RuntimeError(f"❌ LLM API 连续 {max_retry} 次失败，请检查网络或服务状态")

        answer = response.choices[0].message.content.strip()
        self.history.append({"role": "assistant", "content": answer})

        # === ✅ 统计 tokens 和费用 ===
        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens

        price_info = self.model_price_map.get(self.model, {"prompt": 0, "completion": 0})
        cost = (prompt_tokens / 1000 * price_info["prompt"] +
                completion_tokens / 1000 * price_info["completion"])

        # 更新类级别统计
        LLM_agent.total_prompt_tokens += prompt_tokens
        LLM_agent.total_completion_tokens += completion_tokens
        LLM_agent.total_cost += cost

        return answer

    def get_history(self):
        return self.history

    def reset_history(self):
        self.history = []
        if self.system_prompt:
            self.history.append({"role": "system", "content": self.system_prompt})

    @classmethod
    def get_total_usage(cls):
        return {
            "total_prompt_tokens": cls.total_prompt_tokens,
            "total_completion_tokens": cls.total_completion_tokens,
            "total_tokens": cls.total_prompt_tokens + cls.total_completion_tokens,
            "total_cost_usd": round(cls.total_cost, 6)
        }



