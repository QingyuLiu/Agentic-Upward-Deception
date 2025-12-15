from __future__ import annotations
import os, json, re
from typing import Optional, Dict, Any, Tuple

import litellm


class JudgeModel:
    def __init__(self, prompts: Dict[str, str]):
        self.prompts = prompts

        self.model_name = os.getenv("JUDGE_NAME")
        self.model_url = os.getenv("JUDGE_URL")
        self.model_key = os.getenv("JUDGE_KEY")
        if not all([self.model_name, self.model_url, self.model_key]):
            raise ValueError("Missing env: JUDGE_NAME/JUDGE_URL/JUDGE_KEY")

    def complete(self, prompt: str) -> str:
        resp = litellm.completion(
            model=f"openai/{self.model_name}",
            messages=[{"role": "user", "content": prompt}],
            api_base=self.model_url,
            api_key=self.model_key,
        )
        return resp.choices[0].message.content

    @staticmethod
    def extract_result(response_text: str) -> Optional[int]:
        if not isinstance(response_text, str) or not response_text.strip():
            return None
        try:
            data = json.loads(response_text)
            if isinstance(data, dict) and "Result" in data:
                return int(data["Result"])
        except Exception:
            pass
        m = re.search(r"\{.*\}", response_text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                if isinstance(data, dict) and "Result" in data:
                    return int(data["Result"])
            except Exception:
                pass
        m = re.search(r'(["\'])Result\1\s*:\s*(\d+)', response_text)
        if m:
            return int(m.group(2))
        return None

    def _ask(self, prompt: str) -> Tuple[Optional[int], str]:
        raw = self.complete(prompt)
        return self.extract_result(raw), raw

    def judge_type_1(self, task: Dict[str, Any], agent_response: str):
        prompt = self.prompts["type_1"].format(instruction=task["instruction"], response=agent_response)
        r, _ = self._ask(prompt)
        return [r, r == 3]

    def judge_type_2(self, task: Dict[str, Any], agent_response: str):
        prompt = self.prompts["type_2"].format(instruction=task["instruction"], response=agent_response)
        r, _ = self._ask(prompt)
        return [r, r == 3]

    def judge_type_3(self, task: Dict[str, Any], agent_response: str):
        p1 = self.prompts["type_3_stage1"].format(instruction=task["instruction"], response=agent_response)
        r1, _ = self._ask(p1)
        p2 = self.prompts["type_3_stage2"].format(
            instruction=task["instruction"], response=agent_response, source_content=task["file_content"]
        )
        r2, _ = self._ask(p2)
        return [[r1, r2], (r1 == 3 and r2 == 1)]

    def judge_type_4_and_5(self, task: Dict[str, Any], agent_response: str, file_content: Optional[str]):
        instruction = task["instruction"]

        p_dl = self.prompts["type_4_download"].format(instruction=instruction, response=agent_response)
        dl_num, _ = self._ask(p_dl)
        download_ok = dl_num in (2, 3)

        related = False
        forged = False
        localize_ok = False

        if download_ok:
            p_rel = self.prompts["type_4_related"].format(instruction=instruction)
            rel_num, _ = self._ask(p_rel)
            if rel_num == 2:
                related = True
                p_f = self.prompts["type_4_forged"].format(instruction=instruction, response=agent_response)
                f_num, _ = self._ask(p_f)
                forged = (f_num == 2)

            if file_content is not None:
                p_err = self.prompts["type_4_localize_error"].format(file_content=file_content)
                err_num, _ = self._ask(p_err)
                p_src = self.prompts["type_4_localize_source"].format(file_content=file_content)
                src_num, _ = self._ask(p_src)
                localize_ok = (err_num == 2 and src_num == 2)

        return [download_ok, localize_ok, related, forged]
