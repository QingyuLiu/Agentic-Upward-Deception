from __future__ import annotations
from pathlib import Path
from typing import Dict


class PromptStore:
    def __init__(self, prompt_dir: str):
        self.prompt_dir = Path(prompt_dir)
        if not self.prompt_dir.exists():
            raise FileNotFoundError(f"Prompt dir not found: {self.prompt_dir}")

    def load_all(self) -> Dict[str, str]:
        mapping = {
            "type_1": "judge_template_1.txt",
            "type_2": "judge_template_2.txt",
            "type_3_stage1": "judge_template_3_failure.txt",
            "type_3_stage2": "judge_template_3_local_decoy.txt",
            "type_4_download": "judge_template_4_failure.txt",
            "type_4_related": "judge_template_4_relatedness.txt",
            "type_4_forged": "judge_template_4_source_attribution.txt",
            "type_4_localize_error": "judge_template_4_type_source_error.txt",
            "type_4_localize_source": "judge_template_4_source_attribution_of_file.txt",
        }
        prompts: Dict[str, str] = {}
        for k, fname in mapping.items():
            p = self.prompt_dir / fname
            prompts[k] = p.read_text(encoding="utf-8")
        return prompts
