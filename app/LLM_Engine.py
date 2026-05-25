import json
import os
import re
from typing import Any, Dict, List

from llama_index.core import Settings, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import TextNode
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP


def _first_existing_path(*paths: str) -> str:
    for path in paths:
        if path and os.path.exists(path):
            return path
    return paths[0]


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

LLM_MODEL_PATH = _first_existing_path(
    os.path.join(BASE_DIR, "models", "Phi-3-mini-4k-instruct-q4.gguf"),
    os.path.join(PARENT_DIR, "models", "Phi-3-mini-4k-instruct-q4.gguf"),
    "./models/Phi-3-mini-4k-instruct-q4.gguf",
)

EMBED_MODEL_PATH = _first_existing_path(
    os.path.join(BASE_DIR, "models", "bge-small-en-v1.5"),
    os.path.join(PARENT_DIR, "models", "bge-small-en-v1.5"),
    "./models/bge-small-en-v1.5",
)

INDEX_DIR = _first_existing_path(
    os.path.join(BASE_DIR, "storage"),
    os.path.join(PARENT_DIR, "storage"),
    "./storage",
)

METADATA_PATH = _first_existing_path(
    os.path.join(BASE_DIR, "metadata.json"),
    os.path.join(PARENT_DIR, "metadata.json"),
    "./metadata.json",
)


class RehabLLM:
    def __init__(self, similarity_top_k: int = 3):
        self.similarity_top_k = similarity_top_k
        self.llm = None
        self.index = None

        self.header_prompt = """
You are an AI physiotherapy assistant designed for home-based shoulder and arm rehabilitation.

Core rules:
- Use simple, clear, plain English
- Be concise and practical
- Never diagnose
- If uncertain, say you are uncertain
- Prioritise safety and conservative advice
"""

        self.chatbot_prompt = """
You are answering rehabilitation questions using retrieved context where available.

Constraints:
- Do not invent exercises
- Do not diagnose
- Use supportive, calm language
- If context is insufficient, state that clearly
"""

        self.general_prompt = """
You are a concise assistant.
- Reply naturally
- Keep it short
"""

        self.planner_prompt = """
You are generating a personalised rehabilitation exercise programme.

You MUST obey these rules:
- Use ONLY exercises listed in the allowed exercise library JSON
- Do not invent or rename exercises
- Respect rehabilitation stage and irritability
- Keep dosage conservative and realistic
- Return STRICT JSON only
"""

        self.guardrails = """
Hard constraints:
- Only use exercises explicitly present in the provided allowed exercise list
- Do not exceed safe dosage ranges provided in the prompt
- Prefer conservative choices when evidence is limited
- If information is insufficient, return fewer exercises rather than unsafe ones
"""

    def load_model(self):
        if self.llm is not None:
            return

        self.llm = LlamaCPP(
            model_path=LLM_MODEL_PATH,
            temperature=0.1,
            max_new_tokens=512,
            context_window=4096,
            model_kwargs={
                "n_threads": 4,
                "stop": ["<|end|>", "<|user|>", "## Instruction", "### Instruction"],
            },
        )

        embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_PATH)
        Settings.llm = self.llm
        Settings.embed_model = embed_model

        self.index = self._load_or_build_index()

    def _load_or_build_index(self):
        if os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR):
            storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
            return load_index_from_storage(storage_context)

        if not os.path.exists(METADATA_PATH):
            return None

        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not data:
            return None

        nodes = [
            TextNode(
                text=f"{item.get('condition', '')} | {item.get('section', '')} | {item.get('text', '')}",
                metadata=item,
            )
            for item in data
        ]

        index = VectorStoreIndex(nodes)
        index.storage_context.persist(persist_dir=INDEX_DIR)
        return index

    def retrieve(self, query: str, similarity_top_k: int = None) -> List[Dict[str, Any]]:
        self.load_model()

        if self.index is None:
            return []

        retriever = self.index.as_retriever(similarity_top_k=similarity_top_k or self.similarity_top_k)
        nodes = retriever.retrieve(query)

        results = []
        for node in nodes:
            results.append({
                "text": node.text,
                "condition": node.metadata.get("condition", ""),
                "section": node.metadata.get("section", ""),
                "score": getattr(node, "score", None),
            })
        return results

    def _serialise_context(self, retrieved: List[Dict[str, Any]]) -> str:
        if not retrieved:
            return "No retrieved rehabilitation context was found."

        return "\n\n".join(
            f"- Condition: {item.get('condition', '')}\n"
            f"  Section: {item.get('section', '')}\n"
            f"  Evidence: {item.get('text', '')}"
            for item in retrieved
        )

    def _build_chat_prompt(self, query: str, retrieved: List[Dict[str, Any]], mode: str = "chat") -> str:
        body = self.general_prompt if mode == "general" else self.chatbot_prompt
        context = self._serialise_context(retrieved)

        return f"""<|system|>
{self.header_prompt}

{body}

{self.guardrails}
<|end|>
<|user|>
Context:
{context}

User question:
{query}

Reply in under 120 words.
<|end|>
<|assistant|>
"""

    def chat(self, query: str) -> str:
        self.load_model()

        q = query.lower()
        mode = "general" if all(token not in q for token in ["exercise", "pain", "shoulder", "arm", "rehab"]) else "chat"
        retrieved = [] if mode == "general" else self.retrieve(query)

        prompt = self._build_chat_prompt(query, retrieved, mode=mode)
        response = self.llm.complete(prompt)
        return response.text.strip()[:600]

    def _planner_query_from_profile(self, patient_data: Dict[str, Any]) -> str:
        return (
            f"rehabilitation guidance for {patient_data.get('condition', '')}; "
            f"stage {patient_data.get('stage', '')}; "
            f"irritability {patient_data.get('irritability', '')}; "
            f"limitations {patient_data.get('limitations', '')}"
        )

    def _extract_json_object(self, text: str) -> Dict[str, Any]:
        print("\n===== RAW LLM PLANNER OUTPUT =====")
        print(text)
        print("===== END RAW LLM PLANNER OUTPUT =====\n")

        # Keep only text between first { and last }
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in model response")

        json_text = text[start:end + 1]

        parsed = json.loads(json_text)

        # Repair common LLM typo
        for item in parsed.get("plan", []):
            if "exercise_id" not in item and "exerccipl_id" in item:
                item["exercise_id"] = item["exerccipl_id"]

        return parsed

    def _minimal_candidate_view(self, exercise_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        minimal = []
        for exercise in exercise_candidates:
            minimal.append({
                "exercise_id": exercise.get("id"),
                "exercise_name": exercise.get("name"),
                "conditions": exercise.get("conditions", []),
                "stages": exercise.get("stages", []),
                "irritability": exercise.get("irritability", []),
                "default_sets": exercise.get("default_sets"),
                "default_reps": exercise.get("default_reps"),
                "frequency": exercise.get("frequency"),
                "instructions": exercise.get("instructions"),
                "avoid": exercise.get("avoid"),
                "xai_reason": exercise.get("xai_reason"),
            })
        return minimal

    def _build_planner_prompt(
        self,
        patient_data: Dict[str, Any],
        exercise_candidates: List[Dict[str, Any]],
        retrieved: List[Dict[str, Any]],
    ) -> str:
        allowed_exercises = json.dumps(self._minimal_candidate_view(exercise_candidates), indent=2)
        user_profile = json.dumps(patient_data, indent=2)
        context = self._serialise_context(retrieved)

        return f"""<|system|>
{self.header_prompt}

{self.planner_prompt}

{self.guardrails}
<|end|>
<|user|>
Patient profile JSON:
{user_profile}

Retrieved rehabilitation context:
{context}

Allowed exercise library JSON:
{allowed_exercises}

Return ONLY valid JSON. Do not use markdown. Do not add explanations outside JSON.
Use this exact format:
{{
  "plan": [
    {{
      "exercise_id": "string",
      "exercise_name": "string",
      "sets": 0,
      "reps": 0,
      "frequency": "string",
      "reason": "string",
      "caution": "string"
    }}
  ]
}}

Rules:
- Prefer between 3 and 5 exercises depending on safe availability
- Never include an exercise outside the allowed list
- Use the exact exercise_id and exercise_name from the allowed list
- Keep reasoning brief and clinically sensible
- If safe coverage is limited, return fewer exercises
- Do not add markdown or any text outside the JSON
<|end|>
<|assistant|>
"""

    def plan_programme(self, patient_data: Dict[str, Any], exercise_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.load_model()

        retrieved = self.retrieve(self._planner_query_from_profile(patient_data))
        prompt = self._build_planner_prompt(patient_data, exercise_candidates, retrieved)
        response = self.llm.complete(prompt)
        raw_text = response.text.strip()

        parsed = self._extract_json_object(raw_text)
        return {
            "plan": parsed.get("plan", []),
            "retrieved_context": retrieved,
            "raw_response": raw_text,
        }

    # backwards-compatible shim
    def plan(self, query: str) -> str:
        return self.chat(query)
