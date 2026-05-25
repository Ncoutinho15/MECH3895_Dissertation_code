import json
import os
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


VALID_STAGES = {"early", "mid", "late"}
VALID_IRRITABILITY = {"high", "moderate", "low"}


class ExercisePlannerEngine:
    def __init__(self, library_path=None, llm_engine=None):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.library_path = library_path or os.path.join(self.script_dir, "exercise_library.json")
        self.report_dir = os.path.join(self.script_dir, "reports")
        os.makedirs(self.report_dir, exist_ok=True)

        self.llm_engine = llm_engine
        self.library = self.load_library()
        self.validate_library_schema()

    def load_library(self):
        if not os.path.exists(self.library_path):
            raise FileNotFoundError(f"Exercise library not found: {self.library_path}")

        with open(self.library_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def validate_library_schema(self):
        required_root = ["safety_rules", "dosage_rules", "exercises"]
        for key in required_root:
            if key not in self.library:
                raise ValueError(f"Exercise library is missing root key: {key}")

        seen_ids = set()
        required_exercise_keys = [
            "id", "name", "conditions", "stages", "irritability",
            "default_sets", "default_reps", "frequency",
            "instructions", "avoid", "xai_reason"
        ]

        for exercise in self.library.get("exercises", []):
            missing = [key for key in required_exercise_keys if key not in exercise]
            if missing:
                raise ValueError(f"Exercise {exercise.get('id', '<unknown>')} missing keys: {missing}")

            if exercise["id"] in seen_ids:
                raise ValueError(f"Duplicate exercise id: {exercise['id']}")
            seen_ids.add(exercise["id"])

            if not set(s.lower() for s in exercise.get("stages", [])).issubset(VALID_STAGES):
                raise ValueError(f"Invalid stages in exercise {exercise['id']}")

            if not set(i.lower() for i in exercise.get("irritability", [])).issubset(VALID_IRRITABILITY):
                raise ValueError(f"Invalid irritability values in exercise {exercise['id']}")

        return True

    def _normalise_patient_data(self, patient_data):
        profile = {
            "patient_id": str(patient_data.get("patient_id", "")).strip(),
            "name": str(patient_data.get("name", "")).strip(),
            "age": str(patient_data.get("age", "")).strip(),
            "condition": str(patient_data.get("condition", "")).strip(),
            "stage": str(patient_data.get("stage", "")).strip().lower(),
            "irritability": str(patient_data.get("irritability", "")).strip().lower(),
            "limitations": str(patient_data.get("limitations", "")).strip(),
            "affected_side": str(patient_data.get("affected_side", "")).strip(),
            "pain_score": str(patient_data.get("pain_score", "")).strip(),
        }

        if profile["stage"] not in VALID_STAGES:
            raise ValueError(f"Invalid rehabilitation stage: {profile['stage']}")

        if profile["irritability"] not in VALID_IRRITABILITY:
            raise ValueError(f"Invalid irritability: {profile['irritability']}")

        return profile

    def _planner_rules(self):
        return self.library.get("planner_rules", {})

    def _stage_rule(self, stage):
        return self.library.get("dosage_rules", {}).get(stage, {
            "sets_range": [2, 3],
            "reps_range": [8, 12],
            "frequency": "3 days/week"
        })

    def _safe_int(self, value, default):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _matches_condition(self, exercise, condition):
        return any(condition.lower() == c.lower() for c in exercise.get("conditions", []))

    def _matches_stage(self, exercise, stage):
        return stage in [s.lower() for s in exercise.get("stages", [])]

    def _matches_irritability(self, exercise, irritability):
        return irritability in [i.lower() for i in exercise.get("irritability", [])]

    def select_safe_candidates(self, patient_data):
        condition = patient_data.get("condition", "")
        stage = patient_data.get("stage", "")
        irritability = patient_data.get("irritability", "")

        exact_matches = []
        relaxed_irritability_matches = []

        for exercise in self.library.get("exercises", []):
            if not self._matches_condition(exercise, condition):
                continue
            if not self._matches_stage(exercise, stage):
                continue

            if self._matches_irritability(exercise, irritability):
                exact_matches.append(exercise)
            elif self._planner_rules().get("allow_irritability_relaxation", True):
                relaxed_irritability_matches.append(exercise)

        candidates = exact_matches + relaxed_irritability_matches
        target_size = min(
            len(candidates),
            int(self._planner_rules().get("target_programme_size", 5))
        )

        warnings = []
        minimum_safe = int(self._planner_rules().get("minimum_safe_programme_size", 3))
        if len(candidates) < minimum_safe:
            warnings.append(
                f"Only {len(candidates)} stage-appropriate exercise(s) are available for this profile. "
                "The programme has been returned with fewer items rather than relaxing stage safety."
            )

        return candidates[:target_size], warnings

    def _clamp_dosage(self, exercise, stage_rule, proposed_sets=None, proposed_reps=None):
        sets_range = stage_rule.get("sets_range", [2, 3])
        reps_range = stage_rule.get("reps_range", [8, 12])

        default_sets = exercise.get("default_sets", sets_range[0])
        default_reps = exercise.get("default_reps", reps_range[0])

        sets = self._safe_int(proposed_sets, default_sets)
        reps = self._safe_int(proposed_reps, default_reps)

        sets = max(sets_range[0], min(sets, sets_range[1]))
        reps = max(reps_range[0], min(reps, reps_range[1]))
        return sets, reps

    def _build_plan_item(
        self,
        exercise,
        patient_data,
        selection_mode="rule_based",
        llm_item=None,
        retrieved_evidence=None
    ):
        stage_rule = self._stage_rule(patient_data["stage"])
        sets, reps = self._clamp_dosage(
            exercise,
            stage_rule,
            proposed_sets=(llm_item or {}).get("sets"),
            proposed_reps=(llm_item or {}).get("reps")
        )

        frequency = (llm_item or {}).get("frequency") or exercise.get("frequency", stage_rule.get("frequency", ""))

        plan_item = {
            "id": exercise.get("id", ""),
            "name": exercise.get("name", ""),
            "sets": sets,
            "reps": reps,
            "frequency": frequency,
            "hold_seconds": exercise.get("hold_seconds"),
            "instructions": exercise.get("instructions", ""),
            "avoid": exercise.get("avoid", ""),
            "reason": exercise.get("xai_reason", ""),
            "source": exercise.get("source", {}),
        }

        llm_reason = (llm_item or {}).get("reason", "")
        plan_item["xai_explanation"] = self.explain_exercise_selection(
            user_profile=patient_data,
            exercise=exercise,
            retrieved_evidence=retrieved_evidence or [],
            guideline_refs=self.get_guideline_refs(),
            corroboration_scores=self.load_corroboration_scores(),
            selection_mode=selection_mode,
            llm_reason=llm_reason,
        )

        return plan_item

    def _build_rule_based_programme(self, candidates, patient_data):
        programme = []
        for exercise in candidates:
            programme.append(
                self._build_plan_item(
                    exercise=exercise,
                    patient_data=patient_data,
                    selection_mode="rule_based"
                )
            )
        return programme

    def _validate_llm_plan(self, llm_payload, candidates, patient_data):
        target_size = min(
            len(candidates),
            int(self._planner_rules().get("target_programme_size", 5))
        )

        allowed_by_id = {exercise["id"]: exercise for exercise in candidates}
        allowed_by_name = {exercise["name"].lower(): exercise for exercise in candidates}

        programme = []
        used_ids = set()
        retrieved = llm_payload.get("retrieved_context", [])

        for item in llm_payload.get("plan", []):
            if not isinstance(item, dict):
                continue

            exercise = None
            exercise_id = str(item.get("exercise_id", "")).strip()
            exercise_name = str(item.get("exercise_name", "")).strip().lower()

            if exercise_id:
                exercise = allowed_by_id.get(exercise_id)
            if exercise is None and exercise_name:
                exercise = allowed_by_name.get(exercise_name)

            if exercise is None or exercise["id"] in used_ids:
                continue

            programme.append(
                self._build_plan_item(
                    exercise=exercise,
                    patient_data=patient_data,
                    selection_mode="constrained_llm_rag",
                    llm_item=item,
                    retrieved_evidence=retrieved,
                )
            )
            used_ids.add(exercise["id"])

            if len(programme) >= target_size:
                break

        for exercise in candidates:
            if len(programme) >= target_size:
                break
            if exercise["id"] in used_ids:
                continue

            programme.append(
                self._build_plan_item(
                    exercise=exercise,
                    patient_data=patient_data,
                    selection_mode="rule_based_fill",
                    retrieved_evidence=retrieved,
                )
            )
            used_ids.add(exercise["id"])

        return programme

    def generate_programme(self, patient_data):
        patient_profile = self._normalise_patient_data(patient_data)
        candidates, warnings = self.select_safe_candidates(patient_profile)

        if not candidates:
            return {
                "patient_data": patient_profile,
                "programme": [],
                "safety_rules": self.library.get("safety_rules", {}),
                "created_at": datetime.now().strftime("%d/%m/%Y %H:%M"),
                "generation_metadata": {
                    "selection_mode": "empty",
                    "warnings": warnings + ["No safe exercise candidates matched the supplied profile."],
                    "candidate_count": 0,
                }
            }

        generation_mode = "rule_based"
        llm_payload = {"retrieved_context": [], "plan": []}

        if self.llm_engine is not None:
            try:
                llm_payload = self.llm_engine.plan_programme(patient_profile, candidates)
                programme = self._validate_llm_plan(llm_payload, candidates, patient_profile)
                if programme:
                    generation_mode = "constrained_llm_rag"
                else:
                    programme = self._build_rule_based_programme(candidates, patient_profile)
                    warnings.append("LLM output could not be validated; deterministic fallback used.")
            except Exception as exc:
                programme = self._build_rule_based_programme(candidates, patient_profile)
                warnings.append(f"LLM planner fallback triggered: {exc}")
        else:
            programme = self._build_rule_based_programme(candidates, patient_profile)

        return {
            "patient_data": patient_profile,
            "programme": programme,
            "safety_rules": self.library.get("safety_rules", {}),
            "created_at": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "generation_metadata": {
                "selection_mode": generation_mode,
                "warnings": warnings,
                "candidate_count": len(candidates),
                "retrieved_context_count": len(llm_payload.get("retrieved_context", [])),
            }
        }

    def explain_exercise_selection(
        self,
        user_profile,
        exercise,
        retrieved_evidence=None,
        guideline_refs=None,
        corroboration_scores=None,
        selection_mode="rule_based",
        llm_reason=""
    ):
        retrieved_evidence = retrieved_evidence or []
        guideline_refs = guideline_refs or []
        corroboration_scores = corroboration_scores or {}

        condition = user_profile["condition"]
        stage = user_profile["stage"]
        irritability = user_profile["irritability"]

        explanation = {
            "exercise": exercise["name"],
            "selected_because": [],
            "evidence_sources": [],
            "safety_checks": [],
            "confidence_factors": {}
        }

        if condition in exercise.get("conditions", []):
            explanation["selected_because"].append(
                f"The exercise is approved for {condition}."
            )

        if stage in exercise.get("stages", []):
            explanation["selected_because"].append(
                f"It matches the rehabilitation stage: {stage}."
            )

        if irritability in exercise.get("irritability", []):
            explanation["selected_because"].append(
                f"It is suitable for {irritability} irritability symptoms."
            )

        if exercise.get("xai_reason"):
            explanation["selected_because"].append(exercise["xai_reason"])

        if llm_reason:
            explanation["selected_because"].append(llm_reason)

        pain_rules = self.library.get("safety_rules", {}).get("pain_monitoring", {})
        explanation["safety_checks"].append(
            f"Pain must remain at or below {pain_rules.get('max_allowed_pain_during_exercise', 5)}/10 during exercise."
        )
        explanation["safety_checks"].append(
            pain_rules.get(
                "next_morning_rule",
                "Pain should settle by the following morning and not worsen week to week."
            )
        )

        explanation["evidence_sources"].append(exercise.get("source", {}))
        for item in retrieved_evidence[:3]:
            explanation["evidence_sources"].append({
                "condition": item.get("condition", ""),
                "section": item.get("section", ""),
                "text": item.get("text", ""),
            })

        key = str((condition.lower(), "exercises"))
        explanation["confidence_factors"]["corroboration_score"] = corroboration_scores.get(key)
        explanation["confidence_factors"]["selection_mode"] = selection_mode
        explanation["confidence_factors"]["retrieved_context_count"] = len(retrieved_evidence)
        explanation["confidence_factors"]["retrieved_guidelines"] = [
            ref.get("summary", "") for ref in guideline_refs[:3]
        ]

        return explanation

    def save_pdf_report(self, programme_data):
        if not REPORTLAB_AVAILABLE:
            return None

        filename = f"exercise_programme_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        path = os.path.join(self.report_dir, filename)

        doc = SimpleDocTemplate(path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("Exercise Programme Report", styles["Title"]))
        story.append(Spacer(1, 12))

        patient = programme_data.get("patient_data", {})
        metadata = programme_data.get("generation_metadata", {})

        details = [
            ["Patient ID", patient.get("patient_id", "")],
            ["Name", patient.get("name", "")],
            ["Affected Side", patient.get("affected_side", "")],
            ["Pain Score", patient.get("pain_score", "")],
            ["Age", patient.get("age", "")],
            ["Condition", patient.get("condition", "")],
            ["Rehabilitation Stage", patient.get("stage", "")],
            ["Tissue Irritability", patient.get("irritability", "")],
            ["Functional Limitations", patient.get("limitations", "")],
            ["Generation Mode", metadata.get("selection_mode", "")],
        ]

        table = Table(details, colWidths=[160, 330])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.lightblue),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP")
        ]))
        story.append(table)
        story.append(Spacer(1, 18))

        story.append(Paragraph("Generated Exercise Programme", styles["Heading2"]))

        for index, exercise in enumerate(programme_data.get("programme", []), start=1):
            story.append(Paragraph(f"{index}. {exercise['name']}", styles["Heading3"]))
            story.append(Paragraph(
                f"Sets: {exercise['sets']} | Reps: {exercise['reps']} | Frequency: {exercise['frequency']}",
                styles["BodyText"]
            ))

            if exercise.get("hold_seconds"):
                story.append(Paragraph(f"Hold: {exercise['hold_seconds']} seconds", styles["BodyText"]))

            story.append(Paragraph(f"Instructions: {exercise['instructions']}", styles["BodyText"]))
            story.append(Paragraph(f"Avoid: {exercise['avoid']}", styles["BodyText"]))
            story.append(Paragraph(f"Clinical reasoning: {exercise['reason']}", styles["BodyText"]))
            story.append(Spacer(1, 10))

        safety = programme_data.get("safety_rules", {}).get("pain_monitoring", {})
        if safety:
            story.append(Paragraph("Safety Rules", styles["Heading2"]))
            story.append(Paragraph(
                f"Pain should remain at or below {safety.get('max_allowed_pain_during_exercise', 5)}/10 during exercise. "
                f"{safety.get('next_morning_rule', '')}",
                styles["BodyText"]
            ))

        warnings = metadata.get("warnings", [])
        if warnings:
            story.append(Spacer(1, 12))
            story.append(Paragraph("Generation Warnings", styles["Heading2"]))
            for warning in warnings:
                story.append(Paragraph(f"- {warning}", styles["BodyText"]))

        doc.build(story)
        return path

    def get_guideline_refs(self):
        path = os.path.join(self.script_dir, "clinical_guidelines.json")
        if not os.path.exists(path):
            return []

        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        refs = []
        for topic_refs in data.get("guidelines_by_topic", {}).values():
            refs.extend(topic_refs)
        return refs

    def load_corroboration_scores(self):
        path = os.path.join(self.script_dir, "corroboration.json")
        if not os.path.exists(path):
            return {}

        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
