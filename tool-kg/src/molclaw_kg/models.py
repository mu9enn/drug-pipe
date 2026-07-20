from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class Slot(BaseModel):
    name: str
    slot_path: str = ""
    direction: Literal["input", "output", "precondition", "side_effect"] | None = None
    raw_type: str = "unknown"
    default: Any = None
    enum: list[Any] | None = None
    semantic_type: str = "unknown"
    format: str = "unknown"
    unit: str | None = None
    cardinality: Literal["single", "list", "map", "unknown"] = "unknown"
    parameter_kind: Literal["data", "config", "control", "unknown"] = "unknown"
    requirement_status: Literal["required", "optional", "conditional"] = "optional"
    required: bool = False
    description: str = ""
    source: Literal["input_schema", "output_schema", "description", "inferred", "doc"] = "inferred"
    confidence: float = 0.5
    connectable_state: Literal["yes", "no", "unknown"] = "unknown"
    evidence_refs: list[str] = Field(default_factory=list)


class SlotAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    semantic_type: str = "unknown"
    format: str = "unknown"
    parameter_kind: Literal["data", "config", "control", "unknown"] = "unknown"
    connectable: bool | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class SkillDerivedSlot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slot_path: str
    name: str
    direction: Literal["output", "precondition", "side_effect"]
    raw_type: str = "unknown"
    semantic_type: str = "unknown"
    format: str = "unknown"
    cardinality: Literal["single", "list", "map", "unknown"] = "unknown"
    parameter_kind: Literal["data", "config", "control", "unknown"] = "unknown"
    description: str = ""
    connectable: bool | None = None
    evidence_refs: list[str] = Field(min_length=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class SkillDerivedRequirementSet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    set_id: str = Field(pattern=r"^skill::[A-Za-z0-9_.-]+$")
    condition: str
    required_slots: list[str] = Field(default_factory=list)
    optional_slots: list[str] = Field(default_factory=list)
    defaulted_slots: list[str] = Field(default_factory=list)
    execution_meaning: str
    evidence_refs: list[str] = Field(min_length=1)


class ToolAnnotationPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool_id: str
    description_summary: str = ""
    aliases: list[str] = Field(default_factory=list)
    slot_annotations: dict[str, SlotAnnotation] = Field(default_factory=dict)
    skill_derived_slots: list[SkillDerivedSlot] = Field(default_factory=list)
    skill_derived_requirement_sets: list[SkillDerivedRequirementSet] = Field(default_factory=list)
    needs_review: bool = False


class ToolCard(BaseModel):
    # Lean production tool-card fields used by candidate/adjudication/sampling.
    tool_id: str
    title: str
    description_summary: str
    primary_stage: str = "simulation_prediction"
    scheduling_stages: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    schema_slots: list[Slot] = Field(default_factory=list)
    slot_annotations: dict[str, SlotAnnotation] = Field(default_factory=dict)
    skill_derived_slots: list[Slot] = Field(default_factory=list)
    inputs: list[Slot] = Field(default_factory=list)
    outputs: list[Slot] = Field(default_factory=list)
    connectable_inputs: list[Slot] = Field(default_factory=list)
    connectable_outputs: list[Slot] = Field(default_factory=list)
    input_requirement_sets: list[dict[str, Any]] = Field(default_factory=list)
    preconditions: list[Slot] = Field(default_factory=list)
    side_effects: list[Slot] = Field(default_factory=list)
    needs_review: bool = False


class CandidatePair(BaseModel):
    pair_id: str
    source_tool: str
    target_tool: str
    source_stage: str
    target_stage: str
    taxonomy_supporting_stage_pairs: list[list[str]] = Field(default_factory=list)
    proposal_reasons: list[str] = Field(default_factory=list)
    recall_risk: str | None = None


class EdgeTypeDecision(BaseModel):
    type: str
    source_slot: str | None = None
    target_slot_or_precondition: str | None = None
    confidence: float = 0.5
    evidence_ids: list[str] = Field(default_factory=list)


class SatisfiedMapping(BaseModel):
    source_output_slot: str = ""
    target_input_slot: str = ""
    semantic_match: Literal["exact", "compatible", "convertible", "incompatible", "unknown"] = "unknown"
    format_match: Literal["exact", "compatible", "convertible", "incompatible", "unknown"] = "unknown"
    evidence_refs: list[str] = Field(default_factory=list)
    note: str = ""


class UnsatisfiedRequiredInput(BaseModel):
    target_input_slot: str = ""
    reason: str = ""
    can_be_user_provided: bool = True
    can_be_satisfied_by_other_upstream_tool: bool = True


class AdjudicationRecord(BaseModel):
    pair_id: str
    relation_status: Literal["valid", "negative", "uncertain", "alternative"]
    direct_transition: bool
    edge_types: list[EdgeTypeDecision] = Field(default_factory=list)
    negative_reason: str | None = None
    satisfied_mappings: list[SatisfiedMapping] = Field(default_factory=list)
    unsatisfied_required_inputs: list[UnsatisfiedRequiredInput] = Field(default_factory=list)
    context: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    rationale: str = ""
    agent_model: str = "claude-cc-v1"
    agent_confidence: float = 0.5
    raw_payload_hash: str | None = None


class FinalEdge(BaseModel):
    edge_id: str
    pair_id: str
    source_tool: str
    target_tool: str
    edge_type: str | None = None
    direct_transition: bool
    source_slot: str | None = None
    target_slot: str | None = None
    stage_src: str
    stage_tgt: str
    relation_status: Literal["valid", "negative", "uncertain", "alternative"]
    confidence_raw: float
    confidence_calibrated: float
    view: Literal["core", "expanded", "uncertain", "negative"]
    evidence_ids: list[str] = Field(default_factory=list)
    negative_reason: str | None = None
    created_at: str
    run_id: str
