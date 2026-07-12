from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


HateTarget = Literal[
    "Race / ethnicity / nationality", "Religion", "Caste", "Gender / sex",
    "Sexual orientation", "Disability", "Other protected identity", "Unclear",
]
AbuseTarget = Literal[
    "Individual person", "Protected identity group", "Body size / appearance",
    "Intelligence / competence", "Political belief or ideology", "Behavior / actions",
    "Family or personal life", "Other", "Unclear",
]


class Annotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    identity_hate_speech: Literal["0", "1"]
    hate_severity: Literal["low", "medium", "high"] | None
    hate_target_group: HateTarget | None
    abuse_level: Literal["none", "mild", "moderate", "severe"]
    abuse_target: AbuseTarget | None
    counterspeech: Literal["0", "1"] | None
    confidence: Literal["low", "medium", "high"]
    flag_for_review: bool
    notes: str

    @model_validator(mode="after")
    def conditional_fields_are_consistent(self) -> "Annotation":
        if self.identity_hate_speech == "0":
            if self.hate_severity is not None or self.hate_target_group is not None:
                raise ValueError("non-hate annotations require null hate details")
        elif self.hate_severity is None or self.hate_target_group is None:
            raise ValueError("hate annotations require severity and target")
        if self.abuse_level == "none" and self.abuse_target is not None:
            raise ValueError("non-abusive annotations require a null abuse target")
        if self.abuse_level != "none" and self.abuse_target is None:
            raise ValueError("abusive annotations require an abuse target")
        return self
