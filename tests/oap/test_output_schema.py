"""Test for OAP output schema via CandidateProfile."""

from enum import Enum

from peteos.oap.agentic_object import AgenticObject
from peteos.oap.decorators import tool


class JobRole(Enum):
    FRONTEND_DEVELOPER = "frontend-developer"
    BACKEND_DEVELOPER = "backend-developer"
    FULLSTACK_DEVELOPER = "fullstack-developer"
    NOT_FITTING = "not-fitting"


class CandidateProfile(AgenticObject):
    """You are a candidate profiler.
       You reason about job applicants' biographies to determine
       which role they are best suited for and extract their skills.
    """

    def __init__(self, biography: str):
        super().__init__()
        self._biography = biography

    @tool
    def get_biography(self) -> str:
        """Return the biography of the candidate."""
        return self._biography


async def test_candidate_profile_job_role():
    biography = (
        "Sarah spent 5 years building React dashboards and recently "
        "added Python and FastAPI to her toolkit. She has also "
        "deployed microservices on Kubernetes."
    )
    profile = CandidateProfile(biography)
    role = await profile.invoke_agent(
        "Which role is Sarah best suited for?",
        output_schema=JobRole,
    )
    assert role in (JobRole.FULLSTACK_DEVELOPER, JobRole.FRONTEND_DEVELOPER, JobRole.BACKEND_DEVELOPER)


async def test_candidate_profile_skills():
    biography = (
        "Sarah spent 5 years building React dashboards and recently "
        "added Python and FastAPI to her toolkit. She has also "
        "deployed microservices on Kubernetes."
    )
    profile = CandidateProfile(biography)
    skills = await profile.invoke_agent(
        "What skills does Sarah have?",
        output_schema=list[str],
    )
    assert isinstance(skills, list)
    assert len(skills) >= 1
    assert all(isinstance(s, str) for s in skills)
    assert any("react" in s.lower() for s in skills)
