"""Example: Candidate Profile — minimal sOAP example.

Usage: python candidate-profile.py <backend-url> [api_type] [api_key]
"""

from enum import Enum
import sys

import asyncio

from peteos.chatbot.manager import ChatBotManager
from peteos.oap.base import AgenticObjectBase
from peteos.oap.decorators import tool


class JobRole(Enum):
    FRONTEND_DEVELOPER = "frontend-developer"
    BACKEND_DEVELOPER = "backend-developer"
    FULLSTACK_DEVELOPER = "fullstack-developer"
    NOT_FITTING = "not-fitting"


class CandidateProfile(AgenticObjectBase):
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


async def main():
    #from peteos.utils.logger import setup_logging
    #setup_logging(level="DEBUG", debug=True)
    args = sys.argv[1:] + [None] * 3
    url, api_type, api_key = args[:3]
    await ChatBotManager.add_backend("local", url, api_type=api_type, api_key=api_key)

    biography = (
        "Sarah spent 5 years building React dashboards and recently "
        "added Python and FastAPI to her toolkit. She has also "
        "deployed microservices on Kubernetes."
    )
    profile = CandidateProfile(biography)

    # First invocation: determine the job role
    job_role = await profile.invoke_agent(
        prompt="Which role is Sarah best suited for?",
        output_schema=JobRole,
    )
    print(job_role)  # JobRole.FULLSTACK_DEVELOPER

    # Second invocation: extract skills
    skills = await profile.invoke_agent(
        prompt="What skills does Sarah have?",
        output_schema=list[str],
    )
    print(skills)  # ['React', 'Python', 'FastAPI', 'Kubernetes']


if __name__ == "__main__":
    asyncio.run(main())
