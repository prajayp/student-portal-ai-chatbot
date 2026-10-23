"""User roles supported by the portal."""

from enum import StrEnum


class Role(StrEnum):
    STUDENT = "student"
    FACULTY = "faculty"
    ADMINISTRATOR = "administrator"


ALL_ROLES: frozenset[Role] = frozenset(Role)
