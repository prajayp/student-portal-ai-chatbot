"""Role-based flows, multi-turn context and fallback behaviour."""

import pytest

from app.dialogue.handlers import REGISTRY
from app.dialogue.manager import Resolution
from app.dialogue.roles import Role
from app.dialogue.state import SessionStore


# ---------------------------------------------------------------- roles


@pytest.mark.parametrize(
    ("message", "role"),
    [
        ("how do I submit grades", Role.STUDENT),
        ("how do I submit grades", Role.ADMINISTRATOR),
        ("apply for casual leave", Role.STUDENT),
        ("show admission statistics", Role.STUDENT),
        ("show admission statistics", Role.FACULTY),
        ("post a notice", Role.FACULTY),
    ],
)
def test_restricted_intents_are_refused(chat, message: str, role: Role) -> None:
    result = chat(message, role=role)
    assert result.resolved_by is Resolution.RESTRICTED
    assert "only available to" in result.reply


@pytest.mark.parametrize(
    ("message", "role", "intent"),
    [
        ("how do I submit grades", Role.FACULTY, "grade_submission"),
        ("apply for casual leave", Role.FACULTY, "leave_request"),
        ("show admission statistics", Role.ADMINISTRATOR, "admission_stats"),
        ("post a notice", Role.ADMINISTRATOR, "post_notice"),
    ],
)
def test_allowed_roles_get_answers(chat, message: str, role: Role, intent: str) -> None:
    result = chat(message, role=role)
    assert result.intent == intent
    assert result.resolved_by is Resolution.CLASSIFIER


def test_same_intent_differs_by_role(manager) -> None:
    replies = {role: manager.handle(f"s-{role}", role, "what's my attendance").reply for role in Role}
    assert "Your attendance this semester" in replies[Role.STUDENT]
    assert "Attendance in your courses" in replies[Role.FACULTY]
    assert "Attendance overview" in replies[Role.ADMINISTRATOR]


def test_library_limits_depend_on_role(manager) -> None:
    student = manager.handle("a", Role.STUDENT, "library timings").reply
    faculty = manager.handle("b", Role.FACULTY, "library timings").reply
    assert "borrow 4 books" in student
    assert "borrow 10 books" in faculty


def test_notices_are_filtered_by_audience(manager) -> None:
    student = manager.handle("a", Role.STUDENT, "show the latest notices").reply
    admin = manager.handle("b", Role.ADMINISTRATOR, "show the latest notices").reply
    assert "Internal marks upload deadline" not in student
    assert "Internal marks upload deadline" in admin
    assert "awaiting approval" in admin


def test_every_registered_intent_is_trainable(classifier) -> None:
    assert set(REGISTRY) == set(classifier.labels)


# ---------------------------------------------------------------- multi-turn context


def test_department_follow_up_reuses_last_intent(chat) -> None:
    first = chat("What's the exam schedule for CSE?")
    assert first.intent == "exam_schedule"
    second = chat("and for ECE?")
    assert second.intent == "exam_schedule"
    assert second.resolved_by is Resolution.FOLLOW_UP
    assert second.slots["department"] == "ECE"
    assert "Electronics and Communication" in second.reply


def test_semester_follow_up_keeps_department(chat) -> None:
    chat("What's the exam schedule for CSE?")
    chat("and for ECE?")
    result = chat("what about semester 5?")
    assert result.intent == "exam_schedule"
    assert result.slots == {"department": "ECE", "semester": 5}
    assert "EC501" in result.reply


def test_clarification_question_then_answer(chat) -> None:
    question = chat("show me the class timetable")
    assert question.intent == "timetable"
    assert "Which department" in question.reply
    assert "CSE" in question.suggestions

    question = chat("CSE")
    assert question.resolved_by is Resolution.CLARIFICATION
    assert "Which semester" in question.reply

    answer = chat("semester 5")
    assert answer.resolved_by is Resolution.CLARIFICATION
    assert "CSE semester 5 weekly timetable" in answer.reply


def test_remembered_slots_carry_into_a_new_intent(chat) -> None:
    chat("exam schedule for mechanical semester 3")
    result = chat("show the class timetable")
    assert result.intent == "timetable"
    assert "ME semester 3 weekly timetable" in result.reply


def test_new_department_drops_stale_course_code(chat) -> None:
    chat("tell me about CS501")
    result = chat("courses in ECE semester 5")
    assert "course_code" not in result.slots
    assert "ECE semester 5 courses" in result.reply


def test_follow_up_ignored_when_slot_is_irrelevant(chat) -> None:
    chat("what are the library timings")
    result = chat("and for ECE?")
    assert result.intent != "library"


def test_sessions_are_isolated(manager) -> None:
    manager.handle("one", Role.STUDENT, "exam schedule for CSE")
    result = manager.handle("two", Role.STUDENT, "and for ECE?")
    assert result.resolved_by is not Resolution.FOLLOW_UP


def test_role_change_resets_session() -> None:
    store = SessionStore()
    store.get("s", Role.STUDENT).last_intent = "exam_schedule"
    assert store.get("s", Role.FACULTY).last_intent is None


def test_expired_sessions_start_fresh() -> None:
    store = SessionStore(ttl_seconds=-1)
    store.get("s", Role.STUDENT).last_intent = "exam_schedule"
    assert store.get("s", Role.STUDENT).last_intent is None


# ---------------------------------------------------------------- fallback


def test_low_confidence_gibberish_falls_back_with_suggestions(chat) -> None:
    result = chat("qwxz blorf zzkk")
    assert result.intent == "fallback"
    assert result.resolved_by is Resolution.FALLBACK
    assert result.suggestions


def test_out_of_scope_question_is_declined(chat) -> None:
    result = chat("what's the weather like today")
    assert result.intent == "fallback"
    assert "outside what I can help with" in result.reply
