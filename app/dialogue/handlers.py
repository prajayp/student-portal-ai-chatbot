"""Intent handlers: turn (role, slots) into a reply using the university dataset.

Each handler is registered with the roles allowed to use it and the slots it reads,
so the dialogue manager can enforce access and decide whether a follow-up applies.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from app.dialogue.knowledge import intent_suggestions, university
from app.dialogue.roles import ALL_ROLES, Role
from app.dialogue.state import Slots

PROGRAM_YEARS = {"BTECH": 4, "MTECH": 2, "MBA": 2}
PROGRAM_LABELS = {"BTECH": "B.Tech", "MTECH": "M.Tech", "MBA": "MBA", "PHD": "Ph.D."}
SLOT_OPTIONS: dict[str, list[str]] = {
    "department": ["CSE", "ECE", "EEE", "ME", "CE"],
    "semester": ["Semester 1", "Semester 3", "Semester 5", "Semester 7"],
    "program": list(PROGRAM_LABELS.values()),
}
ROLE_NOUNS = {Role.STUDENT: "a student", Role.FACULTY: "a faculty member", Role.ADMINISTRATOR: "an administrator"}
DEFAULT_CHIPS: dict[Role, tuple[str, ...]] = {
    Role.STUDENT: ("exam_schedule", "attendance", "library", "hostel"),
    Role.FACULTY: ("grade_submission", "leave_request", "timetable", "notices"),
    Role.ADMINISTRATOR: ("admission_stats", "post_notice", "results", "notices"),
}


@dataclass(frozen=True)
class Context:
    role: Role
    slots: Slots
    fresh: frozenset[str] = frozenset()  # slot names the user mentioned in this very turn

    @property
    def data(self) -> dict[str, Any]:
        return university()

    @property
    def profile(self) -> dict[str, Any]:
        return self.data["profiles"][self.role.value]


@dataclass
class Response:
    reply: str
    suggestions: list[str] = field(default_factory=list)
    awaiting_slot: str | None = None


Handler = Callable[[Context], Response]


@dataclass(frozen=True)
class IntentSpec:
    name: str
    title: str
    handler: Handler
    roles: frozenset[Role]
    slots: frozenset[str]


REGISTRY: dict[str, IntentSpec] = {}


def intent(name: str, title: str, roles: Iterable[Role] = ALL_ROLES, slots: Iterable[str] = ()) -> Callable[[Handler], Handler]:
    """Register a handler for ``name``."""

    def decorator(fn: Handler) -> Handler:
        REGISTRY[name] = IntentSpec(name, title, fn, frozenset(roles), frozenset(slots))
        return fn

    return decorator


# --------------------------------------------------------------------------- helpers


def chips(role: Role, *intents: str) -> list[str]:
    """Suggestion chips for the given intents (or the role defaults), skipping ones the role can't use."""
    examples = intent_suggestions(role.value)
    names = intents or DEFAULT_CHIPS[role]
    return [examples[n] for n in names if n in examples and role in REGISTRY[n].roles][:4]


def ask(slot: str, question: str) -> Response:
    """Ask a clarification question and remember which slot we are waiting for."""
    return Response(question, suggestions=SLOT_OPTIONS.get(slot, []), awaiting_slot=slot)


def fmt_date(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d:%a} {d.day} {d:%b %Y}"


def inr(amount: int) -> str:
    """Format rupees with Indian digit grouping, e.g. 225000 -> ₹2,25,000."""
    digits = str(amount)
    head, tail = digits[:-3], digits[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return "₹" + ",".join([*groups, tail]) if groups else "₹" + tail


def bullets(lines: Iterable[str]) -> str:
    return "\n".join(f"• {line}" for line in lines)


def dept_name(code: str) -> str:
    return university()["departments"][code]


def course_title(code: str) -> str:
    course = university()["courses"].get(code)
    return f"{code} {course['title']}" if course else code


def other_department(current: str) -> str:
    return next(d for d in SLOT_OPTIONS["department"] if d != current)


def role_note(section: dict[str, Any], role: Role) -> str:
    note = section.get("role_notes", {}).get(role.value)
    return f"\n\n{note}" if note else ""


# --------------------------------------------------------------------------- small talk


@intent("greeting", "greetings")
def greeting(ctx: Context) -> Response:
    uni = ctx.data["university"]["name"]
    focus = {
        Role.STUDENT: "exams, timetables, results, attendance and campus services",
        Role.FACULTY: "grade submission, leave requests, timetables and campus services",
        Role.ADMINISTRATOR: "admission statistics, notices, results and campus services",
    }[ctx.role]
    return Response(
        f"Hi {ctx.profile['name']}! I'm the {uni} assistant. As {ROLE_NOUNS[ctx.role]} you can ask me about {focus}.",
        chips(ctx.role),
    )


@intent("help", "help")
def help_(ctx: Context) -> Response:
    topics = [
        "Academics: exam schedule, class timetable, results, attendance, course details",
        "Admissions: eligibility, deadlines, fees and scholarships, documents",
        "Campus: library, hostel, transport, canteen, Wi-Fi, health centre, notices",
    ]
    if ctx.role is Role.FACULTY:
        topics.append("Faculty tools: grade submission and leave requests")
    if ctx.role is Role.ADMINISTRATOR:
        topics.append("Admin tools: admission statistics and publishing notices")
    reply = (
        "Here's what I can help with:\n" + bullets(topics)
        + "\n\nFollow-ups work too, e.g. ask about CSE exams, then just say \"and for ECE?\"."
    )
    return Response(reply, chips(ctx.role))


@intent("thanks", "thanks")
def thanks(ctx: Context) -> Response:
    return Response("You're welcome! Anything else I can help with?", chips(ctx.role))


@intent("goodbye", "goodbye")
def goodbye(ctx: Context) -> Response:
    return Response("Goodbye, and all the best this semester!")


@intent("fallback", "fallback")
def fallback(ctx: Context) -> Response:
    return Response(
        "I can only help with Example University topics such as academics, admissions and campus services. "
        "Try one of these:",
        chips(ctx.role),
    )


# --------------------------------------------------------------------------- academics


@intent("exam_schedule", "the exam schedule", slots=("department", "semester", "course_code"))
def exam_schedule(ctx: Context) -> Response:
    data, slots = ctx.data, ctx.slots
    schedule = data["exam_schedule"]
    footer = {
        Role.STUDENT: f"Hall tickets are available on the portal from {fmt_date(schedule['hall_ticket_release'])}.",
        Role.FACULTY: "Your invigilation duties are under Faculty Portal > Examinations > Invigilation.",
        Role.ADMINISTRATOR: "Exam cell: "
        + str(sum(len(e) for d in schedule["by_department"].values() for e in d.values()))
        + " papers are scheduled across 5 departments this term.",
    }[ctx.role]

    if code := slots.get("course_code"):
        course = data["courses"].get(str(code))
        if not course:
            return Response(f"I couldn't find a course with code {code}.", ["Tell me about CS501"])
        dept = course["department"] if course["department"] != "Common" else slots.get("department", "CSE")
        entries = schedule["by_department"][dept].get(str(course["semester"]), [])
        entry = next((e for e in entries if e["course_code"] == code), None)
        if not entry:
            return Response(f"{course_title(str(code))} isn't examined this term.", chips(ctx.role, "exam_schedule"))
        return Response(
            f"{course_title(str(code))} (Semester {course['semester']}): {fmt_date(entry['date'])}, {entry['session']}.\n\n{footer}",
            [f"Exam schedule for {dept} semester {course['semester']}", f"Tell me about {code}"],
        )

    dept = slots.get("department")
    if not dept:
        return ask("department", "Sure. Which department's exam schedule do you need?")
    dept = str(dept)
    by_sem = schedule["by_department"][dept]
    sem = slots.get("semester")

    if sem is not None:
        if str(sem) not in by_sem:
            return Response(
                f"Semester {sem} is examined in the even term (April-May 2027); that schedule isn't published yet. "
                f"This term's exams cover semesters {', '.join(by_sem)}.",
                [f"What about semester {s}?" for s in by_sem][:3],
            )
        lines = [f"{fmt_date(e['date'])}: {e['course_code']} {e['course']} ({e['session']})" for e in by_sem[str(sem)]]
        reply = f"{dept} semester {sem} end-semester exams:\n{bullets(lines)}\n\n{footer}"
        return Response(reply, [f"And for {other_department(dept)}?", f"Timetable for {dept} semester {sem}"])

    lines = [
        f"Semester {s}: {fmt_date(entries[0]['date'])} to {fmt_date(entries[-1]['date'])} ({len(entries)} papers)"
        for s, entries in by_sem.items()
    ]
    reply = f"{schedule['term']} for {dept_name(dept)}:\n{bullets(lines)}\n\nAsk about a semester for the full date sheet.\n{footer}"
    return Response(reply, ["What about semester 5?", "What about semester 3?", f"And for {other_department(dept)}?"])


@intent("timetable", "the class timetable", slots=("department", "semester"))
def timetable(ctx: Context) -> Response:
    slots = ctx.slots
    dept = slots.get("department")
    if not dept:
        return ask("department", "Which department's timetable would you like?")
    sem = slots.get("semester")
    if sem is None:
        return ask("semester", f"Which semester of {dept}?")
    dept = str(dept)
    week = ctx.data["timetable"][dept].get(str(sem))
    if not week:
        return Response(
            f"Semester {sem} runs in the even term, so there's no {dept} timetable for it right now. "
            f"Timetables are available for semesters {', '.join(map(str, ctx.data['offered_semesters']))}.",
            SLOT_OPTIONS["semester"],
        )
    lines = []
    for day, entries in week.items():
        parts = [f"{e['time'].split('-')[0]} {e['course_code']}" + (" lab" if "Lab" in e["time"] else "") for e in entries]
        lines.append(f"{day[:3]}: " + " · ".join(parts))
    codes = sorted({e["course_code"] for entries in week.values() for e in entries})
    reply = f"{dept} semester {sem} weekly timetable:\n" + "\n".join(lines) + "\n\nCourses: " + ", ".join(map(course_title, codes))
    if ctx.role is Role.FACULTY:
        mine = [c for c in codes if c in ctx.profile["courses"]]
        if mine:
            reply += f"\n\nYou teach {', '.join(mine)} in this timetable."
    elif ctx.role is Role.ADMINISTRATOR:
        reply += "\n\nRoom allocation changes go through the Timetable Office (Admin Console > Timetable)."
    return Response(reply, [f"And for {other_department(dept)}?", f"Exam schedule for {dept} semester {sem}"])


@intent("results", "results", slots=("department", "semester"))
def results(ctx: Context) -> Response:
    res = ctx.data["results"]
    expected = fmt_date(res["odd_semester_expected"])
    reval = res["revaluation"]
    sem = ctx.slots.get("semester")

    if ctx.role is Role.STUDENT:
        p = ctx.profile
        if sem is not None:
            published = res["published"].get(str(sem))
            status = (
                f"Semester {sem} results were published on {fmt_date(published)}."
                if published
                else f"Semester {sem} results are expected around {expected}."
            )
        else:
            status = f"Odd-semester results (November-December 2026 exams) are expected around {expected}."
        sgpa = ", ".join(f"Sem {s}: {v}" for s, v in p["last_sgpa"].items())
        reply = (
            f"{status}\n\nYour record ({p['id']}): CGPA {p['cgpa']} · SGPA {sgpa}.\n"
            f"Revaluation: {reval['process']} Fee {inr(reval['fee_per_course_inr'])} per course."
        )
        return Response(reply, chips(ctx.role, "attendance", "exam_schedule"))

    pass_pct: dict[str, float] = res["pass_percentage"]
    if ctx.role is Role.FACULTY:
        dept = str(ctx.slots.get("department", ctx.profile["department"]))
        reply = (
            f"Last term's pass percentage for {dept}: {pass_pct[dept]}%.\n"
            "Course-wise analysis for your classes is under Faculty Portal > Assessments > Result Analysis.\n"
            f"This term's results are expected around {expected}."
        )
        return Response(reply, chips(ctx.role, "grade_submission", "attendance"))

    dept = ctx.slots.get("department")
    rows = {str(dept): pass_pct[str(dept)]} if dept else pass_pct
    lines = [f"{d}: {v}%" for d, v in sorted(rows.items(), key=lambda kv: -kv[1])]
    reply = f"Last term's pass percentage by department:\n{bullets(lines)}\n\nOdd-semester results are due around {expected}."
    if not dept:
        lowest = min(pass_pct, key=pass_pct.__getitem__)
        reply += f" {lowest} has the lowest pass rate and may need a review."
    return Response(reply, chips(ctx.role, "attendance", "admission_stats"))


@intent("attendance", "attendance", slots=("department", "course_code"))
def attendance(ctx: Context) -> Response:
    policy = ctx.data["attendance_policy"]
    minimum = policy["minimum_percent"]

    if ctx.role is Role.STUDENT:
        records: dict[str, float] = ctx.profile["attendance"]
        code = ctx.slots.get("course_code") if "course_code" in ctx.fresh else None
        if code and code in records:
            records = {str(code): records[str(code)]}
        lines = [f"{course_title(c)}: {v}%" + (f" (below {minimum}%)" if v < minimum else "") for c, v in records.items()]
        reply = (
            f"Your attendance this semester:\n{bullets(lines)}\n\n"
            f"You need at least {minimum}% in each course to sit the end-semester exam. {policy['condonation_note']}"
        )
        return Response(reply, chips(ctx.role, "exam_schedule", "medical"))

    if ctx.role is Role.FACULTY:
        lines = [f"{course_title(c)}: class average {v}%" for c, v in ctx.profile["class_attendance"].items()]
        reply = (
            f"Attendance in your courses:\n{bullets(lines)}\n\n"
            "Mark attendance under Faculty Portal > Classes > Attendance (editable for 48 hours after each class). "
            f"Students below {minimum}% are flagged automatically."
        )
        return Response(reply, chips(ctx.role, "grade_submission", "timetable"))

    dept = ctx.slots.get("department")
    depts = [str(dept)] if dept else list(policy["department_average"])
    lines = [
        f"{d}: average {policy['department_average'][d]}%, {policy['students_below_minimum'][d]} students below {minimum}%"
        for d in depts
    ]
    return Response("Attendance overview:\n" + bullets(lines), chips(ctx.role, "results", "admission_stats"))


@intent("course_details", "course details", slots=("department", "semester", "course_code"))
def course_details(ctx: Context) -> Response:
    data, slots = ctx.data, ctx.slots
    if code := slots.get("course_code"):
        course = data["courses"].get(str(code))
        if not course:
            return Response(f"I couldn't find a course with code {code}. Codes look like CS501 or EC302.")
        prereq = ", ".join(map(course_title, course["prerequisites"])) or "none"
        owner = "all first-year students" if course["department"] == "Common" else dept_name(course["department"])
        reply = (
            f"{course['code']} {course['title']} ({course['credits']} credits, semester {course['semester']}, {owner})\n"
            f"{course['description']}\n"
            f"Prerequisites: {prereq}. Lab component: {'yes' if course['has_lab'] else 'no'}."
        )
        if ctx.role is Role.FACULTY and code in ctx.profile["courses"]:
            reply += "\n\nYou're teaching this course this term."
        return Response(reply, [f"When is the {code} exam?", "What about semester 7?"])

    dept = slots.get("department")
    if not dept:
        return ask("department", "Which department's courses are you interested in? You can also give a course code like CS501.")
    sem = slots.get("semester")
    if sem is None:
        return ask("semester", f"Which semester of {dept}?")
    courses = [
        c for c in data["courses"].values()
        if c["semester"] == sem and c["department"] in (dept, "Common")
    ]
    if not courses:
        return Response(f"I don't have {dept} semester {sem} courses yet; this term covers semesters 1, 3, 5 and 7.",
                        SLOT_OPTIONS["semester"])
    lines = [f"{c['code']} {c['title']} ({c['credits']} credits)" for c in courses]
    return Response(f"{dept} semester {sem} courses:\n{bullets(lines)}",
                    [f"Tell me about {courses[0]['code']}", f"And for {other_department(str(dept))}?"])


# --------------------------------------------------------------------------- admissions


def _program_overview(ctx: Context, describe: Callable[[str, dict[str, Any]], str], heading: str) -> Response:
    programs = ctx.data["admissions"]["programs"]
    lines = [f"{PROGRAM_LABELS[k]}: {describe(k, p)}" for k, p in programs.items()]
    return Response(f"{heading}\n{bullets(lines)}\n\nPick a programme for details.", SLOT_OPTIONS["program"])


def _applications_note(ctx: Context, program: str) -> str:
    if ctx.role is not Role.ADMINISTRATOR:
        return ""
    count = ctx.data["admissions"]["statistics"]["applications"][program]
    return f"\n\nApplications received so far: {count:,}."


@intent("admission_eligibility", "admission eligibility", slots=("program",))
def admission_eligibility(ctx: Context) -> Response:
    program = ctx.slots.get("program")
    if not program:
        return _program_overview(ctx, lambda _k, p: p["eligibility"].split(";")[0].split(",")[0], "Eligibility at a glance:")
    p = ctx.data["admissions"]["programs"][str(program)]
    reply = f"{p['name']} eligibility: {p['eligibility']}\nSelection: {p['entrance']}." + _applications_note(ctx, str(program))
    return Response(reply, chips(ctx.role, "admission_deadlines", "admission_fees", "admission_documents"))


@intent("admission_deadlines", "admission deadlines", slots=("program",))
def admission_deadlines(ctx: Context) -> Response:
    today = date.today()

    def status(p: dict[str, Any]) -> str:
        opens, closes = date.fromisoformat(p["opens"]), date.fromisoformat(p["deadline"])
        if today < opens:
            return f"opens {fmt_date(p['opens'])}, closes {fmt_date(p['deadline'])}"
        if today <= closes:
            return f"open now, closes {fmt_date(p['deadline'])}"
        return f"closed on {fmt_date(p['deadline'])}"

    program = ctx.slots.get("program")
    cycle = ctx.data["admissions"]["cycle"]
    if not program:
        return _program_overview(ctx, lambda _k, p: status(p), f"Admission windows for {cycle}:")
    p = ctx.data["admissions"]["programs"][str(program)]
    reply = f"{p['name']} admissions {cycle}: applications {status(p)}.\nSelection: {p['entrance']}." + _applications_note(ctx, str(program))
    return Response(reply, chips(ctx.role, "admission_eligibility", "admission_fees", "admission_documents"))


@intent("admission_fees", "fees", slots=("program",))
def admission_fees(ctx: Context) -> Response:
    adm = ctx.data["admissions"]
    program = ctx.slots.get("program")
    if not program:
        return _program_overview(ctx, lambda _k, p: f"{inr(p['tuition_per_year_inr'])} per year", "Tuition fees:")
    p = adm["programs"][str(program)]
    lines = [f"Tuition: {inr(p['tuition_per_year_inr'])} per year"]
    if years := PROGRAM_YEARS.get(str(program)):
        lines.append(f"Total tuition ({years} years): {inr(p['tuition_per_year_inr'] * years)}")
    lines += [f"Application fee: {inr(p['application_fee_inr'])}", adm["installments"]]
    reply = f"{p['name']} fees:\n{bullets(lines)}"
    if ctx.role is Role.FACULTY:
        reply += f"\n\n{adm['staff_ward_concession']}"
    else:
        reply += "\n\nScholarships:\n" + bullets(adm["scholarships"])
    reply += _applications_note(ctx, str(program))
    return Response(reply, chips(ctx.role, "admission_eligibility", "admission_deadlines", "hostel"))


@intent("admission_documents", "admission documents", slots=("program",))
def admission_documents(ctx: Context) -> Response:
    docs = ctx.data["admissions"]["documents"]
    reply = "Bring originals and one self-attested copy of:\n" + bullets(docs)
    if ctx.role is Role.ADMINISTRATOR:
        reply += "\n\nVerification is handled by the Admissions Office, Block A, and logged under Admin Console > Admissions."
    return Response(reply, chips(ctx.role, "admission_deadlines", "admission_fees"))


# --------------------------------------------------------------------------- campus services


@intent("library", "library services")
def library(ctx: Context) -> Response:
    lib = ctx.data["campus_services"]["library"]
    role = ctx.role.value
    lines = [
        f"Timings: {lib['timings']}",
        f"Location: {lib['location']}",
        f"You can borrow {lib['borrow_limit'][role]} books for {lib['loan_days'][role]} days; late fine {inr(lib['fine_per_day_inr'])}/day",
        f"E-resources: {lib['eresources']}",
    ]
    return Response("Central Library:\n" + bullets(lines), chips(ctx.role, "wifi", "canteen"))


@intent("hostel", "hostel information")
def hostel(ctx: Context) -> Response:
    h = ctx.data["campus_services"]["hostel"]
    lines = [h["capacity"], h["rooms"], f"Fee: {inr(h['fee_per_year_inr'])} per year ({h['fee_includes']})",
             f"In-time: {h['in_time']}", h["apply"], f"Contact: {h['warden_office']}"]
    if ctx.role is not Role.STUDENT:
        lines = lines[:3]
    return Response("Hostels:\n" + bullets(lines) + role_note(h, ctx.role), chips(ctx.role, "canteen", "transport"))


@intent("transport", "transport")
def transport(ctx: Context) -> Response:
    t = ctx.data["campus_services"]["transport"]
    routes = [f"{r['route']}: {r['via']} (first pickup {r['first_pickup']})" for r in t["routes"]]
    reply = (
        "College bus routes:\n" + bullets(routes)
        + f"\n\nDepartures from campus: {', '.join(t['departures_from_campus'])}."
        + f"\nFee: {inr(t['fee_per_year_inr'])} per year. {t['pass']}\nContact: {t['office']}"
        + role_note(t, ctx.role)
    )
    return Response(reply, chips(ctx.role, "hostel", "canteen"))


@intent("canteen", "canteen information")
def canteen(ctx: Context) -> Response:
    c = ctx.data["campus_services"]["canteen"]
    lines = [f"{o['name']}: {o['timings']}" for o in c["outlets"]]
    return Response("Food on campus:\n" + bullets(lines) + f"\n\n{c['notes']}", chips(ctx.role, "library", "hostel"))


@intent("wifi", "Wi-Fi support")
def wifi(ctx: Context) -> Response:
    w = ctx.data["campus_services"]["wifi"]
    lines = [
        f"Network: {w['ssid']} (guests: {w['guest_ssid']}){'; eduroam is also available' if w['eduroam'] else ''}",
        w["login"],
        f"Up to {w['devices'][ctx.role.value]} devices per {ctx.role.value} account",
        w["reset"],
        f"Still stuck? {w['helpdesk']}",
    ]
    return Response("Campus Wi-Fi:\n" + bullets(lines), chips(ctx.role, "library", "notices"))


@intent("medical", "medical services")
def medical(ctx: Context) -> Response:
    m = ctx.data["campus_services"]["medical"]
    lines = [f"Emergency: {m['emergency']}", f"{m['centre']}: {m['timings']}", m["counselling"], f"Hospital: {m['hospital']}"]
    if ctx.role is Role.STUDENT:
        lines.append(m["insurance"])
    return Response("Health and wellbeing:\n" + bullets(lines) + role_note(m, ctx.role), chips(ctx.role, "hostel", "notices"))


@intent("notices", "notices")
def notices(ctx: Context) -> Response:
    audience = {Role.STUDENT: "Students", Role.FACULTY: "Faculty"}.get(ctx.role)
    items = [n for n in ctx.data["notices"] if ctx.role is Role.ADMINISTRATOR or n["audience"] in ("All", audience)]
    lines = [f"{fmt_date(n['date'])}: {n['title']}" + (f" [{n['audience']}]" if ctx.role is Role.ADMINISTRATOR else "") for n in items]
    reply = "Latest notices:\n" + bullets(lines)
    if ctx.role is Role.ADMINISTRATOR:
        reply += f"\n\n{ctx.data['academic_admin']['notice_publishing']['pending_approval']} notices are awaiting approval."
    return Response(reply, chips(ctx.role, "post_notice", "exam_schedule", "library"))


# --------------------------------------------------------------------------- role-restricted


@intent("grade_submission", "grade submission", roles=(Role.FACULTY,), slots=("course_code",))
def grade_submission(ctx: Context) -> Response:
    g = ctx.data["academic_admin"]["grade_submission"]
    mine: list[str] = ctx.profile["courses"]
    code = ctx.slots.get("course_code")
    if code and code not in mine:
        scope = f"{code} isn't one of your assigned courses ({', '.join(mine)}), so you can't enter grades for it."
    elif code:
        scope = f"Submitting grades for {course_title(str(code))}."
    else:
        scope = f"Your courses this term: {', '.join(map(course_title, mine))}."
    lines = [
        f"Internal marks deadline: {fmt_date(g['internal_marks_deadline'])}",
        f"Final grades deadline: {fmt_date(g['final_grades_deadline'])}",
        f"Where: {g['portal_path']}",
        g["edit_policy"],
    ]
    return Response(f"{scope}\n{bullets(lines)}", ["Submit grades for CS501", "How do I apply for leave?"])


@intent("leave_request", "leave requests", roles=(Role.FACULTY,))
def leave_request(ctx: Context) -> Response:
    policy = ctx.data["academic_admin"]["leave_policy"]
    balance = ", ".join(f"{k} {v}" for k, v in ctx.profile["leave_balance"].items())
    lines = [f"Your balance: {balance} days", f"Apply: {policy['apply_path']}", f"Approval: {policy['approver']}", policy["notes"]]
    return Response("Leave requests:\n" + bullets(lines), chips(ctx.role, "grade_submission", "timetable"))


@intent("admission_stats", "admission statistics", roles=(Role.ADMINISTRATOR,), slots=("program", "department"))
def admission_stats(ctx: Context) -> Response:
    stats = ctx.data["admissions"]["statistics"]
    program, dept = ctx.slots.get("program"), ctx.slots.get("department")
    apps, prev = stats["applications"], stats["previous_cycle_applications"]
    programs = [str(program)] if program else list(apps)
    app_lines = [f"{PROGRAM_LABELS[p]}: {apps[p]:,} applications ({(apps[p] - prev[p]) / prev[p]:+.1%} vs last cycle)" for p in programs]
    depts = [str(dept)] if dept else list(stats["seats"])
    seat_lines = [f"{d}: {stats['filled_last_cycle'][d]}/{stats['seats'][d]} B.Tech seats filled last cycle" for d in depts]
    reply = (
        f"Admission statistics as of {fmt_date(stats['as_of'])}:\n{bullets(app_lines)}\n\nIntake:\n{bullets(seat_lines)}"
    )
    return Response(reply, ["And for MBA?", "Show the latest notices", "Seats filled in ECE"])


@intent("post_notice", "publishing notices", roles=(Role.ADMINISTRATOR,))
def post_notice(ctx: Context) -> Response:
    n = ctx.data["academic_admin"]["notice_publishing"]
    lines = [f"Go to {n['path']}", f"Choose an audience: {', '.join(n['audiences'])}", n["approval"],
             f"{n['pending_approval']} notices are currently awaiting approval"]
    return Response("Publishing a notice:\n" + bullets(lines), chips(ctx.role, "notices", "admission_stats"))
