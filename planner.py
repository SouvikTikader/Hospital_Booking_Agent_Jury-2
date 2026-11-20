"""LLM PLANNER (ReAct loop): the model decides the next action from observations; Python validates every call.

Actions the model may choose:
  contract_net(day, after) - negotiate with all doctor agents (Unit 4 tool)
  propose                  - stop and present the best proposal to the patient
  give_up                  - nothing suitable, go to waitlist
Hard guardrails enforced in code (the LLM cannot override them):
  * specialty is fixed by the intent, never chosen by the model
  * time is never earlier than the patient asked for (clamped)
  * the day may move later by at most MAX_DAYS_AFTER days, and every relaxation is disclosed + needs approval
  * a repeated call is rejected; any invalid output falls back to a deterministic plan
Decision tiers: native tool calling (LangChain bind_tools) -> JSON output -> deterministic fallback
"""
import json
import re
from datetime import date, timedelta

import agents
import llm

try:  # LangChain tool schemas (the model only *chooses* them; Python validates and executes the real work)
    from langchain_core.tools import tool

    @tool
    def contract_net(day: str, after: str) -> str:
        """Ask every doctor agent for its earliest free slot on `day` (YYYY-MM-DD) after `after` (HH:MM, 24h)."""
        return ""

    @tool
    def propose() -> str:
        """Stop searching and present the best proposal to the patient for approval."""
        return ""

    @tool
    def give_up() -> str:
        """Stop because no suitable slot exists in the allowed range; add the patient to the waitlist."""
        return ""

    TOOLS = [contract_net, propose, give_up]
except ImportError:
    TOOLS = None

MAX_STEPS = 5
MAX_DAYS_AFTER = 2

TOOL_SYSTEM = """You are a hospital scheduling planner. Call exactly ONE tool now.
contract_net asks every doctor agent for its earliest free slot on a day after a time.
Rules: first call contract_net with the patient's requested day and time. If the history shows proposals, call propose.
If the history shows no proposals, call contract_net for the next day with the same time. Never use an earlier time
than requested. Never go more than 2 days past the requested day. If nothing works, call give_up."""

SYSTEM = """You are a hospital scheduling planner. Reply with ONE JSON object only, no other text.
Options:
{"thought": "<short reason>", "action": "contract_net", "day": "YYYY-MM-DD", "after": "HH:MM"}
{"thought": "<short reason>", "action": "propose"}
{"thought": "<short reason>", "action": "give_up"}
contract_net asks every doctor agent for its earliest free slot on that day after that time.
Rules: first call contract_net with the patient's requested day and time. If it returns proposals, choose propose.
If it returns no proposals, try the next day with the same time. Never use an earlier time than requested.
Never go more than 2 days past the requested day. If nothing works, choose give_up."""


def _fallback(st):
    if not st["tried"]:
        return {"action": "contract_net", "day": st["req_day"], "after": st["req_after"],
                "thought": "ask all doctor agents for proposals"}
    if st["bids"]:
        return {"action": "propose", "thought": "proposals are available"}
    nxt = date.fromisoformat(st["tried"][-1][0]) + timedelta(days=1)
    if (nxt - date.fromisoformat(st["req_day"])).days <= MAX_DAYS_AFTER:
        return {"action": "contract_net", "day": nxt.isoformat(), "after": st["req_after"],
                "thought": "no proposals, try the next day"}
    return {"action": "give_up", "thought": "no suitable slot within the allowed range"}


def _validate(d, st):
    """Return a safe decision dict or None."""
    if not isinstance(d, dict):
        return None
    act = d.get("action")
    thought = str(d.get("thought", ""))[:160]
    if act == "propose" and st["bids"]:
        return {"action": "propose", "thought": thought}
    if act == "give_up" and not st["bids"]:
        return {"action": "give_up", "thought": thought}
    if act == "contract_net":
        try:
            day = date.fromisoformat(str(d.get("day")))
        except ValueError:
            return None
        gap = (day - date.fromisoformat(st["req_day"])).days
        if gap < 0 or gap > MAX_DAYS_AFTER:
            return None
        after = str(d.get("after") or "")
        if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", after):
            return None
        after = max(after, st["req_after"])  # guardrail: never earlier than requested
        if (day.isoformat(), after) in st["tried"]:
            return None
        return {"action": "contract_net", "day": day.isoformat(), "after": after, "thought": thought}
    return None


def _decide(st):
    hist = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(st["history"])) or "(none yet)"
    user = (f"Today is {date.today().isoformat()}. Patient wants a {st['specialty']} appointment on "
            f"{st['req_day']} after {st['req_after']}.\nHistory:\n{hist}")
    if TOOLS:  # tier 1: native tool calling through LangChain bind_tools
        res = llm.tool_call(TOOL_SYSTEM, user, TOOLS)
        if res and res[1]:
            c = res[1][0]
            d = _validate({**(c.get("args") or {}), "action": c.get("name"), "thought": res[0].strip()}, st)
            if d:
                d["thought"] = d["thought"] or f"model chose {c.get('name')}"
                return d, "LLM-tools"
    raw = llm.chat(SYSTEM, user, json_mode=True)  # tier 2: JSON output
    if raw:
        try:
            d = _validate(json.loads(raw), st)
            if d:
                return d, "LLM"
        except ValueError:
            pass
    return _fallback(st), "fallback"


def run_planner(intent, patient_id, window=60):
    st = {"specialty": intent["specialty"], "req_day": intent["day"], "req_after": intent["after"],
          "tried": [], "history": [], "bids": []}
    trace, acl, found = [], [], None
    for step in range(1, MAX_STEPS + 1):
        d, src = _decide(st)
        if d["action"] == "contract_net":
            trace.append(f"PLANNER {step} [{src}]: {d['thought']} -> contract_net({d['day']}, {d['after']})")
            bids, log = agents.ManagerAgent(window).contract_net(st["specialty"], d["day"], d["after"], patient_id)
            acl += log
            st["tried"].append((d["day"], d["after"]))
            if bids:
                st["bids"] = bids
            if bids:
                found = {"bids": bids, "day": d["day"], "after": d["after"]}
                obs = f"{len(bids)} proposals, best {bids[0]['doctor']} at {bids[0]['time']}"
            else:
                obs = "no proposals"
            st["history"].append(f"contract_net({d['day']}, {d['after']}) -> {obs}")
            trace.append(f"OBSERVATION: {obs}")
        elif d["action"] == "propose":
            trace.append(f"PLANNER {step} [{src}]: {d['thought']} -> propose")
            break
        else:
            trace.append(f"PLANNER {step} [{src}]: {d['thought']} -> give_up")
            found = None
            break
    if found is None:
        return {"bids": [], "day": intent["day"], "after": intent["after"], "trace": trace, "acl": acl}
    return {"bids": found["bids"], "day": found["day"], "after": found["after"], "trace": trace, "acl": acl}
