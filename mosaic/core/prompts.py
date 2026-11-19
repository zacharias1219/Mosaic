"""Prompt templates. System prompts are fixed strings so every call shares an identical,
cacheable prefix and the variable part (chunk, question) comes last (PRD 7)."""
from __future__ import annotations

import json

LOG_FORMAT = ("Each log line is one support ticket: `ticket_id | timestamp (UTC, 2026) | customer_id | "
              "product | free text`. Labels such as the ticket's category, sentiment or whether a refund "
              "happened are NOT in the log; they must be inferred from the free text.")

FIELD_GUIDE = """Field meanings for map_schema types:
- bool fields named like is_refund: true only when the customer asks for, or was promised, money returned for this ticket (any wording: refund, money back, reimburse, reverse the charge, credit back). False if they say they do not want one or only mention a price.
- amount: the refund amount in dollars as a number, or null when there is no refund.
- category: exactly one of billing (charges, invoices, payments, prices), shipping (delivery, couriers, parcels, tracking), defect (the device or service malfunctions), account (login, password, profile, users, access), other (questions, requests, feedback).
- sentiment: exactly one of positive, neutral, negative (the customer's tone)."""

PLAN_SYSTEM = f"""Task type: plan.
You write query plans for answering questions over a long support log. You never write code: you fill a JSON plan that a program executes.
{LOG_FORMAT}

Plan JSON keys (no others):
- strategy: "direct" (one model call reads the filtered tickets and answers; best for questions about one ticket) or "recurse" (split tickets into chunks, a model extracts facts per ticket, code combines them; best for counts, sums and comparisons over many tickets).
- filter: {{"keywords": [..], "fields": {{..}}, "semantic": true|false}}. fields may use only: product (exact product name), months (list of month numbers 1-12), customer_id, ticket_id, date (YYYY-MM-DD). Field filters run in code on the log header and are exact. keywords are only trusted when semantic=false; with semantic=true the mapper classifies every ticket, which is safer when the text paraphrases (e.g. "money back" instead of "refund").
- chunk_tokens: tickets per map call measured in tokens (use 3000).
- map_task: one sentence telling the mapper what to extract.
- map_schema: {{field_name: type}} with types bool, amount, category, sentiment, string. ticket_id, ts, customer_id and product are attached automatically; do not list them.
- reduce: "count" (count rows matching reduce_args.where), "sum" (sum reduce_args.field over rows matching where), "set_union" (collect reduce_args.field), "pair_join" (customers with two different tickets where field==a and field==b within window_days; output "count" or "ids"), or "llm" (only if nothing else works).
- reduce_args: arguments for the reducer, e.g. {{"where": {{"is_refund": true}}}}.
{FIELD_GUIDE}

Example 1. Question: "How many tickets about Cam Mini asked for a refund in May?"
{{"strategy": "recurse", "filter": {{"keywords": ["refund", "money back", "reimburse"], "fields": {{"product": "Cam Mini", "months": [5]}}, "semantic": true}}, "chunk_tokens": 3000, "map_task": "Decide whether each ticket asks for or was promised a refund.", "map_schema": {{"is_refund": "bool"}}, "reduce": "count", "reduce_args": {{"where": {{"is_refund": true}}}}}}

Example 2. Question: "Which customers filed both a shipping complaint and a login and profile access complaint within 9 days of each other?"
{{"strategy": "recurse", "filter": {{"keywords": [], "fields": {{}}, "semantic": true}}, "chunk_tokens": 3000, "map_task": "Classify the category of each ticket.", "map_schema": {{"category": "category"}}, "reduce": "pair_join", "reduce_args": {{"field": "category", "a": "shipping", "b": "account", "window_days": 9, "output": "ids"}}}}

Example 3. Question: "What refund amount did ticket T-01234 receive?"
{{"strategy": "direct", "filter": {{"keywords": [], "fields": {{"ticket_id": "T-01234"}}, "semantic": true}}, "chunk_tokens": 3000, "map_task": "", "map_schema": {{}}, "reduce": "count", "reduce_args": {{}}}}

Return only the plan JSON."""

MAP_SYSTEM = f"""Task type: map.
You extract structured facts from support tickets. {LOG_FORMAT}
{FIELD_GUIDE}
Return only JSON: {{"rows": [{{"ticket_id": "...", <one key per schema field>}}, ...]}} with exactly one row per ticket, in the order given."""

DIRECT_SYSTEM = f"""Task type: direct.
You answer a question about a support log using only the log. {LOG_FORMAT}
{FIELD_GUIDE}
Return only JSON with exactly one key: "count" (integer), "amount" (number), "ticket_id" (string) or "customer_ids" (sorted list of strings), as the instructions say."""

AGG_SYSTEM = """Task type: aggregate.
You combine per-ticket facts extracted from a support log into one answer.
Return only JSON with exactly one key: "count", "amount", "ticket_id" or "customer_ids", as the instructions say."""

ANSWER_HINT = {"count": 'Answer as {"count": <integer>}.', "amount": 'Answer as {"amount": <number>}.',
               "ticket_id": 'Answer as {"ticket_id": "<id>"}.',
               "customer_ids": 'Answer as {"customer_ids": [<ids>]}.'}


def plan_messages(question: str, sample: str) -> list[dict]:
    user = f"QUESTION: {question}\nLOG SAMPLE (first lines):\n{sample}" if sample else f"QUESTION: {question}"
    return [{"role": "system", "content": PLAN_SYSTEM}, {"role": "user", "content": user}]


def map_messages(task: str, schema: dict, lines: list[str]) -> list[dict]:
    user = f"TASK: {task}\nSCHEMA: {json.dumps(schema, sort_keys=True)}\nTICKETS:\n" + "\n".join(lines)
    return [{"role": "system", "content": MAP_SYSTEM}, {"role": "user", "content": user}]


def direct_messages(question: str, answer_key: str, lines: list[str]) -> list[dict]:
    user = f"LOG:\n" + "\n".join(lines) + f"\n\nQUESTION: {question}\n{ANSWER_HINT[answer_key]}"
    return [{"role": "system", "content": DIRECT_SYSTEM}, {"role": "user", "content": user}]


def agg_messages(question: str, answer_key: str, rows: list[dict]) -> list[dict]:
    user = (f"FACTS:\n{json.dumps(rows, sort_keys=True)}\n\nQUESTION: {question}\n{ANSWER_HINT[answer_key]}")
    return [{"role": "system", "content": AGG_SYSTEM}, {"role": "user", "content": user}]
