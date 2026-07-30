"""Tests for the deployed service. Run with: python test_app.py (or pytest)."""
from __future__ import annotations

import json

from app import app

HIGH_EARNER = {
    "age": 45, "education-num": 16, "hours-per-week": 60,
    "capital-gain": 15000, "capital-loss": 0,
    "workclass": "Private", "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial", "relationship": "Husband", "sex": "Male",
}
LOW_EARNER = {
    "age": 20, "education-num": 7, "hours-per-week": 20,
    "capital-gain": 0, "capital-loss": 0,
    "workclass": "Private", "marital-status": "Never-married",
    "occupation": "Other-service", "relationship": "Own-child", "sex": "Female",
}


def run_tests() -> None:
    client = app.test_client()
    passed = failed = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  PASS  {name}")
        else:
            failed += 1
            print(f"  FAIL  {name} {detail}")

    print("Health endpoint")
    response = client.get("/health")
    body = response.get_json()
    check("returns 200", response.status_code == 200)
    check("reports ok", body.get("status") == "ok", str(body))
    check("exposes model metrics", "roc_auc" in body, str(body))

    print("HTML form")
    response = client.get("/")
    check("form renders", response.status_code == 200)
    check("contains the age field", b'name="age"' in response.data)
    response = client.post("/", data=HIGH_EARNER)
    check("form POST renders a prediction", response.status_code == 200
          and (b">50K" in response.data or b"&gt;50K" in response.data))

    print("JSON API")
    response = client.post("/api/predict", json=HIGH_EARNER)
    high = response.get_json()
    check("high earner -> 200", response.status_code == 200, str(high))
    check("high earner -> >50K", high.get("prediction") == ">50K", str(high))

    response = client.post("/api/predict", json=LOW_EARNER)
    low = response.get_json()
    check("low earner -> <=50K", low.get("prediction") == "<=50K", str(low))
    check("high earner scores above low earner",
          high["probability_above_50k"] > low["probability_above_50k"],
          f'{high["probability_above_50k"]} vs {low["probability_above_50k"]}')

    print("Input validation")
    response = client.post("/api/predict", json={"age": 30})
    check("missing fields -> 400", response.status_code == 400)
    check("error names the missing fields",
          "Missing required field" in response.get_json().get("error", ""))

    bad = dict(HIGH_EARNER, age="not-a-number")
    response = client.post("/api/predict", json=bad)
    check("non-numeric age -> 400", response.status_code == 400)

    response = client.post("/api/predict", data="raw", content_type="text/plain")
    check("non-JSON body -> 400", response.status_code == 400)

    print()
    print(f"{passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)


def test_service() -> None:
    """Entry point for pytest, which collects `test_*` functions only.

    Without this wrapper `pytest test_app.py` exits with code 5 (no tests
    collected) and silently looks like a pass in CI.
    """
    run_tests()


if __name__ == "__main__":
    run_tests()
