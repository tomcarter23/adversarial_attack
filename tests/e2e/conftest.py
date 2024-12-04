import pytest

# Global variables to track the test results
total_tests = 0
passed_tests = 0


def pytest_runtest_logreport(report):
    global total_tests, passed_tests
    if report.when == "call":  # We only care about the actual test call phase
        total_tests += 1
        if report.outcome == "passed":
            passed_tests += 1


def pytest_sessionfinish(session, exitstatus):
    global total_tests, passed_tests
    if total_tests > 0:
        pass_rate = (passed_tests / total_tests) * 100
        print(f"\nTest Pass Rate: {pass_rate:.2f}%")
        if pass_rate < 70:
            session.exitstatus = 1  # Fail the test run if the pass rate is less than 70%
