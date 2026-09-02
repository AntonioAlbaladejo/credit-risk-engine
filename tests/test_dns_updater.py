"""Tests for the DuckDNS updater Lambda.

The function is deployed by hand to AWS, so nothing here proves the deployed
copy matches this file. What it does protect is the logic that fails quietly:
acting on the wrong task state would publish an address that is about to
disappear, and DuckDNS answers 200 to a rejected update, so a missing check on
the body would read as success forever.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "infra" / "dns_updater"))
import lambda_function  # noqa: E402


def event(last="RUNNING", desired="RUNNING"):
    """An ECS Task State Change event, shaped as EventBridge delivers it."""
    return {
        "detail": {
            "lastStatus": last,
            "desiredStatus": desired,
            "taskArn": "arn:task/1",
            "attachments": [
                {
                    "type": "eni",
                    "details": [{"name": "networkInterfaceId", "value": "eni-123"}],
                }
            ],
        }
    }


@pytest.fixture
def invoke(monkeypatch):
    """Run the handler against a stubbed EC2 and DuckDNS, and report the call."""
    monkeypatch.setenv("DUCKDNS_DOMAIN", "credit-risk-engine")
    monkeypatch.setenv("DUCKDNS_TOKEN", "secret-token")
    called = []

    def run(task_event, body="OK", public_ip="1.2.3.4"):
        association = {"Association": {"PublicIp": public_ip}} if public_ip else {}
        boto3 = MagicMock()
        boto3.client.return_value.describe_network_interfaces.return_value = {
            "NetworkInterfaces": [association]
        }

        response = MagicMock()
        response.read.return_value = body.encode()
        response.__enter__.return_value = response

        def urlopen(url, timeout=None):
            called.append(url)
            return response

        monkeypatch.setattr(lambda_function.urllib.request, "urlopen", urlopen)
        with patch.dict(sys.modules, {"boto3": boto3}):
            return lambda_function.lambda_handler(task_event, None)

    return run, called


class TestDnsUpdater:
    """Test the DuckDNS updater"""

    def test_a_running_task_publishes_its_address(self, invoke):
        """The whole point: a new task means a new address to point at"""
        run, called = invoke
        assert run(event()) == {"updated": "1.2.3.4"}
        assert "ip=1.2.3.4" in called[0]
        assert "domains=credit-risk-engine" in called[0]

    @pytest.mark.parametrize(
        "last,desired", [("PENDING", "RUNNING"), ("RUNNING", "STOPPED")]
    )
    def test_a_task_not_both_up_and_staying_up_is_ignored(self, invoke, last, desired):
        """A task on its way in or out has no address worth publishing"""
        run, called = invoke
        assert run(event(last=last, desired=desired)) == {"skipped": last}
        assert called == []

    def test_a_task_with_no_public_address_is_ignored(self, invoke):
        """A task in a private subnet must not blank the record"""
        run, called = invoke
        assert run(event(), public_ip=None) == {"skipped": "no public ip"}
        assert called == []

    def test_a_refused_update_raises_rather_than_reporting_success(self, invoke):
        """DuckDNS answers 200 to a rejection, so only the body says so"""
        run, _ = invoke
        with pytest.raises(RuntimeError, match="refused"):
            run(event(), body="KO")
