"""Point a DuckDNS name at the public IP of whichever ECS task is running.

A Fargate task gets a fresh public IP every time it is replaced, so the
address of the demo moves under a deployment and under any replacement ECS
decides on its own. An EventBridge rule on `ECS Task State Change` calls this
on every transition, which covers both -- a step in the deployment pipeline
would only ever see the first.

Deployed by hand from the AWS console, so this file is the reviewable copy
rather than the deployed one; they drift the moment somebody edits the
function in place. Worth wiring into CD or Terraform only if it ever changes
more than once a year. See infra/dns_updater/README.md for the setup.

Environment:
    DUCKDNS_DOMAIN: the subdomain, without the ".duckdns.org" suffix.
    DUCKDNS_TOKEN: the DuckDNS account token. Never committed.
"""

import logging
import os
import urllib.parse
import urllib.request

logger = logging.getLogger()
logger.setLevel(logging.INFO)

DUCKDNS_URL = "https://www.duckdns.org/update"


def public_ip(detail: dict) -> str | None:
    """Resolve a task's public IP from the event that announced it.

    The event names the network interface but not its address, so the
    association has to be read back from EC2.

    Args:
        detail: The `detail` object of an ECS Task State Change event.

    Returns:
        The public IPv4 address, or None if the task has no association yet.
    """
    for attachment in detail.get("attachments", []):
        for item in attachment.get("details", []):
            if item.get("name") == "networkInterfaceId":
                # Imported here, not at module scope: boto3 ships with the
                # Lambda runtime and is not a dependency of this repository.
                import boto3

                interfaces = boto3.client("ec2").describe_network_interfaces(
                    NetworkInterfaceIds=[item["value"]]
                )["NetworkInterfaces"]
                return interfaces[0].get("Association", {}).get("PublicIp")
    return None


def lambda_handler(event: dict, context) -> dict:
    """Update the DNS record when a task finishes coming up.

    Args:
        event: An EventBridge `ECS Task State Change` event.
        context: Lambda context, unused.

    Returns:
        A short record of what was done, which is what lands in the log.

    Raises:
        RuntimeError: DuckDNS rejected the update. It answers 200 either way,
            so the body is the only thing that says whether it worked.
    """
    detail = event.get("detail", {})
    # Every transition fires the rule -- PENDING, RUNNING, STOPPED. Only a task
    # that has finished coming up and is meant to stay carries an address worth
    # publishing.
    if (detail.get("lastStatus"), detail.get("desiredStatus")) != (
        "RUNNING",
        "RUNNING",
    ):
        return {"skipped": detail.get("lastStatus")}

    address = public_ip(detail)
    if not address:
        logger.warning(
            "Task reached RUNNING with no public IP: %s", detail.get("taskArn")
        )
        return {"skipped": "no public ip"}

    domain = os.environ["DUCKDNS_DOMAIN"]
    query = urllib.parse.urlencode(
        {"domains": domain, "token": os.environ["DUCKDNS_TOKEN"], "ip": address}
    )
    # The token rides in the query string, so the URL never reaches a log line.
    with urllib.request.urlopen(f"{DUCKDNS_URL}?{query}", timeout=10) as response:
        body = response.read().decode().strip()
    if body != "OK":
        raise RuntimeError(f"DuckDNS refused the update for {domain}: {body!r}")

    logger.info("Pointed %s.duckdns.org at %s", domain, address)
    return {"updated": address}
