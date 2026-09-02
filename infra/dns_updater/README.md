# DuckDNS updater

Keeps a stable hostname pointing at the service, which otherwise moves.

A Fargate task is given a fresh public IP every time it is replaced, and it is
replaced on every deployment and on any health failure ECS decides to act on.
An EventBridge rule calls a Lambda on `ECS Task State Change`, and the Lambda
writes the new address to DuckDNS.

Triggering on the ECS event rather than from the deployment pipeline is the
point: a step in `cd.yml` would only ever see the replacements a deployment
causes, and would leave the record stale after any other.

## What this costs

Nothing. DuckDNS is free, and one Lambda invocation per deployment sits far
inside the free tier.

## Setup

Deployed by hand, so this file is the record of what was done.

**1. DuckDNS.** Sign in at [duckdns.org](https://www.duckdns.org), copy the
token, and add a subdomain. The token is a password: it goes in the Lambda's
environment, never in this repository.

**2. The function.** Lambda → *Create function* → author from scratch, Python
3.13, name `credit-risk-dns-updater`. Paste `lambda_function.py` and deploy.
Under *Configuration*:

- *Environment variables*: `DUCKDNS_DOMAIN` (the subdomain alone, no
  `.duckdns.org`) and `DUCKDNS_TOKEN`.
- *General configuration*: timeout 30 s. The default 3 s is not enough for an
  EC2 call and an outbound request on a cold start.
- *Permissions* → the execution role → *Add permissions* → inline policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "ec2:DescribeNetworkInterfaces",
    "Resource": "*"
  }]
}
```

The event names the network interface but not its address, so the association
has to be read back from EC2. `DescribeNetworkInterfaces` takes no resource
ARN, which is why this cannot be narrowed further.

**3. The rule.** EventBridge → *Rules* → *Create rule*, name
`credit-risk-task-started`, on the default event bus, *Rule with an event
pattern*. Custom pattern:

```json
{
  "source": ["aws.ecs"],
  "detail-type": ["ECS Task State Change"],
  "detail": {
    "clusterArn": ["arn:aws:ecs:eu-west-1:<account-id>:cluster/credit-risk-cluster"],
    "lastStatus": ["RUNNING"],
    "desiredStatus": ["RUNNING"]
  }
}
```

Target: the Lambda. EventBridge adds the invoke permission itself.

The pattern already filters on status, and the handler checks it again. That is
deliberate: the filter keeps the function from being woken for every transition,
and the check keeps it correct if the rule is ever loosened.

## Checking it works

Force a new task — deploy, or ECS → the service → *Update service* with *Force
new deployment*. Then:

```bash
dig +short <subdomain>.duckdns.org
curl http://<subdomain>.duckdns.org:8000/health
```

The Lambda's CloudWatch log group holds one line per invocation, naming the
address it published. `{"skipped": ...}` means the event was not a task
settling into place.

## Known limits

- **The record is HTTP on port 8000, with no certificate.** A browser will say
  so. Fixing that means a name you own and a proxy that terminates TLS, or a
  load balancer.
- **Two tasks would fight over the record.** The service runs one. Raising the
  desired count means this needs to become a load balancer instead.
- **A hand-deployed function drifts from the file beside it.** Nothing checks
  they agree. Worth wiring into CD only if it changes more than once a year.
