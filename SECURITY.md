# Security Policy

## Supported Versions

Substrax is in early development. Security fixes target the latest released
version and the current `main` branch.

## Reporting a Vulnerability

Please report suspected vulnerabilities privately by emailing
<security@avitai.bio>.

Include:

- Affected Substrax version or commit.
- Environment details, including Python, JAX, and operating system versions.
- A minimal reproduction or proof of impact.
- Any known mitigations.

We will acknowledge reports as soon as practical, investigate privately, and
coordinate disclosure once a fix or mitigation is available.

## Scope

Security-sensitive areas include dependency handling, checkpoint loading and
restoration, experiment-tracking integrations that talk to remote services,
file-system writes, and CI release automation.
