# Security Policy

## Supported Versions

| Version  | Supported |
|----------|-----------|
| 0.1.0a1  | yes (pre-alpha; security only) |

## Reporting a Vulnerability

Open a private security advisory: https://github.com/hinanohart/polyalign/security/advisories/new

We aim to acknowledge within 7 days. Coordinated disclosure preferred; please do not file a public issue for unpatched vulnerabilities.

## Dependency policy

- Core dependencies: numpy, scipy, POT, typer.
- Optional extras: torch, transformers, sae-lens, mamba-ssm.
- Dependabot is configured for **security-updates only** (no version bumps). See `.github/dependabot.yml`.
