# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue. Report privately via
[GitHub Security Advisories](../../security/advisories/new) or by email to the
maintainer. You'll get an acknowledgement within 72 hours.

## Notes

- The default configuration processes only the bundled sample corpus and any
  documents you POST; nothing is sent to third parties.
- The optional `semantic` extra and any generative `LLMProvider` you wire in may
  call external services — review their data handling before enabling.
- The container runs as a non-root user.

## Supported versions

The latest minor release on `main` receives security fixes.
