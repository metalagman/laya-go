# Security policy

Report suspected vulnerabilities privately to the repository owner before
opening a public issue. Do not include model inputs, credentials, proprietary
bundles, or exploit data in public reports.

The supported security boundary is local, in-process inference over explicitly
supplied immutable artifacts. Model acquisition, credentials, external caches,
ADK permissions, deployment isolation, and downstream effects are application
responsibilities. Run `task release:verify` before publication and never attach
the protected native/model inputs to an issue or release.
