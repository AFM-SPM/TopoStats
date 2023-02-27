---
name: Bug report
about: Create a report to help us improve
title: ''
labels: 'bug'
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**

- Command used to run topostats e.g. `python pygwytracing.py` / `python topostats/topotracing.py`
- Config file (you can attach this or paste it below)

**Expected behavior**
A clear and concise description of what you expected to happen.

**Output**

If applicable please include the output error, this can be a copy and paste of the output (preferable) or a screenshot.

Any images or output files that have been produced can be attached to this bug report (by default they will be under the
`output` directory unless you have customised the configuration).

** TopoStats version

Please report the version of TopoStats you are using. There are several ways of doing this, either with `pip` or
`run_topostats`. Please copy and paste all output from either of the following, although the later is preferable if you
have installed TopoStats from source/GitHub and are working on features.

- `pip show topostats`
- `run_topostats --version`


** Your computer configuration (please complete the following information):**

- OS: e.g. windows, MacOS, linux; please include OS version
- Python version: paste the results of typing `python --version`.
- Optionally, your installed packages: the best way to get this is to copy and paste the results of typing `pip freeze`.

**Additional context**

Add any other context about the problem here.
