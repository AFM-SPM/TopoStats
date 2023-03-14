---
name: Bug report
about: Create a report to help us improve
title: ''
labels: 'bug'
assignees: ''

---

**NB** TopoStats `< 2.0.0` is no longer actively maintained. If you have encountered an issue using this deprecated
version please try TopoStats `>= 2.0.0`, there are instructions on
[installation](https://afm-spm.github.io/TopoStats/installation.html),
[usage](https://afm-spm.github.io/TopoStats/usage.html) and
[configuration](https://afm-spm.github.io/TopoStats/configuration.html).

## Checklist

Please try and tick off each of these items when filing the bug report. There are further instructions on each below.

* [ ] Describe the bug.
* [ ] Include the configuration file.
* [ ] Copy of the output.
* [ ] The exact command that failed. This is what you typed at the command line, including any options.
* [ ] TopoStats version, this is reported by `run_topostats --version`
* [ ] Operating System and Python Version

## Describe the bug
A clear and concise description of what the bug is.

## Copy of the output

Please copy and paste the output that is shown below within the `\`` (triple-backticks).

```
<-- PASTE OUTPUT HERE -->
```


## Include the configuration file.

If no configuration file was specified with the `-c`/`--config-file` option the defaults were used, please use the
`run_topostats --create-config-file crash.yaml` to save these to the `crash.yaml` file and copy the contents below.

``` yaml
<-- PASTE CONTENT OF crash.yaml HERE -->
```


## To Reproduce

If it is possible to share the file (e.g. via cloud services) that caused the error that would greatly assist in reproducing and investigating the problem. In addition the _exact_ command used that failed should be pasted below.

```
<-- PASTE FAILING COMMAND HERE -->
```


## Output

Any output files that have been produced can be attached to this bug report (by default they will be under the `output` directory unless you have customised the configuration).

## TopoStats version

Please report the version of TopoStats you are using. There are several ways of doing this, either with `pip` or
`run_topostats`. Please copy and paste all output from either of the following commands.

- `pip show topostats`
- `run_topostats --version`

```
<-- PASTE TOPOSTATS VERSION -->
```

## Operating System and Python Version

### Operating System

Please let us know what operating system you are using, if you have used more than one then tick all boxes.

* [ ] Windows
* [ ] MacOS Intel (pre-2021)
* [ ] MacOS M1/M2 (post-2021)
* [ ] GNU/Linux (please include distribution)

### Python Version

Please let us know the version of Python you are using, paste the results of `python --version`

```
<-- PASTE PYTHON VERSION -->
```

### Optional : Python Packages

If you are able to provide a list of your installed packages that may be useful. The best way to get this is to copy and paste the results of typing `pip freeze`.

```
<-- PASTE PYTHON PACKAGE INFORMATION -->
```

## Additional context

Add any other context about the problem here.
