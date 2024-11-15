# TopoStats Pull Requests

Please provide a descriptive summary of the changes your Pull Request introduces.

The [Software Development](https://afm-spm.github.io/TopoStats/main/contributing.html#software-development) section of
the Contributing Guidelines may be useful if you are unfamiliar with linting, pre-commit, docstrings and testing.

**NB** - This header should be replaced with the description but please complete the below checklist or a short
description of why a particular item is not relevant.

---

Before submitting a Pull Request please check the following.

- [ ] Existing tests pass.
- [ ] Documentation has been updated and builds.
- [ ] Pre-commit checks pass.
- [ ] New functions/methods have typehints and docstrings.
- [ ] New functions/methods have tests which check the intended behaviour is correct.

## Optional

### `topostats/default_config.yaml`

If adding options to `topostats/default_config.yaml` please ensure.

- [ ] There is a comment adjacent to the option explaining what it is and the valid values.
- [ ] A check is made in `topostats/validation.py` to ensure entries are valid.
- [ ] Add the option to the relevant sub-parser in `topostats/entry_point.py`.
