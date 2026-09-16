# Local DeepSeek Harness patch

The main repository pins the upstream submodule revision. `optional-read-image.patch`
preserves the local readImage switch used to keep the text-only evaluation tool catalog stable.
It includes the upstream-style regression test. No unpublished submodule commit is required.

After `git submodule update --init slime-wd/deepseek-harness`, from repository root:

```sh
git -C slime-wd/deepseek-harness apply --check ../../patches/deepseek-harness/optional-read-image.patch
git -C slime-wd/deepseek-harness apply ../../patches/deepseek-harness/optional-read-image.patch
```

If `git apply --reverse --check` succeeds, the patch is already present. Never reset an
existing dirty checkout to apply it. A patched submodule is expected to show local modifications.
