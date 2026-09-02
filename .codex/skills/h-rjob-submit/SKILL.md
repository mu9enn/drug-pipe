---
name: h-rjob-submit
description: Prepare, validate, submit, and inspect normal GPU RJob workloads on PJLab H-cluster using the drug-pipe account's known namespace, image, mounts, and resource profiles. Use when Codex needs to generate or execute an `rjob submit` command that queues a real training, evaluation, conversion, or data job. Do not use for worker-internal `ray job submit` or for designing the training configuration itself.
---

# H-Cluster RJob Submit

Queue a reproducible workload whose RJob entrypoint is the real foreground task. Treat resource allocation and workload launch as one operation: a queued job must be ready to start useful work without a later SSH login.

## Resolve the job before submitting

1. Determine whether the user wants only a command/review or an actual submission. Running `rjob submit` consumes shared cluster resources; execute it only when the user has authorized submission in the current request. Never submit while merely creating or editing this skill.
2. Resolve the exact workload entrypoint from current repository files or a user-provided command. Prefer a versioned launcher on the mounted GPFS path. Do not invent a script path or training arguments.
3. Resolve the job name, image, namespace, charged group, mounts, replicas, GPUs/CPU/memory per replica, and any required environment variables. Read [local-profiles.md](references/local-profiles.md) for the known drug-pipe defaults and topology rules.
4. Read [submission-template.md](references/submission-template.md) when rendering, checking, submitting, or monitoring a command.

If no concrete executable workload is available, stop at a clearly marked command template and identify what is missing. Do not replace the missing workload with an idle command.

## Preserve these submission invariants

- Never use `sleep`, an interactive `bash`, `tail -f`, an infinite no-op loop, or a dummy service to hold GPUs. Do not disguise an idle allocation as training.
- The container command must remain in the foreground until the workload succeeds or fails. Use `exec` for the final process. A launcher that starts `tmux`, `nohup`, or a background process and then exits is not a valid RJob entrypoint.
- Use `bash -lc 'set -euo pipefail; ...; exec ...'` when environment initialization or a working-directory change is needed. Keep secrets out of `-x` traces and command-line arguments.
- Use shared, container-visible paths below `/root/slime_sxy/group-space/...`; do not submit a launcher that exists only under the login host's `/home/...` path.
- Keep job names lowercase and use only letters, digits, and hyphens. Avoid underscores and uppercase letters.
- Interpret `--gpu`, `--cpu`, and `--memory` as resources for each replica. `--memory` is MiB on the documented/current CLI; verify with `rjob submit --help` if the installed version is uncertain.
- Use `-P 1` for a single node, including one 8-GPU node. For multi-node jobs, `-P` is the node/replica count and resources remain per replica.
- `--enable-sshd` may be retained for diagnosis, but SSH is never a substitute for a real entrypoint.
- Do not add `--delete`, delete a same-named job, alter priority, select idle/preemptible mode, or add node tags merely to make scheduling succeed unless the user explicitly requests that change.
- Do not repeatedly resubmit a queued job. A successfully created Pending/Queued RJob is a successful submission, not a reason to create duplicates.

## Validate at the right depth

Before an authorized submission:

- Source `/etc/profile.d/ssh-init.sh` on the login/development host and confirm `rjob` is available.
- Inspect `rjob submit --help` for any uncertain or version-dependent flag. Do not silently upgrade `brainpp` or edit pip configuration unless the user asks.
- Confirm the host-side launcher and required config files exist on the mounted GPFS storage. Run focused static checks such as `bash -n` for a shell launcher when safe.
- Check that the command performs the intended real task, uses the expected container paths, does not daemonize, and propagates a nonzero exit code.
- For multi-node training, confirm the launcher consumes the platform-injected distributed variables and that communication flags match the topology. Do not overwrite injected NCCL interface/HCA/GID variables.
- Show the fully resolved command to the user before execution when material values remain inferred or when the request is review-only.

After submission, capture the RJob identifier/output and query it once with an explicit namespace. Report whether it was created and its current state. Do not wait for GPUs or poll continuously unless the user asks for monitoring.

## Diagnose without changing the request

For a job that stays queued, inspect the RJob state/events and compare the **per-node** free GPU, CPU, and memory against the per-replica request. Aggregate free resources do not prove that any one node can fit a replica. Also check namespace, charged-group permission, private-machine selection, replica/Gang requirements, mounts, image pull errors, and restrictive positive/negative tags.

For a job that starts and fails, inspect the job logs and events before proposing changes. The RJob result follows the foreground command's exit status, so preserve the first real workload error rather than replacing the command with a keepalive.
