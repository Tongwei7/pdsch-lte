# LTE PDSCH occupancy dataset

Server code root: `/home/luorongwei/pdsch_lte`.
Server data root: `/data/luorongwei/pdsch_lte`.

The active `matlab/` and `tools/` directories use the commented sources.
Each source has one canonical location, without reading copies or directory
aliases. See `代码阅读指南.md` and `首轮训练说明.md`.

The third stage defaults to **2,000 captures**, at most **20,000 10 ms windows**.
Planned captures and successful/rejected captures are reported separately.

## Git maintenance

See `GIT维护指南.md` for commits, branches, GitHub upload and recovery.
Edit the single server working directory through VS Code Remote SSH. Datasets,
weights and installed dependencies stay outside Git. MATLAB source bytes and
small synthetic MAT assets are preserved for generator revision checks.
`tools/record_run.py` records the current commit before an experiment.
`tools/package_project.py --output ...` exports a clean committed tree only on request.

Target: the selected LTE cell's PDSCH data-RE occupancy, including SI PDSCH.
Other cells are interference, never positive target labels. A segment denotes
contiguous occupancy, not a UE identity or a recoverable scheduling instance.

The generator uses MATLAB LTE Toolbox. See `DATA_CONTRACT.md` for the exact
training interface and the difference between receiver features and audit truth.
Generation is staged: physical-layer and training-reader acceptance first, then
pilot and production. A process launch is not a dataset acceptance result.

## Server use

Open `pdsch_lte.code-workspace` in the SSH VS Code window. It shows the server
code and data roots together. The generator never accesses the measured Y drive.

Run from a server bash terminal, in `/home/luorongwei/pdsch_lte`:

```
python3 tools/run_first_training.py

# Generate a new batch only, without training:
python3 tools/run_server.py --batch first_train_2000 --captures 2000 --chain --workers 4
```

`--chain` advances acceptance → pilot → production automatically only after
each preceding stage's gates pass. `--workers 8` uses eight independent MATLAB
processes with two computational threads each; GPU cards are not required for
this waveform-generation path.

Use `nohup ... > logs/first_training.log 2>&1 < /dev/null &` for an unattended
server process. No Codex session or VS Code connection is needed while it runs.
Generation status: `batches/<batch>/pipeline_status.json` under the data root.
Training status: `training_runs/first_train_2000/workflow_status.json`.
The first-training workflow prefers a checked subset of existing production
data and creates no MAT copies. If insufficient, it generates a new isolated
batch. Legacy acceptance/pilot/production outputs remain intact.
Resume reads the saved immutable plan, checks code/runtime identity and each
task's parameter hash. A mismatch requires a new batch name; old data is never
silently overwritten or reused under changed parameters.

The acceptance plan has 22 independent/paired captures with all 11 scenario
types. Its 66 short windows are for compatibility, not formal model evaluation.
Pilot: 500 captures, at most 5,000 usable 10 ms windows. Production: 2,000
captures, at most 20,000 windows. Stage seed ranges are disjoint.
Sync rejections are explicitly counted, so
requested and accepted counts can differ. Do not report planned samples as saved.

Execution failures and missing jobs block chaining. Acceptance permits no
sync rejections; pilot/production require at least 80% success and all 11
scenario types. These engineering gates do not establish model accuracy.
First training uses one GPU, at most 60 epochs and validation-based early
stopping, then evaluates the independent test split. Existing split/group
boundaries are preserved.

New routine shards retain float32 receiver I/Q, labels, public maps and small
diagnostics. Full CRS arrays and time-domain IQ are kept for acceptance and
approximately 2% diagnostic captures; the existing reader does not use the
full CRS arrays.

The launch script installs NumPy/SciPy into this project's `.deps` only when
needed. The existing system PyTorch remains unchanged. Snapshot training code
is in `training_compat`; it is the current staged reader/model copied for
verification, not a replacement for the separate model-development directory.

The standard-waveform cases include QPSK/16QAM/64QAM, localized and distributed
allocations, transmit diversity, and two-port rank-two spatial multiplexing.
The latter is observed through one receive branch for occupancy; the code does
not claim to decode both independent user streams from a single branch.

EPA/EVA/ETU use LTE Toolbox fading profiles. Two-ray cases are physically
generated notch stress tests, not a measured identification of propagation.
The first version uses fixed measured-family coarse center hypotheses validated
by received PSS/SSS and at least two consistent PBCH frames. It is not an
arbitrary-wideband spectrum-search receiver. Unknown actual CFO/timing are
estimated; transmitted values are used only to validate label alignment.

SIB1/SIB2 are synthetically encoded valid ASN.1 payloads, with SI-RNTI DCI and
standard DL-SCH/PDSCH mapping. There is no full core network, dynamic RRC session
or feedback-driven HARQ simulation. Traffic/MCS proportions are coverage choices
and must not be advertised as recovered ordinary-user measured statistics.
