# LTE PDSCH occupancy dataset

Server code root: `/home/luorongwei/pdsch_lte`.
Server data root: `/data/luorongwei/pdsch_lte`.

The active `matlab/` and `tools/` directories use the commented sources.
Each source has one canonical location, without reading copies or directory
aliases. See `代码阅读指南.md`
for the verified comment-only revision compatibility and deployment checks.

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
python3 tools/run_server.py --stage acceptance --workers 2
python3 tools/run_server.py --stage pilot --workers 4
python3 tools/run_server.py --stage production --workers 4
```

`--chain` advances acceptance → pilot → production automatically only after
each preceding stage's gates pass. `--workers 8` uses eight independent MATLAB
processes with two computational threads each; GPU cards are not required for
this waveform-generation path.

Use `nohup ... > logs/<stage>_launcher.log 2>&1 < /dev/null &` for an unattended
server process. No Codex session or VS Code connection is needed while it runs.
`/data/luorongwei/pdsch_lte/pipeline_status.json` contains current progress.
Each completed capture is checkpointed. Repeating a stage resumes matching
generator revisions; mismatched revisions regenerate that stage's owned files.

The acceptance plan has 22 independent/paired captures with all 11 scenario
types. Its 66 short windows are for compatibility, not formal model evaluation.
Pilot: 500 captures, at most 5,000 usable 10 ms windows. Production: 20,000
captures, at most 200,000 windows. Sync rejections are explicitly counted, so
requested and accepted counts can differ. Do not report planned samples as saved.

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

## Git maintenance

See `GIT维护指南.md` for the single server working directory, commits, GitHub backup, branches and run provenance. `tools/record_run.py` records a pre-launch commit; `tools/package_project.py --output ...` exports committed code only on explicit request. Datasets and installed dependencies stay outside Git.
