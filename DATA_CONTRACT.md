# Dataset contract: selected-cell LTE PDSCH occupancy

Dataset version: `lte-pdsch-occupancy-v1`. Existing-reader schema remains
`pdsch-occupancy-proposal-v1` in manifest.csv because all required fields and
their semantics are preserved exactly. Extra fields are documented below;
the existing model does not automatically consume them.

## Supervised target

Only the selected target cell's actual PDSCH DATA resource elements are positive,
including SI-RNTI PDSCH. CRS, PSS/SSS, PBCH, PCFICH, PHICH and PDCCH are zero.
An interfering cell's PDSCH alone is also zero. All transmit antenna planes are
unioned. Occupancy survives low received power and decode failure.

* `rx_real`, `rx_imag`: finite single `[K,14,1,S]`; actual receiver FFT output.
* `subframe_index`: int64 `[1,S]`; capture-relative target subframe coordinate.
* `label_re`: uint8 `[K,14,S]`; transmitter data mapping, aligned to receiver.
* `label_prb`: uint8 `[K/12,S]`; union over 12 subcarriers and 14 symbols.
* `label_prb_slot`: uint8 `[K/12,2,S]`; union over 12 subcarriers and 7 symbols.
* `public_excluded`, `public_valid`: uint8 `[K,14,S]`; receiver-derived known
  non-PDSCH sites, with explicit unknowns. No transmitter CFI is substituted.
* `crs_h_real`, `crs_h_imag` (optional diagnostics): single `[K,14,1,2,S]`; receiver-estimated channels
  for target CRS ports 0/1, NOT ideal PDSCH effective channels in all modes.
* `crs_noise_variance`: single `[1,S]`; receiver estimation residual, not oracle
  AWGN power. May include interference and estimator error.
* `cfi_estimate`, `cfi_hard_mismatch`: receiver PCFICH decision and codeword
  hard-mismatch fraction. PCFICH has no independent CRC.

K is 600/900/1200 for 10/15/20 MHz. Subcarrier axis increases with frequency,
DC is omitted. All exposed coordinate VALUES start at zero; MATLAB array
indexing remains one based. Logical time order is subframe then its 14 symbols.
Network windows contain 10 consecutive subframes: `[C,K,140]`. Windows need not
start at a radio-frame boundary. SFN and time offsets are audit metadata.

## Isolation

Training input is I/Q/power and optionally receiver public maps. Extra CRS
fields are preserved on acceptance/diagnostic captures for a future explicitly versioned input extension. The
existing network's PRB head is per subframe, not per slot: slot labels are
available for later head expansion and precise evaluation.

Separate audit JSON contains seed, selected PCI, actual grants, RNTI, MCS,
transmission scheme, channel, noise and known timing. Never pass these oracle
fields to the model. Synchronization uses received signals; oracle timing only
validates and attaches labels. Sync/PCI/MIB/frame-alignment failures are recorded
and rejected, never converted into negative examples. Difficult successful
captures retain their true labels even if payload decoding fails.

## Splits and files

MAT files use `-v7` (not v7.3), one 100 ms shard initially. A manifest row has
schema,path,split,group_id,alignment_valid plus scenario and checksum fields.
Paths are relative to the manifest. All continuous captures, paired allocations
and their same-origin channel/noise variants share a group_id and stay in one
split. Independent split-specific seeds are fixed before generation. Training
normalization and thresholds never fit on validation/test inputs/labels.

The send ledger is saved with each capture; RE and PRB consistency and resource
collisions are checked. Read acceptance calls the actual current training
reader and performs a forward/backward pass through its current U-Net. Such a
pass validates compatibility, not learned detection performance.

## Model scope and evidence

Measured evidence supports FDD, normal CP, two CRS ports, bandwidths 10/15/20 MHz,
carrier centering errors, same-frequency PCI interference and partially observed
adjacent LTE carriers. Ordinary-user TM/MCS/load distributions and hardware
calibration remain unknown. Their synthetic distributions are explicit design
assumptions. This version starts with transmit diversity and also covers
CRS-based spatial multiplexing where verified; it does not claim all LTE TM,
RRC/MAC behavior, or that finite simulated data fully reproduces reality.
