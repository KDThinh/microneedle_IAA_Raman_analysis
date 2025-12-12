# Raman Processing Guide

This document walks through running new Raman/fluorescence datasets through the shared pipeline.

---

## 1. Repository Layout

- `config/pipeline.yml` – central list of **profiles**. Each profile describes where the raw data lives, metadata about the experiment, processing parameters, and plotting/output preferences (e.g., time-series x-axis).
- `main_test_v3.py` / `main_test_v4.py` – thin wrappers that delegate to the modular CLIs in `src/cli/`.
- `main_in_vitro.py` – streamlined pipeline for cuvette/in-vitro data.
- `src/pipeline/` – shared modules (ingestion, plotting, post-processing, utilities).
- `src/cli/` – CLI implementations (batch processing, FFT generation, batch processors, etc.).
- `compile_batch_summaries.py`, `compile_fft_data.py` – utility scripts that aggregate outputs across profiles (replacing the legacy master-CSV append workflow).

---

## 2. Create or Modify a Profile

Profiles let you define experiment-specific settings once. Edit `config/pipeline.yml` and either:

1. **Duplicate an existing profile** and change the paths/metadata, or  
2. **Create a new entry** from scratch.
3. **Auto-generate** them with `scripts/generate_profiles.py` (see section 7).

Example:

```yaml
profiles:
  bok_choy_run2:
    inherits: in_planta_default          # optional; copies defaults
    data_source:
      requires_google_drive: true        # true ⇒ resolve under GoogleDrive\My Drive\Work
      raman_relative_path: DiSTAP/Research/.../Bok Choy/.../Run 2/Raw data/combined_raman_data.txt
      temp_relative_path: DiSTAP/Research/.../Temperature and humidity/Indoor_thermometer.csv
    metadata:
      plant_type: Bok Choy
      treatment: Control
      temp_hum_control: No
      light_cycle: 6to22
      replicate_number: 2
```

> Tip: keep the profile names descriptive (`bok_choy_run2`, `nb_shade_run3`, etc.) so you can rerun later without guessing paths.

---

## 3. Run the Pipeline

From `Script/swnt_iaa_analysis_v2`:

```powershell
python main_test_v4.py --profile bok_choy_run2
```

During the run you will:

1. See where the script expects the raw and temperature files.
2. Be prompted for start/end datetime filters (press Enter to accept defaults).
3. Generate plots + CSVs in `Raw data/test_outputs_v4_*` (timestamped).
4. Finish with self-contained batch outputs (later aggregated via `compile_batch_summaries.py` or `compile_fft_data.py` as needed).

### Alternate entry points

- `python main_test_v3.py --profile ...` – legacy v3 pipeline (older plotting, same data products).
- `python main_in_vitro.py --profile in_vitro_default` – for cuvette runs (no temperature input needed by default).

---

## 4. Use Command-Line Overrides (Optional)

If you don’t want to edit `pipeline.yml`, override settings inline:

```powershell
python main_updated_v1.py ^
  --profile in_planta_default ^
  --config-override data_source.raman_relative_path="DiSTAP/.../combined_raman_data.txt" ^
  --config-override metadata.plant_type="Bok Choy" ^
  --config-override metadata.light_cycle=6to22
```

(Use `\` instead of `^` if you’re not on PowerShell.)

Overrides can target any dotted key inside the profile (`processing.window_size`, `outputs.timeseries_x_axis`, etc.).

---

## 5. Checklist Before Running

1. **PyYAML installed** – `pip install pyyaml`.
2. **Paths verified** – The Raman `.txt` and temperature `.csv` exist at the paths listed in the profile or overrides.
3. **Google Drive mounted** – Only if `requires_google_drive: true`.
4. **Output expectations** – Plan which aggregation script (`compile_batch_summaries.py`, `compile_fft_data.py`) you’ll run after generating the batch outputs.

---

## 6. Troubleshooting Tips

- **File not found**: double-check `raman_relative_path` (relative to `My Drive/Work/…` when using Google Drive). Use absolute paths with `requires_google_drive: false`.
- **Datetime parsing errors**: ensure the Raw data file’s first two columns contain date/time in the standard format (`MM-DD-YYYY HH:MM:SS.sss`). If not, re-export with the expected format.
- **Savitzky–Golay window warnings**: the ingestion validator will auto-correct odd/even window sizes when possible; otherwise adjust `processing.window_size`.
- **Aggregation scripts can’t find batch results**: verify each profile has at least one timestamped `test_outputs_v{N}_...` directory with the expected `batch_summary_*` or `fft_full_*` CSVs.

---

## 7. Auto-Generate Profiles (`scripts/generate_profiles.py`)

When many experiments follow the same structure, let the helper script build profiles automatically.

### Location

`scripts/generate_profiles.py`

### What it does

- Recursively scans `IAA Nanosensor Experiment/**/Raw data/*.txt`
- Infers metadata from folder names (plant, treatment, light, temperature control, run number)
- Creates concise profiles that `inherit` from an existing base (default: `in_planta_default`)

### Common command

```powershell
python scripts/generate_profiles.py ^
  --base-profile in_planta_default ^
  --profile-prefix auto ^
  --overwrite ^
  --dry-run
```

Flags:
- `--base-profile` – template to inherit processing settings from.
- `--profile-prefix` – string prepended to each generated profile name.
- `--overwrite` – replace existing profiles with the same name (omit to skip duplicates).
- `--dry-run` – list what would change without modifying `pipeline.yml`.
- `--work-root` – set if your Google Drive “Work” folder is not auto-detected.
- `--data-root` – relative path from Work to the experiment root (defaults to `DiSTAP/.../IAA Nanosensor Experiment`).

After reviewing the dry run, rerun without `--dry-run` to write the profiles. The script keeps entries alphabetized so `pipeline.yml` remains easy to navigate.

---

With profiles in place—whether handcrafted or auto-generated—processing each experiment is as simple as running the appropriate command. Feel free to extend this guide with experiment-specific notes. Happy processing!

