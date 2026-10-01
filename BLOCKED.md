# BLOCKED — what only the author (or a GPU, or a person) can do

Every entry: what is needed, the exact steps, the expected time. Ordered by urgency (the date it blocks).
Nothing here needs money; nothing here may be done with a misstated age or someone else's account.
Updated 1 October 2026.

<!-- entries are maintained in this one file; the final ordering is set at the end of each session -->

## 1. Build the Gate 1 validation packet from the private release — by 12 October (blocks Gate 1, 18 October)

- **Needed:** the private v0.3 item files (`data/release/v0.3/noilai_{test,dev,core,main}.jsonl`, `attested.jsonl`), which exist
  only on the author's machine (git-ignored; the build seed is withheld).
- **Steps:**
  1. `pip install -r requirements.txt` (adds `openpyxl` for the workbooks).
  2. `make validation VALIDATORS="A B C"` (or `"A B"` with two validators) → `data/validation/validation_<V>.xlsx` per validator,
     CSV copies, and the author's keys (`B_key.json`, `A_calibration_key.json`, `D_engine.json`, `E_key.json` — never send these).
  3. Check `data/validation/validation_manifest.json`: `sizes.n_sample` 360, `n_controls` 48, hours per validator.
  4. Upload each workbook to Google Drive → open with Google Sheets (dropdowns survive), share each with its validator only.
- **Time:** ~2 minutes to build, ~20 minutes to upload and share.

<!-- more entries are appended below as the session proceeds -->
