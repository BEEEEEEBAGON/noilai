# Migration record
The NóiLái project was developed on a branch of the repository `BEEEEEEBAGON/ntcf` (the NTCF paper, causal discovery on ICU time series), under the folder `noilai/`, and moved to this repository on 1 October 2026 with its history.
- Original repository: https://github.com/BEEEEEEBAGON/ntcf
- Original pull request: https://github.com/BEEEEEEBAGON/ntcf/pull/1 (draft, closed without merging)
- Original head: `83c9bb628a64b01aaa6404e18c058530c1ae2283` (28 commits on top of ntcf `main` at `d38287b978c422b7abf97f182ae084ecb852854d`; every commit touched only `noilai/`)
- Method: `git filter-repo --subdirectory-filter noilai` on a fresh clone branched at that head; authors, dates and messages are preserved; `git filter-repo` rewrote the commit hashes quoted inside commit messages to the new hashes (the documents under `docs/` and the committed manifests still cite the ORIGINAL hashes and were not edited; use the table below to translate).
The frozen data (`data/release/v0.3`, public manifest) and the pre-registration were not edited. The manifest's `git_commit` (`f94f655…`) is the original freeze commit, which maps to the new hash in the table.
Edits made during the migration (everything else is byte-identical to the original head):

- `docs/MIGRATION.md` (this file)
- `.github/workflows/tests.yml` (new CI)
- `scripts/kaggle_build_notebooks.py` and the four regenerated notebooks: `REPO_URL` placeholder points at this repository
- `paper/figures/fig1_tokens.{tex,json}` and `fig1_tokens_note.tex`: regenerated so the provenance comment names this checkout's path
- `README.md`: origin note
## Commit map (original ntcf hash → hash in this repository)

| original | new | subject |
|---|---|---|
| `03af93f8da5be744a8e57a8d3f5846b0f1b61f5b` | `f43e7e9c49951da20d71d51648b8c455b5902c0c` | noilai: snapshot of the interrupted paper/docs review round (in progress; six pre-registra |
| `1f5214b2aca7686fe54a8b7feaebbe8376c3a79d` | `4e936001177dd51674a83e4adde1977bf16bdcb7` | noilai: red-team corrections to the generator and attested seed; seeded main and C2 sample |
| `20b225df071758af0c7fc04291026fd246a84fd2` | `71a86bb097702c890e196b712942741d6957770c` | noilai: snapshot of the adversarial-review fixes in progress (generator and eval-harness r |
| `21fbeec9d7061d5bbf6c299cc9b20c907c1b4e53` | `4a56625bd9b2d822521a41cc5525ac7113a966ff` | noilai: implementation decisions record, v0.1 build manifest, results log entry |
| `43ace965fe559693b36df32295ead111c3060aa1` | `8a4092ee50b636bd5b55e209e7ec7e66c9c06d4f` | noilai: E4 fixes from the design review and the E2 fixed-effects analysis |
| `4c358e68ce5d7513097aca104602fc3c8c9aa01c` | `7f6bb9f6b60daecae6ef7938768d6be0db8610af` | noilai: constants module, count-reconciliation script (fixes the Gemma 3 NFD figure: 2.83, |
| `5692c9176e5dacc03acb1c6ffcb1f2c94d44b29e` | `93481344a3289d7f4c61e2d12eb984c3ea046022` | noilai: lint cleanup (ruff --fix) on the core modules, scripts and tests |
| `640b0469740b0b1cc9e1ae0ddc44604f59daacbf` | `0cdfcaba6e6cbfb61351e3a1a209b0a432c3a00f` | noilai: statistics package and E4 probing/patching code |
| `699634ca73be733dd3580bccf37443125c6e55c0` | `a01a65fc7c00ebe9589719cad72722922c5f37e0` | noilai: Vietnamese syllable parser, nói lái generator, re-encoder, tokenizer audit (ACL 20 |
| `771b6900ee7721bf667f863fb608cb7297b1e123` | `48b61b5fc4f303075d392974f673db25b8027bbc` | noilai: attested-example builder, validation and human-baseline form tools, qu- exclusion, |
| `7dadbf08b60be47fe2352c0f6276aa582f3193a1` | `a26e0d7ba1d7a7f7596758cad982f73d6294718f` | noilai: lint pass (imports, stale noqa, PEP 604 optionals), executable scripts, arm_scope  |
| `83c9bb628a64b01aaa6404e18c058530c1ae2283` | `75abbe10d79cc36ff72222115b17e5f0ffc189c6` | noilai: release v0.3 built from the freeze commit 62f63ae (clean tree, content 92d332e5, s |
| `975be2781a822cfb413afd89a7627a22e25c1f91` | `00815d36bc7621797845dd7cf5ea6ecd91a78d8f` | noilai: v0.2 generator per the binding design review |
| `9ce53815a006a0bd73c537a72c27701769d95155` | `fa5165510825878dfe65250d14d444e62d5feae8` | noilai: snapshot of the adversarial-review fixes, rounds 3-4 (statistics/E4 and cloud kit) |
| `aa54c3a10cdff88c4719c46daa78cab9b6e6af31` | `8656cfdac50866574a0f030cbba5fd93b89bbcf3` | noilai: qu- exclusion covers T2 inputs; release-invariant tests; v0.1 rebuilt |
| `af03e10afb826e36567b716e055b7a13dcd8ebd0` | `6312aca449c46300a0309e2281e7656a57447f65` | noilai: snapshot of the paper/docs review round in progress |
| `b03c7d5e04d681f3de3b61bbfe4598c18e662cea` | `2780a811b38d2a9db59b144cb55b32755f80ff5e` | noilai: item-level tokenization audit, tokenizer consistency check, week-1 checklist and o |
| `b223548a740960056b853565dbd8138bd5dba9ca` | `41a9d05ec3e034c24e2e37a025a78c057515fb65` | noilai: evaluation harness, cloud run kit and paper skeleton as delivered by the implement |
| `b39dc95b82fbbe3a1300301f2ade58e8ce7b5f46` | `30d915b8c4b71f9bce0f4029497339a2e7eb59de` | noilai: uniform pseudo sampling when no lexical pairs (C2 set builds: 500 items); attested |
| `be640d012b9c8103435bf1a58a01c9eddce1c389` | `498fb5cc6e35a82f739d8db9f7850f0c649380cc` | noilai: snapshot of the fix-round edits in progress (eval harness, cloud kit) |
| `bf74af052003ad0776c9bb7edfecb6084ac2b6ca` | `c5a23017f8019b1cbe9a8e48a5b9898cbeb8c818` | noilai: snapshot of the review follow-up round in progress (seed redaction, census consume |
| `ca2dcfbc5214ff84176bae492572546468cbbdcf` | `53e6dd0aad15b4cc7b9d3268679adbd2f63a2d50` | noilai: extended syllable inventory with a disk-cached word-list index; Makefile |
| `e263f7b0e1167e65c3a367cf3b4cdcc764361e38` | `3bac2d1240a074b297287b7af725fa7d455b0103` | noilai: deviations log, placement-convention counter (XCOPA-vi: 60 old vs 2 new), results  |
| `ed2d28a7abbeb9e87faf35969966a5adc2a7de12` | `316fda09a0b7b88a39bfe4176a5615ac94c80cfc` | noilai: snapshot of in-progress agent deliverables (eval harness, cloud kit configs, promp |
| `ee28792129bc9f7b044a6ae52a78363735c9ddbf` | `02d062d5cc4a0d99c282cb5dbca6dfe1498e4930` | noilai: requirements and a pip-freeze lock of the build environment |
| `f2b36daabcef867380e665c96ab6985dbe9c65e5` | `e29b6e154ace3e033c8aeffab8cf52334ce49e09` | noilai: adversarial review, round 5 (paper and study documents): 47 findings fixed, doc-vs |
| `f94f655f4d6e04fc548979a4c03010d109da2de5` | `62f63ae5a7370823286bb51efb2e813dfe52ab8e` | noilai: review follow-ups closed and the v0.3 freeze: build seed withheld (public/private  |
| `fd27363bc1057fb21258ccc767468490a0fb5913` | `087fdfc66cad06df1ffbdf500f402a300f4226c3` | noilai: snapshot of in-progress agent deliverables (2) |
