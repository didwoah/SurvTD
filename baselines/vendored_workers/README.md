# Tracked copies of workers that live inside nested git repos

`baselines/dynamic_deephit_pytorch/` and `baselines/dsa_for_eep/` were vendored with
their upstream `.git` directories intact. Git will not track any path inside a nested
repository from the parent — `git add -f` on such a path silently does nothing — so the
project-authored workers written into those trees are **invisible to this repository's
history**.

That is how the D16 repair (`ddh_worker.py`'s training truncation, the sequence-length
label leak that put Dynamic-DeepHit below chance on Framingham) came within one
`rm -rf baselines/dynamic_deephit_pytorch` of being lost.

The copies here are the authority. To restore after a re-clone of the vendored tree:

    cp baselines/vendored_workers/ddh_worker.py baselines/dynamic_deephit_pytorch/

**The permanent fix** — not done here because it rewrites a vendored tree another
session may be using — is to delete the nested `.git` directories, so the vendored code
becomes ordinary tracked files of this repository:

    rm -rf baselines/dynamic_deephit_pytorch/.git baselines/dsa_for_eep/.git

`baselines/deep_tcsr/`, `baselines/tcsr/` and `baselines/signature_survival/` already
have no nested `.git` and are tracked normally; their workers need no copy here.
