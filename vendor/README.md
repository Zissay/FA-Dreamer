# Preserved Dreamer runtime

`dreamerv3-installed.tar.gz` is the installed `dreamerv3==1.3.0` source captured
in the 500 km/h Dreamer training **pretrain** snapshot dated 2026-10-08.
The training launcher extracts it to ignored `vendor/runtime/` and adds that
directory to `PYTHONPATH` for the Dreamer subprocess. This keeps the upstream
runtime source fixed while leaving site-packages unchanged.

The archive contains source, not a trained model or replay. Required third-party
dependencies still come from `requirements/dreamer.txt`; the complete observed
package list is under `requirements/locks/`.

Dreamer upstream: <https://github.com/danijar/dreamerv3>.
The upstream MIT notice is retained in `LICENSE.dreamerv3`.
The original 50 km/h launch runtime was not archived; this same 1.3.0 runtime
is a compatibility choice for its historical source candidate.
