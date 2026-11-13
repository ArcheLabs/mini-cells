# E1 engineering prototype

Run e1_t4.ipynb with Tesla T4, Internet and GITHUB_TOKEN. The check
uses one private model and its native eager attention. It compares
first-chunk hidden states against E0 and checks that the selected
layer retains no KV while the other full-attention layers advance.
The selected layer sees only the current model call. Continued calls
are single-token on the pinned DeltaNet backend.

T4 execution and language-quality measurements are pending. Titans
fusion, gradient updates and outer LM training are subsequent gates.
