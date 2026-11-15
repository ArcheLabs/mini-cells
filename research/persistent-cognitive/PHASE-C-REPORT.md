# Phase C CPU verification

Three tests passed. The fixture checks controller initialization, the serial
reference, prefix causality, outer gradients and finite bounded fast weights
and momentum over 513 writes. Orthogonal address probes distinguish a
recent key, an older key and unseen directions.

The Phase C brief writes squared-error sum. The preceding implementation
and mathematical specification explicitly use mean squared error.
This change preserves the mean and its existing gradient scale.
The default control targets are eta=.5, beta=.1, alpha=.01 and rho=.99.
The actual initial learning coefficient is .005 and shrinkage is .0001.
These are conservative fixture defaults. Retention on pretrained Qwen
features remains unverified. Address solves run in FP32 or FP64.

Phase A/B remains PARTIAL. Pretrained Memory-on fusion and cross-window
retrieval are outstanding. The CPU test does not measure either.

The completion review adds two independent CPU oracles. A diagonal
covariance checks the spectral shrinkage coefficients and rejects an
asymmetric matrix. Direct autograd checks the configured learning and
forgetting coefficients, momentum sign and covariance increment.
