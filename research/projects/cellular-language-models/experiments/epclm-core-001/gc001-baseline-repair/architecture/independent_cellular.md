# Independent cellular and broadcast communication control

Both use 16×128 state, 2048 vocabulary, alpha .1, K4 and the unchanged Core parameter
structure/readout. `event[b,i,j] = embedding[token_b,(j-8*i) mod128]` is exactly
`roll(e,8*i)`; all arms share this deterministic parameter-free input. Generic embeddings
produce distinct cell inputs; a symmetric/periodic learned embedding could still yield
coinciding representations, which is a measured outcome rather than a guarantee to hide.

IC-ALL-SELF uses `center(norm_i)+left(norm_i)+right(norm_i)`. All three matrices train;
no recurrent operation aggregates cells. C1-BROADCAST replaces only message sources with
ring neighbors `(i-1) mod16` / `(i+1) mod16`, synchronously. Both start with the same
entire parameter hash. Global final readout is identical and allowed.

This differs from original entry-only C1 and from the old center-only NO-COMM. The latter
is not renamed or recycled as the independent model. Shared parameters across cells do
not create a recurrent state read between cells; direct perturbation isolation is tested.
