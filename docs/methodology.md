# Methodology and Metric Definitions

## Sentence representations

For decoder block `l`, let token activations be `H_l` with shape `T x d`. RepSteer
maps them to one sentence representation using attention-mask-aware mean pooling,
the final non-padding token, or feature-wise minimum or maximum pooling. Layer zero
is the first decoder block output, not the embedding input.

## Difference-of-means vectors

For binary variable `x`, condition `c`, and layer `l`, the stored direction is:

```text
v_l,c = mean(h_l | x=positive, c) - mean(h_l | x=negative, c)
```

If normalization is enabled, steering uses `v_l,c / ||v_l,c||`. The archive also
retains the unnormalized class gap for normalized projection analyses. To target the
positive class, RepSteer adds `alpha * v`; to target the negative class, it adds
`-alpha * v`.

This estimator is difference of means. It is canonical CAA only when the supplied
contrast texts were constructed with the CAA contrastive prompt method. Dataset
examples grouped by labels do not become CAA merely because their means are
subtracted.

## Decodability

Decodability is held-out linear classification of one named variable while balancing
samples across every full condition. It measures whether the variable is linearly
readable, but does not require that its code generalize across nuisance contexts.

## CCGP

Cross-condition generalization performance trains a classifier for one named
variable on nuisance contexts and evaluates it on a held-out context. Results are
averaged over held-out contexts and resamplings. High CCGP indicates a variable uses
a code that transfers across changes in the other experimental variables.

## Parallelism score

For a named variable, RepSteer constructs its positive-minus-negative direction in
each nuisance context. PS is the average pairwise Pearson alignment of those
context-specific directions in the Decodanda backend. High PS means the concept
directions are similarly oriented across contexts.

## Shattering dimensionality

SD enumerates balanced dichotomies of complete condition identities, removes sign
complements, trains held-out classifiers for each dichotomy, and aggregates their
accuracies. SD describes the complete representational condition space. It is not
specific to language, style, concept, or any other single named variable.

`sd_mean_accuracy` is the mean classification accuracy over evaluated dichotomies.
Thresholded SD is the fraction above `sd_accuracy_threshold` when that output is
requested.

## Standardization

With `zscore: true`, classification metrics use statistics fitted only on their
training split. Decodanda PS uses its pair-local standardization procedure. This
avoids leaking held-out test statistics into classifiers and matches the production
backend semantics.

## Native versus Decodanda

The Decodanda engine is the reference implementation for comparisons with prior
experiments. The native engine is self-contained and useful for unit tests or quick
prototyping, but its values should not be presented as numerically interchangeable
with Decodanda results.
