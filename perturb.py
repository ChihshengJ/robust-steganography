def perturb(p, r, delta):
  N = vocab_size

  # Build I: set of indices in [N] for which p_i ∈ [2δ, 1 − 2δ].
  I = set()
  for i, p_i in enumerate(p):
    if (p_i >= 2 * delta and p_i <= 1 - 2 * delta):
      I.add(i)

  # Set w to be the number of indices in [N] for which r_i = 1 and δ′ = δw/(N − w).
  w = sum(r)
  delta_prime = (delta * w) / (N - w)

  # Adjust probabilities
  for j in I:
    if (r[j] == 1):
      p[j] += delta
    else:
      p[j] -= delta_prime
  # the j not in I stay the same and since p was updated in place this has been handled
  p = p / sum(p)

  return p