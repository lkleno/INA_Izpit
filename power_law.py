def power_law(G, kmin = 1):
  n = 0
  sumk = 0
  for _, k in G.degree():
    if k >= kmin:
      # sumk += math.log(k / (kmin - 0.5))
      sumk += math.log(k)
      n += 1
  # return 1 + n / sumk if n > 0 else math.nan
  return 1 + 1 / (sumk / n - math.log(kmin - 0.5)) if n > 0 else math.nan