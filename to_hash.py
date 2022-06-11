def to_hash(i, j):
  if i <= j:
    i, j = j, i
  return i * (i - 1) // 2 + j
