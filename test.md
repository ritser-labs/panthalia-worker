|               | Brute Force | KR           | BM                         | KMP        | Suffix Trees         | Suffix Array        |
|---------------|-------------|--------------|----------------------------|------------|----------------------|---------------------|
| **preproc.**  | —           | O(m)         | O(m + \|Σ\|)               | O(m)       | O(\|Σ\|n^2) → O(\|Σ\|n) | O(nlogn) → O(n)    |
| **search time** (preproc excluded) | O(nm)      | O(n + m) expected | O(n + \|Σ\|) with good suffix often better | O(n)       | O(m(\|Σ\|))          | O(mlogn) → O(m + logn) |
| **extra space** | —           | O(1)         | O(m + \|Σ\|)               | O(m)       | O(n)                  | O(n)               |
