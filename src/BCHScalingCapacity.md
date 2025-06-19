# Bitcoin Cash Scaling Capacity

## Current throughput benchmark

* **Total transactions per second:** **202 000 tx/s**  
* **Total inputs per second:** 451 000  
* **Total outputs per second:** 506 000  

---

## 1. Required Block Size at 202 k TPS

| Step | Result |
|------|--------|
| Measured throughput | **202 000 tx/s** |
| Block interval (BCH) | **600 s** (10 min) |
| Transactions per block | 202 000 × 600 = **121.2 million** |
| Average size/tx (≈ 2.2 inputs + 2.5 outputs) | **≈ 430 bytes** |
| **Estimated block size** | 121.2 M × 430 B ≈ **52 GB** |

> *If every transaction were minimal (≈ 150 B) the block would be ~18 GB; with a typical P2PKH footprint (≈ 250 B) it would be ~30 GB.*

---

## 2. Network Capacity per Day and Year

* **Transactions per day:** 202 000 × 86 400 = **17.45 billion**  
* **Transactions per year:** 17.45 B × 365 ≈ **6.37 trillion**

---

## 3. Can This Cover All Global Payments?

| Metric | World Demand (non-cash)^1 | Capacity at 202 k TPS |
|--------|---------------------------|-----------------------|
| Non-cash payments, 2023 | ~1.3 trillion | **6.37 trillion** |
| Projected for 2027 | ~2.3 trillion | **6.37 trillion** |
| Coverage | 490 % (2023) / 277 % (2027) | — |

*Visa’s average load is ~639 million tx/day (≈ 7 400 peak TPS); the stated BCH throughput is ~27 × higher over a full day.*

---

## 4. “People × Transactions per Day” Supported

| Transactions **per person per day** | **Population supported** |
|-------------------------------------|--------------------------|
| 1 | **17.45 billion** |
| 2 | 8.73 billion |
| 3 | 5.82 billion |
| 5 | 3.49 billion |
| 10 | 1.75 billion |

Example: **3.5 billion people** could each make **five payments a day** without exhausting the 202 k TPS ceiling.

---

## 5. Practical Considerations

* **Propagation:** 18–52 GB blocks require multi-gigabit links, high-IOPS NVMe storage, and aggressive block-compression/forwarding techniques.  
* **State growth:** UTXO set and mempool would expand at unprecedented rates, stressing RAM and disk.  
* **Decentralization trade-off:** As hardware requirements climb, the number of hobby-grade full nodes falls, concentrating validation.

---

### Footnote  
1. *World Payments Report 2024* (Capgemini) for non-cash volumes. Visa load from Visa Inc. FY 2024 filings.
