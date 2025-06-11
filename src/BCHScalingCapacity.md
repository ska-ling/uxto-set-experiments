# Bitcoin Cash Scaling Capacity

## Current throughput benchmark

* **Total transactions per second:** **115 000 tx/s**  
* **Total inputs per second:** 288 000  
* **Total outputs per second:** 316 000  

---

## 1. Required Block Size at 115 k TPS

| Step | Result |
|------|--------|
| Measured throughput | **115 000 tx/s** |
| Block interval (BCH) | **600 s** (10 min) |
| Transactions per block | 115 000 × 600 = **69.0 million** |
| Average size/tx (≈ 2.6 inputs + 2.9 outputs) | **≈ 490 bytes** |
| **Estimated block size** | 69.0 M × 490 B ≈ **34 GB** |

> *If every transaction were minimal (≈ 150 B) the block would be ~10 GB; with a typical P2PKH footprint (≈ 250 B) it would be ~17 GB.*

---

## 2. Network Capacity per Day and Year

* **Transactions per day:** 115 000 × 86 400 = **9.94 billion**  
* **Transactions per year:** 9.94 B × 365 ≈ **3.63 trillion**

---

## 3. Can This Cover All Global Payments?

| Metric | World Demand (non-cash)^1 | Capacity at 115 k TPS |
|--------|---------------------------|-----------------------|
| Non-cash payments, 2023 | ~1.3 trillion | **3.63 trillion** |
| Projected for 2027 | ~2.3 trillion | **3.63 trillion** |
| Coverage | 279 % (2023) / 158 % (2027) | — |

*Visa’s average load is ~639 million tx/day (≈ 7 400 peak TPS); the stated BCH throughput is ~15 × higher over a full day.*

---

## 4. “People × Transactions per Day” Supported

| Transactions **per person per day** | **Population supported** |
|-------------------------------------|--------------------------|
| 1 | **9.94 billion** |
| 2 | 4.97 billion |
| 3 | 3.31 billion |
| 5 | 1.99 billion |
| 10 | 0.99 billion |

Example: **2 billion people** could each make **five payments a day** without exhausting the 115 k TPS ceiling.

---

## 5. Practical Considerations

* **Propagation:** 10–34 GB blocks require multi-gigabit links, high-IOPS NVMe storage, and aggressive block-compression/forwarding techniques.  
* **State growth:** UTXO set and mempool would expand at unprecedented rates, stressing RAM and disk.  
* **Decentralization trade-off:** As hardware requirements climb, the number of hobby-grade full nodes falls, concentrating validation.

---

### Footnote  
1. *World Payments Report 2024* (Capgemini) for non-cash volumes. Visa load from Visa Inc. FY 2024 filings.
