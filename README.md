# DNN Assignment 2: RNN, LSTM, GRU and Transformer

Chosen values: hidden size h = 128 for Problem 1, model width d = 128 for Problem 2, and seeds 7, 21 and 42. My student ID 2026AK05422 ends in 2, which is even, so the P1.5(c) ablation is gradient clipping (norm 1.0 vs none) for the vanilla RNN at T = 200.

## Environment
Google Colab with the free T4 GPU, Python 3.13.18, PyTorch 2.14.1+cu130, numpy 2.4.6, pandas 2.2.3, matplotlib 3.10.0 and scikit-learn 1.6.1. The versions are pinned in `requirements.txt`.

## How to run
1. Upload the `problem1/` and `problem2/` folders to the Colab file area (`/content`). All paths are relative, so Drive does not need to be mounted.
2. Open `problem1/problem1.ipynb`, select a T4 runtime and use Run all. The first cell changes into `problem1/` if needed. The notebook downloads UCI HAR, creates `./data`, `./figures` and `./cache`, and adds one row per run to `../results.csv`.
3. Run `problem2/problem2.ipynb` the same way, but only after Problem 1, because table P2-T1 reads the P1.3 rows from `../results.csv`. It downloads Tiny Shakespeare and writes `generated_samples.txt`.
4. If Colab disconnects, use Run all again. Finished runs are loaded from `./cache` and are not trained again.

## Runtime (Colab T4, fresh session, empty cache)
Problem 1 takes about 15 minutes: P1.3 105 s, P1.4 601 s, P1.5b 187 s, plus the no-clip runs and the gradient study. Problem 2 takes about 13 minutes: P2.3 250 s, P2.4 165 s, the ablations 204 s and the bonus 128 s, plus the context probes and the data download. The total is about 30 minutes.

## Assumptions
- The variants table is ignored, as the instructor announced, and the 3-token attention example is my own.
- The seeds set the model initialisation, the shuffling and the HAR validation-subject split. The adding-problem data seeds stay at 0, 1 and 2.
- All HAR runs (P1.3, P1.5b, P2.3) and all adding-problem runs (P1.4, P1.5c) use Adam with lr 1e-3 for 30 epochs. The test set is evaluated once, at the epoch with the best validation score.
- Every model in P1.4 is trained with gradient clipping at norm 1.0. The clip = 1.0 arm of P1.5(c) is the P1.4 RNN run at T = 200, so only the no-clip arm is trained separately.
- For P1.5(a) the trained library weights are copied into my own cells, which are unrolled step by step, because cuDNN does not give per-step hidden-state gradients.
- The parity checks are asserted in float64 on CPU. The float32 differences are printed.
- Task A uses patches of 8 time steps plus a CLS token, 2 pre-LN blocks and 4 heads. The language model uses AdamW (lr 1e-3, weight decay 0.01) with a 200-step warm-up and cosine decay, 5,000 steps, top-k 40 and the prompt `ROMEO:`. The heads and no-mask ablations share a 1,500-step budget and are only compared with each other.
- For the bonus, a 2-layer LSTM in the style of Problem 1 is compared with the Transformer on the same character task.

## Known issues
- P2.4, P2.5 and the bonus use a single seed, so they have no error bars. Small differences there, for example between head counts, are not treated as real.
- The timings were measured on a T4 and will differ on other hardware.
