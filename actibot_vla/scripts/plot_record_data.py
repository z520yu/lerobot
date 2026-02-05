#!/usr/bin/env python3
import os, re, glob, argparse
import numpy as np
import matplotlib.pyplot as plt

def load_flat(file):
    arr = np.load(file, allow_pickle=True)
    return arr.item()  # 扁平dict

def natural_sort_key(p):
    m = re.search(r"step_(\d+)\.npy$", os.path.basename(p))
    return int(m.group(1)) if m else p

def collect(record_dir, take_index=0):
    files = sorted(glob.glob(os.path.join(record_dir, "step_*.npy")), key=natural_sort_key)
    first_actions, states, steps = [], [], []
    for f in files:
        d = load_flat(f)
        a = d["outputs/actions"]  # (H, D)
        s = d.get("outputs/state", d.get("inputs/state"))
        first_actions.append(a[take_index])  # 取第 take_index 个horizon动作，默认0
        states.append(s)
        steps.append(natural_sort_key(f))
    return np.asarray(first_actions), np.asarray(states), np.asarray(steps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="policy_records 目录")
    ap.add_argument("--hidx", type=int, default=0, help="取第几个horizon动作（默认0）")
    ap.add_argument("--dims", type=str, default="", help="只画这些维度, 逗号分隔，如 '0,1,6'")
    args = ap.parse_args()

    A, S, steps = collect(args.dir, take_index=args.hidx)  # A: [N_steps, D]
    D = A.shape[1]
    dims = [int(x) for x in args.dims.split(",") if x.strip()] if args.dims else list(range(D))

    # 折线：每个维度一条线，展示动作随 step 变化
    plt.figure(figsize=(12, 6))
    for d in dims:
        plt.plot(steps, A[:, d], label=f"a{d}")
    plt.xlabel("step")
    plt.ylabel("action value")
    plt.title(f"Action[{args.hidx}] over steps")
    plt.legend(ncol=8, fontsize=8)
    plt.grid(True, alpha=0.3)

    # 热力图：动作维度 x step
    plt.figure(figsize=(12, 5))
    plt.imshow(A.T, aspect="auto", interpolation="nearest", cmap="coolwarm")
    plt.colorbar(label="action value")
    plt.yticks(range(D), [f"a{d}" for d in range(D)])
    plt.xticks(range(0, len(steps), max(1, len(steps)//10)), steps[::max(1, len(steps)//10)])
    plt.title(f"Action[{args.hidx}] heatmap (dims x steps)")

    # 如需对比“相对/绝对”，再画 (A - S[:,:D])（仅当你知道哪些维度是delta）
    if S.ndim == 2 and S.shape[1] >= D:
        # 例：前6关节为delta，最后1为绝对（常见：6关节+1夹爪）
        delta_mask = np.array([True]*min(6, D) + [False]*(D-min(6, D)))
        resid = np.where(delta_mask, A - S[:, :D], np.nan)
        plt.figure(figsize=(12, 6))
        for d in np.where(delta_mask)[0]:
            plt.plot(steps, resid[:, d], label=f"delta(a{d}) = action - state")
        plt.xlabel("step"); plt.ylabel("delta")
        plt.title("Estimated delta (assuming first 6 dims are delta)")
        plt.legend(ncol=6, fontsize=8); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()