import os
import numpy as np
import pandas as pd

OUT_PATH = "data/raw/social_media_engagement_dataset.csv"


def main():
    os.makedirs("data/raw", exist_ok=True)
    rng = np.random.default_rng(42)

    platforms = ["Instagram", "Facebook", "Twitter"]
    post_types = ["Text", "Image", "Video"]

    n = 2000
    df = pd.DataFrame({
        "platform": rng.choice(platforms, size=n),
        "post_type": rng.choice(post_types, size=n),
        "post_length": rng.integers(5, 300, size=n),
    })

    # synthetic engagement_rate with some signal
    base = rng.normal(0.04, 0.02, size=n)
    boost = (df["platform"].eq("Instagram") * 0.01) + (df["post_type"].eq("Video") * 0.015)
    df["engagement_rate"] = (base + boost).clip(0, 0.25)

    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote sample dataset to {OUT_PATH}")


if __name__ == "__main__":
    main()
