import httpx


def get_embedding(text: str, lm_studio_url: str, model: str) -> list[float]:
    r = httpx.post(
        f"{lm_studio_url}/embeddings",
        json={"model": model, "input": text},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]
