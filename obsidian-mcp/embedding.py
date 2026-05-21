import httpx


async def embed(text: str, lm_studio_url: str, model: str) -> list[float]:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{lm_studio_url}/embeddings",
            json={"model": model, "input": text},
        )
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]
