from langchain_core.embeddings import Embeddings


def make_embedder(
    provider: str,
    model: str,
    lm_studio_url: str = "",
    openai_api_key: str = "",
    gemini_api_key: str = "",
) -> Embeddings:
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model, api_key=openai_api_key or None)

    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=model, google_api_key=gemini_api_key or None)

    # default: lm_studio (OpenAI-compatible local server)
    # check_embedding_ctx_length=False prevents LangChain from pre-tokenizing text
    # into token-ID arrays, which LM Studio doesn't accept — it needs raw strings.
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=model,
        openai_api_base=lm_studio_url or "http://host.docker.internal:1234/v1",
        openai_api_key="lm-studio",
        check_embedding_ctx_length=False,
    )
