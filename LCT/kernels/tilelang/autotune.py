"""Launch configurations for native TileLang codec autotuning."""

# Retain the three measured winners and two lower-thread-count alternatives.
# Larger tiles reduce barriers; 128 threads give each thread two streams.
DECODE_AUTOTUNE_CONFIGS = (
    {"threads": 256, "row_tile": 8, "unroll": 4},
    {"threads": 256, "row_tile": 16, "unroll": 8},
    {"threads": 256, "row_tile": 32, "unroll": 8},
    {"threads": 128, "row_tile": 8, "unroll": 4},
    {"threads": 128, "row_tile": 32, "unroll": 8},
)

# Keep both measured scalar winners, both transfer widths, and a 128-thread option.
ENCODE_AUTOTUNE_CONFIGS = (
    {"threads": 256, "row_tile": 16, "unroll": 8, "stage_fields": False, "vector_width": 8},
    {"threads": 256, "row_tile": 32, "unroll": 8, "stage_fields": False, "vector_width": 8},
    {"threads": 256, "row_tile": 16, "unroll": 4, "stage_fields": False, "vector_width": 4},
    {"threads": 256, "row_tile": 16, "unroll": 4, "stage_fields": True, "vector_width": 8},
    {"threads": 128, "row_tile": 32, "unroll": 8, "stage_fields": True, "vector_width": 8},
)


def encode_autotune_configs(*inputs: object, precomputed: bool = False,
                            vector_aligned: bool = False, **kwargs: object) -> list[dict[str, int | bool]]:
    """Remove candidates that compile identically for the active input layout."""
    configs, seen = [], set()
    for config in ENCODE_AUTOTUNE_CONFIGS:
        key = (config["threads"], config["row_tile"], config["unroll"],
               config["vector_width"] if vector_aligned else 0,
               config["stage_fields"] if vector_aligned and not precomputed else False)
        if key not in seen:
            seen.add(key)
            configs.append(config)
    return configs
