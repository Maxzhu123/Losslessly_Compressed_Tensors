"""Storage-lane mapping for shared Triton, TileLang, and pointwise kernels.

The default swizzle is ``lane ^ (row & 254)``. It mixes columns while
preserving lane parity and adjacent pairs; rows 2k and 2k+1 share a mask.
Linear mapping leaves each lane unchanged and requires statistically
interchangeable columns. Both mappings are their own inverse.
"""
import triton

from ... import config

LANE_MAPPING = config.lane_mapping


@triton.jit
def _swizzle_lane(row_base, step, lane):
    # Even mask: keeps lanes 2p/2p+1 adjacent so the decoder's store stays vectorisable.
    return lane ^ ((row_base + step) & 254)


@triton.jit
def _linear_lane(row_base, step, lane):
    return lane


if LANE_MAPPING == "linear":
    swizzle_lane = _linear_lane

    def swizzle_row_mask(row_index):
        return 0

    def swizzle_col(lane, row_index):
        return lane
else:
    swizzle_lane = _swizzle_lane

    def swizzle_row_mask(row_index):
        """Row shift for the TileLang C snippets, which apply ``(col + i) ^ shift``."""
        return row_index & 254

    def swizzle_col(lane, row_index):
        return lane ^ (row_index & 254)
