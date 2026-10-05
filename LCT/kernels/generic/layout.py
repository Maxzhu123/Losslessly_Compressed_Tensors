"""Storage-to-logical indexing shared by the generic Triton kernels."""

import triton


@triton.jit
def swizzle_lane(row_base, step, lane):
    """Map a storage lane to its logical column in the 256-lane codec layout.

    Generic encode, decode, overflow, and pointwise paths use this mapping.
    The logical row is row_base + step; keeping the block's row_base separate
    lets the compiler hoist its rotation out of decode loops.
    Other backends must use a compatible mapping when sharing compressed data.
    """
    row_shift = row_base & 255
    return (lane + ((row_shift + step) & 255)) & 255
