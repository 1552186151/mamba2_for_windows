import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


def main():
    print("=" * 60)
    print("PyTorch:", torch.__version__)
    print("Triton:", triton.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("Capability:", torch.cuda.get_device_capability(0))
    print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, cannot run Triton CUDA test.")

    n = 1024
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)

    block_size = 256
    grid = (triton.cdiv(n, block_size),)

    add_kernel[grid](x, y, out, n, BLOCK_SIZE=block_size)

    torch.cuda.synchronize()

    max_error = torch.max(torch.abs(out - (x + y))).item()

    print("Output shape:", tuple(out.shape))
    print("Max error:", max_error)

    if max_error < 1e-6:
        print("Triton CUDA test OK")
    else:
        raise RuntimeError("Triton result is incorrect.")


if __name__ == "__main__":
    main()