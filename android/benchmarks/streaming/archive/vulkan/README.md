# Historical Vulkan comparisons

These instrumentation sources produced the recorded v1, v2 and v3 comparisons.
They are preserved as evidence and are not part of the active Android test source
set. The current SDK accepts only the measured v2 bundle and defaults to its tiled
kernels, so running these older comparisons against it would be invalid.

To reproduce the original comparison code, use repository commit
`ad4ceda39ba6ab08c4fda0c4852f0a5bbbbbae44` with its corresponding pinned artifacts.
The current source set retains `VulkanProfileTest` for v2 diagnostics. Reproduction
requires separately prepared model artifacts; none of these files starts a cloud
test automatically.
