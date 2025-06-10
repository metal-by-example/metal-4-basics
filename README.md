# Getting Started with Metal 4

This repository contains the sample code accompanying the article ["Getting Started with Metal 4"](https://metalbyexample.com/metal-4) on MetalByExample.com.

The sample app renders a simple USDZ scene of a tropical island featuring a treasure chest showcasing physically based rendering with image-based lighting. It includes Metal 3 and Metal 4 renderers and is intended to demonstrate Metal 4's new pipeline creation APIs (including the new pipeline compiler interface) and command submission model (including the new command allocator interface). Additionally, it shows how to use argument buffers to bind numerous resources at once, along with how to use residency sets to make groups of resources resident on the GPU.

![A screenshot of the sample app](screenshots/metal4.png)

To run the sample app in Metal 4 mode, you should use Xcode 26 (Beta 2) or later and run on iOS 26 (Beta 3) or macOS 26 (Beta 3) or later. This sample may be subject to revision due to changes in the Metal 4 API, which is in beta as of this writing.
