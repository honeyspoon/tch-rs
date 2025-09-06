//! Example of using AOTInductor for optimized PyTorch model inference.
//!
//! This example demonstrates how to:
//! 1. Load a pre-compiled PyTorch model (.pt2 file)
//! 2. Run inference with the model
//! 3. Process the outputs
//!
//! To generate a .pt2 file, you can use Python:
//! ```python
//! import torch
//! import torch.nn as nn
//!
//! # Define a simple model
//! class SimpleModel(nn.Module):
//!     def __init__(self):
//!         super().__init__()
//!         self.linear = nn.Linear(10, 5)
//!     
//!     def forward(self, x):
//!         return torch.relu(self.linear(x))
//!
//! # Create model and example input
//! model = SimpleModel()
//! example_input = torch.randn(1, 10)
//!
//! # Compile and export
//! compiled_model = torch.compile(model, backend="aot_eager")
//! torch.export.export(compiled_model, (example_input,)).save("simple_model.pt2")
//! ```

use anyhow::Result;
use std::env;

#[cfg(feature = "aoti")]
fn main() -> Result<()> {
    use tch::{aoti::ModelPackage, Device, Kind, Tensor};

    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        println!("Usage: {} <model.pt2>", args[0]);
        println!("Please provide the path to a .pt2 model file");
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("Loading AOT model from: {}", model_path);

    // Load the pre-compiled model
    let model = ModelPackage::load(model_path)?;
    println!("Model loaded successfully!");

    // Create some example input tensors
    // Note: Adjust dimensions based on your specific model requirements
    let input = Tensor::randn([1, 10], (Kind::Float, Device::Cpu));
    println!("Input tensor shape: {:?}", input.size());

    // Run inference
    println!("Running inference...");
    let outputs = model.run(&[input])?;

    println!("Inference completed!");
    println!("Number of outputs: {}", outputs.len());

    for (i, output) in outputs.iter().enumerate() {
        println!("Output {}: shape {:?}", i, output.size());
        output.print();
    }

    Ok(())
}

#[cfg(not(feature = "aoti"))]
fn main() {
    println!("This example requires the 'aoti' feature to be enabled.");
    println!("Run with: cargo run --example aot-inference --features aoti");
    std::process::exit(1);
}
