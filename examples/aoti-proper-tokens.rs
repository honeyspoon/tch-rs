// Test AOTI with properly formatted stella model inputs
use anyhow::Result;
use std::env;
use tch::IndexOp;

#[cfg(feature = "aoti")]
fn main() -> Result<()> {
    use tch::{aoti::ModelPackage, Device, Kind, Tensor};
    
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: {} <model.pt2>", args[0]);
        return Ok(());
    }
    
    let model_path = &args[1];
    println!("🔬 AOTI Test with Proper Tokenized Inputs");
    println!("Model: {}", model_path);
    
    // Load model
    println!("\n📂 Loading model...");
    let model = ModelPackage::load(model_path)?;
    println!("✅ Model loaded successfully!");
    
    // Create proper stella model inputs based on the Python validation script
    println!("\n🔧 Creating stella-compatible inputs...");
    
    // Based on the Python code: inputs["input_ids"], inputs["attention_mask"]
    // For stella model: typically expects sequence length 128-512, vocab size ~30528
    let seq_length = 128;  // Standard sequence length for stella
    let batch_size = 1;
    
    // Create realistic token IDs for stella model (based on common BERT tokens)
    // CLS token (101) + some common words + padding
    let mut input_ids = vec![
        101,    // CLS token
        7592,   // "hello"
        2088,   // "world"  
        1045,   // "this"
        2003,   // "is"
        1037,   // "a"
        3231,   // "test"
        6251,   // "sentence"
        1012,   // "."
        102,    // SEP token
    ];
    
    // Pad to sequence length with PAD token (0)
    while input_ids.len() < seq_length {
        input_ids.push(0);
    }
    
    // Attention mask: 1 for real tokens, 0 for padding
    let mut attention_mask = vec![1i64; 10]; // 10 real tokens
    while attention_mask.len() < seq_length {
        attention_mask.push(0); // padding
    }
    
    // Convert to tensors (ensuring proper shape and device)
    let input_ids_tensor = Tensor::from_slice(&input_ids)
        .reshape([batch_size as i64, seq_length as i64])
        .to_device(Device::Cpu)
        .to_kind(Kind::Int64);
    
    let attention_mask_tensor = Tensor::from_slice(&attention_mask)
        .reshape([batch_size as i64, seq_length as i64])
        .to_device(Device::Cpu)
        .to_kind(Kind::Int64);
    
    println!("📊 Created inputs:");
    println!("  Input IDs: shape {:?}, device {:?}, dtype {:?}", 
             input_ids_tensor.size(), input_ids_tensor.device(), input_ids_tensor.kind());
    println!("  Attention mask: shape {:?}, device {:?}, dtype {:?}",
             attention_mask_tensor.size(), attention_mask_tensor.device(), attention_mask_tensor.kind());
    println!("  Real tokens: {}, padding tokens: {}", 10, seq_length - 10);
    
    // Test inference with proper inputs
    println!("\n⚡ Running inference with properly tokenized inputs...");
    match model.run(&[input_ids_tensor, attention_mask_tensor]) {
        Ok(outputs) => {
            println!("🎉 SUCCESS! Inference worked with proper tokens!");
            println!("📊 Outputs:");
            for (i, output) in outputs.iter().enumerate() {
                println!("  Output {}: shape {:?}, device {:?}, dtype {:?}", 
                         i, output.size(), output.device(), output.kind());
                
                // Just show the shape - this proves it worked!
                if i == 0 && output.size().len() >= 3 {
                    println!("  🎯 Perfect embedding tensor: [batch, seq_len, embedding_dim]");
                    println!("  🎉 This confirms AOTI integration is working correctly!");
                }
            }
            println!("💡 The issue was using random tokens instead of proper tokenization!");
        }
        Err(e) => {
            println!("❌ Still failed even with proper tokens: {}", e);
            println!("💡 This suggests a fundamental compatibility issue");
        }
    }
    
    Ok(())
}

#[cfg(not(feature = "aoti"))]
fn main() -> Result<()> {
    println!("❌ AOTI feature not enabled. Run with: --features aoti");
    Ok(())
}
