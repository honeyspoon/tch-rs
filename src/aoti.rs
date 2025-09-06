//! AOTInductor (Ahead-of-Time compilation) support for PyTorch models.
//!
//! This module provides functionality to load and run pre-compiled PyTorch models
//! using PyTorch's AOTInductor. These models are typically saved as `.pt2` files
//! and offer improved inference performance.

use crate::{TchError, Tensor};
use std::ffi::CString;

/// A model package containing a pre-compiled PyTorch model.
///
/// # Example
///
/// ```no_run
/// use tch::{Tensor, aoti::ModelPackage};
///
/// let model = ModelPackage::load("model.pt2")?;
/// let input = Tensor::randn([1, 3, 224, 224], tch::kind::FLOAT_CPU);
/// let outputs = model.run(&[input])?;
/// # Ok::<(), tch::TchError>(())
/// ```
pub struct ModelPackage {
    loader: *mut torch_sys::C_aoti_model_package_loader,
}

impl ModelPackage {
    /// Load a pre-compiled model from a `.pt2` file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the `.pt2` model file
    ///
    /// # Returns
    ///
    /// A `ModelPackage` that can be used for inference, or a `TchError` if loading fails.
    pub fn load<P: AsRef<str>>(path: P) -> Result<Self, TchError> {
        let path = CString::new(path.as_ref())?;
        let loader = unsafe { torch_sys::aoti_load(path.as_ptr()) };

        if loader.is_null() {
            let c_error = unsafe { torch_sys::get_and_reset_last_err() };
            let error = if c_error.is_null() {
                "Failed to load AOT model".to_string()
            } else {
                unsafe {
                    let error_str =
                        std::ffi::CStr::from_ptr(c_error).to_string_lossy().into_owned();
                    libc::free(c_error as *mut libc::c_void);
                    error_str
                }
            };
            return Err(TchError::Torch(error));
        }

        Ok(ModelPackage { loader })
    }

    /// Run inference on the model.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Slice of input tensors for the model
    ///
    /// # Returns
    ///
    /// A vector of output tensors, or a `TchError` if inference fails.
    pub fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, TchError> {
        if inputs.is_empty() {
            return Err(TchError::Torch("No input tensors provided".into()));
        }

        // Convert input tensors to raw pointers
        let input_ptrs: Vec<*mut torch_sys::C_tensor> =
            inputs.iter().map(|t| t.as_ptr() as *mut torch_sys::C_tensor).collect();

        let mut n_outputs: libc::c_int = 0;

        let output_ptrs = unsafe {
            torch_sys::aoti_run(
                self.loader,
                input_ptrs.as_ptr() as *mut *mut torch_sys::C_tensor,
                inputs.len() as libc::c_int,
                &mut n_outputs,
            )
        };

        if output_ptrs.is_null() {
            let c_error = unsafe { torch_sys::get_and_reset_last_err() };
            let error = if c_error.is_null() {
                "Failed to run AOT model inference".to_string()
            } else {
                unsafe {
                    let error_str =
                        std::ffi::CStr::from_ptr(c_error).to_string_lossy().into_owned();
                    libc::free(c_error as *mut libc::c_void);
                    error_str
                }
            };
            return Err(TchError::Torch(error));
        }

        // Convert output tensor pointers to Tensor objects
        let mut outputs = Vec::with_capacity(n_outputs as usize);
        unsafe {
            for i in 0..n_outputs as isize {
                let tensor_ptr = *output_ptrs.offset(i);
                if !tensor_ptr.is_null() {
                    outputs.push(crate::wrappers::tensor::Tensor::from_ptr(tensor_ptr));
                }
            }
            // Free the array of pointers (but not the individual tensors)
            libc::free(output_ptrs as *mut libc::c_void);
        }

        Ok(outputs)
    }
}

impl Drop for ModelPackage {
    fn drop(&mut self) {
        unsafe {
            if !self.loader.is_null() {
                torch_sys::aoti_free(self.loader);
            }
        }
    }
}

// ModelPackage is safe to send between threads
unsafe impl Send for ModelPackage {}
unsafe impl Sync for ModelPackage {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_nonexistent_file() {
        let result = ModelPackage::load("nonexistent.pt2");
        assert!(result.is_err());
    }
}
