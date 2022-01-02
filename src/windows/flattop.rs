//! Flat Top Window

use super::{WindowError, WindowErrorCode};
use std::error::Error;


/// Function to calculate the value of the tap at `index` of `window_length`
/// 
/// Uses the flat top algorithm to generate taps in a window.
/// 
/// # Example
/// 
/// ```
/// use solid::windows::flattop;
/// 
/// let window_len = 25;
/// let mut h = vec![0.0; 25];
/// for i in 0..window_len {
///     h[i] = match flattop::flattop_f32(i, window_len) {
///         Ok(val) => val,
///         _ => 0.0
///     };
/// }
/// 
/// ```
pub fn flattop_f32(index: usize, window_length: usize) -> Result<f32, Box<dyn Error>> {
    if index > window_length {
        return Err(Box::new(WindowError(WindowErrorCode::OutOfBounds)))
    }

    let a0: f32 = 1.000;
    let a1: f32 = 1.930;
    let a2: f32 = 1.290;
    let a3: f32 = 0.388;
    let a4: f32 = 0.028;
    let t = 2.0 * std::f32::consts::PI * index as f32 / (window_length - 1) as f32;

    Ok(a0 - a1 * t.cos() + a2 * (2.0 * t).cos() - a3 * (3.0 * t).cos() + a4 * (4.0 * t).cos())
}

/// Function to calculate the value of the tap at `index` of `window_length`
/// 
/// Uses the flat top algorithm to generate taps in a window.
/// 
/// # Example
/// 
/// ```
/// use solid::windows::flattop;
/// 
/// let window_len = 25;
/// let mut h = vec![0.0; 25];
/// for i in 0..window_len {
///     h[i] = match flattop::flattop(i, window_len) {
///         Ok(val) => val,
///         _ => 0.0
///     };
/// }
/// 
/// ```
pub fn flattop(index: usize, window_length: usize) -> Result<f64, Box<dyn Error>> {
    if index > window_length {
        return Err(Box::new(WindowError(WindowErrorCode::OutOfBounds)))
    }

    let a0: f64 = 1.000;
    let a1: f64 = 1.930;
    let a2: f64 = 1.290;
    let a3: f64 = 0.388;
    let a4: f64 = 0.028;
    let t = 2.0 * std::f64::consts::PI * index as f64 / (window_length - 1) as f64;

    Ok(a0 - a1 * t.cos() + a2 * (2.0 * t).cos() - a3 * (3.0 * t).cos() + a4 * (4.0 * t).cos())
}