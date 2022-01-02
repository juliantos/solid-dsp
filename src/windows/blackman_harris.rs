//! Blackman harris Window
//! 
//! frederic j. harris, "On the Use of Windows for Harmonic
//! Analysis with the Discrete Fourier Transform," Proceedings of the
//! IEEE, vol. 66, no. 1, January, 1978.
use super::{WindowError, WindowErrorCode};

use std::error::Error;

/// Function to calculate the value of the tap at `index` of `window_length`
/// 
/// Uses the 4th order Blackman harris approach to generate taps in a window.
/// 
/// # Example
/// 
/// ```
/// use solid::windows::blackman_harris;
/// 
/// let window_len = 25;
/// let mut h = vec![0.0; 25];
/// for i in 0..window_len {
///     h[i] = match blackman_harris::blackman_harris_f32(i, window_len) {
///         Ok(val) => val,
///         _ => 0.0
///     };
/// }
/// 
/// ```
pub fn blackman_harris_f32(index: usize, window_length: usize) -> Result<f32, Box<dyn Error>> {
    if index > window_length {
        return Err(Box::new(WindowError(WindowErrorCode::OutOfBounds)))
    }

    let a0: f32 = 0.35875;
    let a1: f32 = 0.48829;
    let a2: f32 = 0.14128;
    let a3: f32 = 0.01168;

    let t = (2.0 * std::f32::consts::PI * index as f32) / (window_length - 1) as f32;

    Ok(a0 - a1 * t.cos() + a2 * (2.0 * t).cos() - a3 * (3.0 * t).cos())
}

/// Function to calculate the value of the tap at `index` of `window_length`
/// 
/// Uses the 4th order Blackman harris approach to generate taps in a window.
/// 
/// # Example
/// 
/// ```
/// use solid::windows::blackman_harris;
/// 
/// let window_len = 25;
/// let mut h = vec![0.0; 25];
/// for i in 0..window_len {
///     h[i] = match blackman_harris::blackman_harris(i, window_len) {
///         Ok(val) => val,
///         _ => 0.0
///     };
/// }
/// 
/// ```
pub fn blackman_harris(index: usize, window_length: usize) -> Result<f64, Box<dyn Error>> {
    if index > window_length {
        return Err(Box::new(WindowError(WindowErrorCode::OutOfBounds)))
    }

    let a0: f64 = 0.35875;
    let a1: f64 = 0.48829;
    let a2: f64 = 0.14128;
    let a3: f64 = 0.01168;

    let t = 2.0 * std::f64::consts::PI * index as f64 / (window_length - 1) as f64;

    Ok(a0 - a1 * t.cos() + a2 * (2.0 * t).cos() - a3 * (3.0 * t).cos())
}

/// Function to calculate the value of the tap at `index` of `window_length`
/// 
/// Uses the 7th order Blackman harris approach to generate taps in a window.
/// 
/// # Example
/// 
/// ```
/// use solid::windows::blackman_harris;
/// 
/// let window_len = 25;
/// let mut h = vec![0.0; 25];
/// for i in 0..window_len {
///     h[i] = match blackman_harris::blackman_harris7_f32(i, window_len) {
///         Ok(val) => val,
///         _ => 0.0
///     };
/// }
/// 
/// ```
pub fn blackman_harris7_f32(index: usize, window_length: usize) -> Result<f32, Box<dyn Error>> {
    if index > window_length {
        return Err(Box::new(WindowError(WindowErrorCode::OutOfBounds)))
    }

	let a0: f32 = 0.27105;
	let a1: f32 = 0.43329;
	let a2: f32 = 0.21812;
	let a3: f32 = 0.06592;
	let a4: f32 = 0.01081;
	let a5: f32 = 0.00077;
	let a6: f32 = 0.00001;

    let t = 2.0 * std::f32::consts::PI * index as f32 / (window_length - 1) as f32;

    Ok(a0 - a1 * t.cos() + a2 * (2.0 * t).cos() - a3 * (3.0 * t).cos() + 
        a4 * (4.0 * t).cos() - a5 * (5.0 * t).cos() + a6 * (6.0 * t).cos())
}

/// Function to calculate the value of the tap at `index` of `window_length`
/// 
/// Uses the 7th order Blackman harris approach to generate taps in a window.
/// 
/// # Example
/// 
/// ```
/// use solid::windows::blackman_harris;
/// 
/// let window_len = 25;
/// let mut h = vec![0.0; 25];
/// for i in 0..window_len {
///     h[i] = match blackman_harris::blackman_harris7(i, window_len) {
///         Ok(val) => val,
///         _ => 0.0
///     };
/// }
/// 
/// ```
pub fn blackman_harris7(index: usize, window_length: usize) -> Result<f64, Box<dyn Error>> {
    if index > window_length {
        return Err(Box::new(WindowError(WindowErrorCode::OutOfBounds)))
    }

	let a0: f64 = 0.27105;
	let a1: f64 = 0.43329;
	let a2: f64 = 0.21812;
	let a3: f64 = 0.06592;
	let a4: f64 = 0.01081;
	let a5: f64 = 0.00077;
	let a6: f64 = 0.00001;

    let t = 2.0 * std::f64::consts::PI * index as f64 / (window_length - 1) as f64;

    Ok(a0 - a1 * t.cos() + a2 * (2.0 * t).cos() - a3 * (3.0 * t).cos() + 
        a4 * (4.0 * t).cos() - a5 * (5.0 * t).cos() + a6 * (6.0 * t).cos())
}