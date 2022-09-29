//! Kaiser-Bessel Derived window
use super::kaiser::kaiser;
use super::{WindowError, WindowErrorCode};

use std::error::Error;

/// Function to calculate the value of the tap at `index` of `window_length`
///
/// Uses the Kaiser-Bessel Derived formula to generate taps in the window
///
/// # Example
///
/// ```
/// use solid::windows::kaiser_bessel;
/// use solid::filter::firdes::kaiser_beta;
///
/// let window_len = 24;
/// let mut h = vec![0.0; window_len];
/// let beta = kaiser_beta(0.35);
/// for i in 0..window_len {
///     h[i] = match kaiser_bessel::kaiser_bessel(i, window_len, beta) {
///         Ok(val) => val,
///         _ => -10.0
///     };
///     assert_ne!(h[i], -10.0);
/// }
///
/// ```
pub fn kaiser_bessel(index: usize, window_length: usize, beta: f64) -> Result<f64, Box<dyn Error>> {
    if index > window_length {
        return Err(Box::new(WindowError(WindowErrorCode::OutOfBounds)));
    } else if window_length == 0 {
        return Err(Box::new(WindowError(WindowErrorCode::EmptyWindow)));
    } else if window_length % 2 == 1 {
        return Err(Box::new(WindowError(WindowErrorCode::OddLength)));
    }

    let m = window_length / 2;
    if index >= m {
        return kaiser_bessel(window_length - index - 1, window_length, beta);
    }

    let mut w0: f64 = 0.0;
    let mut w1: f64 = 0.0;
    for i in 0..=m {
        let w = kaiser(i, m + 1, beta)?;
        w1 += w;
        if i <= index {
            w0 += w;
        }
    }

    Ok((w0 / w1).sqrt())
}
