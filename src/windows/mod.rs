//! Windowing functions used to calculate taps

pub mod hamming;
pub mod blackman_harris;
pub mod flattop;
pub mod kaiser;
pub mod hann;
pub mod triangular;
pub mod rcostaper;
pub mod kaiser_bessel;

use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum WindowErrorCode{
    OutOfBounds,
    BetaLessThanZero,
    EmptyWindow,
    TriangularSubLength,
    TriangularZeroLength,
    TaperLength,
    OddLength
}

#[derive(Debug)]
pub struct WindowError(WindowErrorCode);

impl fmt::Display for WindowError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let error_code = match self.0 {
            WindowErrorCode::OutOfBounds => "Out of Bounds",
            WindowErrorCode::BetaLessThanZero => "Beta given is less than 0",
            WindowErrorCode::EmptyWindow => "Empty Window",
            WindowErrorCode::TriangularSubLength => "Sub length must be in Window Length + {-1,0,1}",
            WindowErrorCode::TriangularZeroLength => "Sub length must not be 0",
            WindowErrorCode::TaperLength => "Taper Length must not exceed Window Length / 2",
            WindowErrorCode::OddLength => "The Window Length Must Be Even"
        };
        write!(f, "Window Error: {}", error_code)
    }
}

impl Error for WindowError {}
