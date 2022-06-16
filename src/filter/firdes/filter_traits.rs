use super::super::fir_filter::FIRFilter;
use super::{filter_autocorrelation, filter_crosscorrelation, filter_isi, filter_energy};

pub trait Firdes {
    type Output;

    fn autocorrelation(&self, lag: isize) -> Self::Output;
    fn crosscorrelation(&self, rhs: Self, lag: isize) -> Self::Output;
    fn isi(&self, sps: usize, delay: usize) -> (Self::Output, Self::Output);
    fn energy(&self, fc: Self::Output, fft_size: usize) -> Self::Output;
}

impl<T: Copy> Firdes for FIRFilter<f64, T> {
    type Output = f64;

    fn autocorrelation(&self, lag: isize) -> Self::Output {
        filter_autocorrelation(self.coefficients(), lag)
    }

    fn crosscorrelation(&self, rhs: Self, lag: isize) -> Self::Output {
        filter_crosscorrelation(self.coefficients(), rhs.coefficients(), lag)
    }

    fn isi(&self, sps: usize, delay: usize) -> (Self::Output, Self::Output) {
        filter_isi(self.coefficients(), sps, delay)
    }

    fn energy(&self, fc: Self::Output, fft_size: usize) -> Self::Output {
        match filter_energy(self.coefficients(), fc, fft_size) {
            Ok(energy) => energy,
            Err(e) => {
                if cfg!(debug_assertion) {
                    println!("{}", e)
                }
                0.0
            }
        }
    }
}