pub mod dft;
pub mod mixed_radix;
pub mod rader;
pub mod rader2;
pub mod radix2;

use super::dot_product::DotProduct;

use std::error::Error;
use std::fmt;
use std::mem::ManuallyDrop;

use num::complex::Complex;
use slow_primes::is_prime_miller_rabin;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum FFTDirection {
    FORWARD,
    REVERSE,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum FFTType {
    DEFAULT,
    FORWARD,
    REVERSE,
    REDFT00,
    REDFT01,
    REDFT10,
    REDFT11,
    RODFT00,
    RODFT01,
    RODFT10,
    RODFT11,
    MDCT,
    IMDCT,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum FFTMethod {
    DEFAULT,
    RADIX2,
    MIXEDRADIX,
    RADER,
    RADER2,
    DFT,
    UNKNOWN,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum FFTFlags {
    ESTIMATE,
    MEASURE,
}

type FFTExecuteFunction =
    fn(&FFT, input: &[Complex<f64>]) -> Result<Vec<Complex<f64>>, Box<dyn Error>>;

struct Dft {
    #[allow(dead_code)]
    twiddle: Vec<Complex<f64>>,
    dot_products: Vec<DotProduct<Complex<f64>>>,
    execute_function: FFTExecuteFunction,
}

struct Radix2 {
    twiddle: Vec<Complex<f64>>,
    reverse_indexes: Vec<usize>,
    m: usize,
    log2: bool,
    execute_function: FFTExecuteFunction,
}

struct MixedRadix {
    p: usize,
    q: usize,
    twiddle: Vec<Complex<f64>>,
    p_fft: Box<FFT>,
    q_fft: Box<FFT>,
    execute_function: FFTExecuteFunction,
}

struct Rader {
    seq: Vec<usize>,
    dft: Vec<Complex<f64>>,
    fft: Box<FFT>,
    ifft: Box<FFT>,
    execute_function: FFTExecuteFunction,
}

struct Rader2 {
    seq: Vec<usize>,
    nfft_prime: usize,
    dft: Vec<Complex<f64>>,
    fft: Box<FFT>,
    ifft: Box<FFT>,
    execute_function: FFTExecuteFunction,
}

union DataUnion {
    dft: ManuallyDrop<Dft>,
    radix2: ManuallyDrop<Radix2>,
    mixed_radix: ManuallyDrop<MixedRadix>,
    rader: ManuallyDrop<Rader>,
    rader2: ManuallyDrop<Rader2>,
}

fn fft_is_radix2(nfft: usize) -> bool {
    let mut d = 0;
    let mut m = 0;
    let mut t = nfft;
    for i in 0..(std::mem::size_of::<[usize; 8]>()) {
        d += t & 1;
        if !m > 0 && (t & 0x1) > 0 {
            m = i;
        };
        t >>= 1;
    }

    d == 1
}

fn fft_estimate_method(nfft: usize) -> FFTMethod {
    let method: FFTMethod;

    if nfft == 0 {
        method = FFTMethod::UNKNOWN
    } else if nfft <= 8 || nfft == 11 || nfft == 13 || nfft == 16 || nfft == 17 {
        method = FFTMethod::DFT
    } else if fft_is_radix2(nfft) {
        method = FFTMethod::MIXEDRADIX
    } else if is_prime_miller_rabin(nfft as u64) {
        if fft_is_radix2(nfft - 1) {
            method = FFTMethod::RADER
        } else {
            method = FFTMethod::RADER2
        }
    } else {
        method = FFTMethod::MIXEDRADIX
    }

    method
}

#[derive(Debug, PartialEq)]
enum FFTErrorCode {
    NotEnoughBuffer,
    RadixFFTNotMultipleOf2,
    BadExecuteMethod,
}

#[derive(Debug)]
pub struct FFTError(FFTErrorCode);

impl fmt::Display for FFTError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "FFT Error: {:?}", self.0)
    }
}

impl Error for FFTError {}

#[allow(dead_code)]
pub struct FFT {
    nfft: usize,
    fft_direction: FFTDirection,
    fft_type: FFTType,
    fft_method: FFTMethod,
    fft_flags: FFTFlags,
    data: DataUnion,
}

// TODO[epic=fast]  Add Option for FFTW3 Bingings
impl FFT {
    pub fn new(nfft: usize, direction: FFTDirection, flags: FFTFlags) -> Self {
        let method = fft_estimate_method(nfft);

        match method {
            FFTMethod::RADIX2 => radix2::create_radix2_fft_plan(nfft, direction, flags),
            FFTMethod::DFT => dft::create_dft_plan(nfft, direction, flags),
            FFTMethod::MIXEDRADIX => mixed_radix::create_mixed_radix_plan(nfft, direction, flags),
            FFTMethod::RADER => rader::create_rader_plan(nfft, direction, flags),
            FFTMethod::RADER2 => rader2::create_rader2_plan(nfft, direction, flags),
            _ => mixed_radix::create_mixed_radix_plan(nfft, direction, flags),
        }
    }

    pub fn execute(&self, input: &[Complex<f64>]) -> Result<Vec<Complex<f64>>, Box<dyn Error>> {
        let output = match self.fft_method {
            FFTMethod::RADIX2 => {
                let execute_fn = unsafe { self.data.radix2.execute_function };
                execute_fn(self, input)?
            }
            FFTMethod::DFT => {
                let execute_fn = unsafe { self.data.dft.execute_function };
                execute_fn(self, input)?
            }
            FFTMethod::MIXEDRADIX => {
                let execute_fn = unsafe { self.data.mixed_radix.execute_function };
                execute_fn(self, input)?
            }
            FFTMethod::RADER => {
                let execute_fn = unsafe { self.data.rader.execute_function };
                execute_fn(self, input)?
            }
            FFTMethod::RADER2 => {
                let execute_fn = unsafe { self.data.rader2.execute_function };
                execute_fn(self, input)?
            }
            _ => return Err(Box::new(FFTError(FFTErrorCode::BadExecuteMethod))),
        };

        Ok(output)
    }
}

impl fmt::Display for FFT {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "FFT Plan [{:?}] [n={}] [{:?}] [type={:?}]",
            self.fft_direction, self.nfft, self.fft_method, self.fft_type
        )?;
        match self.fft_method {
            FFTMethod::MIXEDRADIX => {
                unsafe {
                    writeln!(
                        f,
                        " [P={}, Q={}]",
                        self.data.mixed_radix.p, self.data.mixed_radix.q
                    )?
                };
                unsafe { write!(f, "PFFT:{}", self.data.mixed_radix.p_fft)? };
                unsafe { write!(f, "QFFT:{}", self.data.mixed_radix.q_fft) }
            }
            FFTMethod::RADER => {
                writeln!(f)?;
                unsafe { write!(f, "FFT:{}", self.data.rader.fft)? };
                unsafe { write!(f, "IFFT:{}", self.data.rader.ifft) }
            }
            FFTMethod::RADER2 => {
                unsafe { writeln!(f, "[Prime={}]", self.data.rader2.nfft_prime)? };
                unsafe { write!(f, "FFT:{}", self.data.rader2.fft)? };
                unsafe { write!(f, "IFFT:{}", self.data.rader2.ifft) }
            }
            _ => {
                writeln!(f)
            }
        }
    }
}
