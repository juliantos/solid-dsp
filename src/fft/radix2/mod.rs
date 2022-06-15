use super::super::fft::{FFT, FFTError, FFTErrorCode, FFTDirection, FFTFlags, FFTType, FFTMethod, DataUnion, Radix2};
use super::super::resources::msb_index;

use std::mem::ManuallyDrop;

use num::complex::Complex;

fn fft_reverse_index(i: usize, n: usize) -> usize {
    let mut j = 0;
    let mut k = i;
    for _ in 0..n {
        j <<= 1;
        j |= k & 1;
        k >>= 1
    }
    j
}

pub fn create_radix2_fft_plan(nfft: usize, direction: FFTDirection, flags: FFTFlags) -> FFT {
    let log2 = (nfft as f64).log2();
    let r_log2 = log2.round();

    let m = msb_index(nfft) - 1;
    let d = if direction == FFTDirection::FORWARD { -1.0 } else { 1.0 };
    let mut reverse_indexes = Vec::new();
    let mut twiddle = Vec::new();
    for i in 0..nfft {
        reverse_indexes.push(fft_reverse_index(i, m));
        let exponent = Complex::from_polar(1.0, d * 2.0 * std::f64::consts::PI * (i as f64) / (nfft as f64));
        twiddle.push(exponent);
    }
    FFT {
        nfft: nfft,
        fft_type: if direction == FFTDirection::FORWARD { FFTType::FORWARD } else { FFTType::REVERSE },
        fft_direction: direction,
        fft_method: FFTMethod::RADIX2,
        fft_flags: flags,
        data: DataUnion { radix2: ManuallyDrop::new(Radix2 {
            m: m,
            reverse_indexes: reverse_indexes,
            twiddle: twiddle,
            log2: log2 == r_log2 && log2 > 1.0,
            execute_function: radix2_execute
        })}
    }
}

pub fn radix2_execute(fft: &FFT, input: &[Complex<f64>]) -> Result<Vec<Complex<f64>>, Box<dyn std::error::Error>> {
    let mut output = Vec::with_capacity(fft.nfft);
    unsafe{ output.set_len(fft.nfft) };

    if input.len() < fft.nfft {
        return Err(Box::new(FFTError(FFTErrorCode::NotEnoughBuffer)))
    } else if unsafe { !fft.data.radix2.log2 } {
        return Err(Box::new(FFTError(FFTErrorCode::RadixFFTNotMultipleOf2)))
    }

    let nfft = (fft.nfft >> 2) << 2;
    unsafe { 
        for i in (0..nfft).step_by(4) {
            output[i]   = input[fft.data.radix2.reverse_indexes[i]];
            output[i+1] = input[fft.data.radix2.reverse_indexes[i+1]];
            output[i+2] = input[fft.data.radix2.reverse_indexes[i+2]];
            output[i+3] = input[fft.data.radix2.reverse_indexes[i+3]];
        } 
    };

    let mut n1;
    let mut n2 = 1;
    let mut stride = nfft;
    let m = unsafe { fft.data.radix2.m };
    for _ in 0..m {
        n1 = n2;
        n2 *= 2;
        stride >>= 1;
        
        let mut twiddle_index = 0;

        for j in 0..n1 {
            let t = unsafe { fft.data.radix2.twiddle[twiddle_index] };
            twiddle_index = (twiddle_index + stride) % nfft;

            for k in (j..nfft).step_by(n2) {
                let y           = output[k + n1] * t;
                output[k + n1]  = output[k] - y;
                output[k]       = output[k] + y;
            }
        }
    }

    Ok(output)
}
