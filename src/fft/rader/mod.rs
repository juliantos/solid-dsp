use super::super::fft::{DataUnion, FFTDirection, FFTFlags, FFTMethod, FFTType, Rader, FFT};
use super::super::resources::{modpow, primitive_root_prime};

use std::mem::ManuallyDrop;

use num::complex::Complex;

pub fn create_rader_plan(nfft: usize, direction: FFTDirection, flags: FFTFlags) -> FFT {
    let g = primitive_root_prime(nfft);
    let mut seq = Vec::new();
    for i in 0..(nfft - 1) {
        seq.push(modpow(g, i + 1, nfft));
    }

    let d = if direction == FFTDirection::FORWARD {
        -1.0
    } else {
        1.0
    };
    let mut time_domain_buffer = Vec::new();
    for item in seq.iter().take(nfft - 1) {
        let exponent = Complex::from_polar(
            1.0,
            d * 2.0 * std::f64::consts::PI * (*item as f64) / (nfft as f64),
        );
        time_domain_buffer.push(exponent)
    }

    let fft = FFT::new(nfft - 1, FFTDirection::FORWARD, flags);
    let ifft = FFT::new(nfft - 1, FFTDirection::REVERSE, flags);

    let freq_domain_buffer = fft.execute(&time_domain_buffer).unwrap_or_default();

    FFT {
        nfft,
        fft_type: if direction == FFTDirection::FORWARD {
            FFTType::FORWARD
        } else {
            FFTType::REVERSE
        },
        fft_direction: direction,
        fft_method: FFTMethod::RADER,
        fft_flags: flags,
        data: DataUnion {
            rader: ManuallyDrop::new(Rader {
                seq,
                dft: freq_domain_buffer,
                fft: Box::new(fft),
                ifft: Box::new(ifft),
                execute_function: rader_execute,
            }),
        },
    }
}

pub fn rader_execute(
    fft: &FFT,
    input: &[Complex<f64>],
) -> Result<Vec<Complex<f64>>, Box<dyn std::error::Error>> {
    let mut output = Vec::with_capacity(fft.nfft);
    let _remaining = output.spare_capacity_mut();
    unsafe { output.set_len(fft.nfft) };

    let mut time_domain_buffer = Vec::new();
    for i in 0..(fft.nfft - 1) {
        let k = unsafe { fft.data.rader.seq[fft.nfft - i - 2] };
        time_domain_buffer.push(input[k]);
    }

    let mut freq_domain_buffer = unsafe { fft.data.rader.fft.execute(&time_domain_buffer)? };

    freq_domain_buffer = unsafe {
        freq_domain_buffer
            .iter()
            .zip(fft.data.rader.dft.iter())
            .map(|(&x, &y)| x * y)
            .collect()
    };

    time_domain_buffer = unsafe { fft.data.rader.ifft.execute(&freq_domain_buffer)? };

    output[0] = input[0..fft.nfft].iter().sum();

    for (i, item) in time_domain_buffer.iter().enumerate().take(fft.nfft - 1) {
        let k = unsafe { fft.data.rader.seq[i] };
        output[k] = (*item / (fft.nfft - 1) as f64) + input[0];
    }

    Ok(output)
}
