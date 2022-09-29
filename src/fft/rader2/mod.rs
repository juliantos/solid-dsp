use super::super::fft::{DataUnion, FFTDirection, FFTFlags, FFTMethod, FFTType, Rader2, FFT};
use super::super::resources::{modpow, primitive_root_prime};

use std::mem::ManuallyDrop;

use num::complex::Complex;

pub fn create_rader2_plan(nfft: usize, direction: FFTDirection, flags: FFTFlags) -> FFT {
    let g = primitive_root_prime(nfft);
    let mut seq = Vec::new();
    for i in 0..(nfft - 1) {
        seq.push(modpow(g, i + 1, nfft));
    }

    let mut nfft_prime = (2 * nfft - 4) - 1;
    let mut m = 0;
    while nfft_prime > 0 {
        nfft_prime >>= 1;
        m += 1;
    }
    nfft_prime = 1 << m;

    let d = if direction == FFTDirection::FORWARD {
        -1.0
    } else {
        1.0
    };
    let mut time_domain_buffer = Vec::new();
    for i in 0..(nfft_prime) {
        let exponent = Complex::from_polar(
            1.0,
            d * 2.0 * std::f64::consts::PI * (seq[i % (nfft - 1)] as f64) / (nfft as f64),
        );
        time_domain_buffer.push(exponent)
    }

    let fft = FFT::new(nfft_prime, FFTDirection::FORWARD, flags);
    let ifft = FFT::new(nfft_prime, FFTDirection::REVERSE, flags);

    let freq_domain_buffer = fft.execute(&time_domain_buffer).unwrap_or_default();

    FFT {
        nfft,
        fft_type: if direction == FFTDirection::FORWARD {
            FFTType::FORWARD
        } else {
            FFTType::REVERSE
        },
        fft_direction: direction,
        fft_method: FFTMethod::RADER2,
        fft_flags: flags,
        data: DataUnion {
            rader2: ManuallyDrop::new(Rader2 {
                seq,
                nfft_prime,
                dft: freq_domain_buffer,
                fft: Box::new(fft),
                ifft: Box::new(ifft),
                execute_function: rader2_execute,
            }),
        },
    }
}

pub fn rader2_execute(
    fft: &FFT,
    input: &[Complex<f64>],
) -> Result<Vec<Complex<f64>>, Box<dyn std::error::Error>> {
    let mut output = Vec::with_capacity(fft.nfft);
    let _remaining = output.spare_capacity_mut();
    unsafe { output.set_len(fft.nfft) };

    let nfft_prime = unsafe { fft.data.rader2.nfft_prime };
    let seq = unsafe { &fft.data.rader2.seq };

    let mut x_prime = vec![Complex::new(0.0, 0.0); nfft_prime];

    x_prime[0] = input[seq[fft.nfft - 2]];
    for i in 1..(fft.nfft - 1) {
        let k = seq[fft.nfft - i - 2];
        x_prime[i + nfft_prime - fft.nfft + 1] = input[k];
    }

    let mut xi_prime = unsafe { fft.data.rader2.fft.execute(&x_prime)? };

    xi_prime = unsafe {
        xi_prime
            .iter()
            .zip(fft.data.rader2.dft.iter())
            .map(|(&x, &y)| x * y)
            .collect()
    };

    x_prime = unsafe { fft.data.rader2.ifft.execute(&xi_prime)? };

    output[0] = input[0..fft.nfft].iter().sum();

    for i in 0..(fft.nfft - 1) {
        let k = seq[i];
        output[k] = (x_prime[i] / (nfft_prime) as f64) + input[0];
    }

    Ok(output)
}
