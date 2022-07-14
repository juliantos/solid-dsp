use solid::filter::firdes;
use solid::filter::iirdes;
use solid::filter::fir_filter::FIRFilter;
use solid::filter::filter::Filter;
use solid::filter::iir_filter::{IIRFilter, IIRFilterType};
use solid::filter::auto_correlator::AutoCorrelator;
use solid::circular_buffer::CircularBuffer;
use solid::auto_gain_control::AGC;
use solid::nco::NCO;
use solid::fft::{FFT, FFTDirection, FFTFlags};
use solid::math::poly::*;

use std::error::Error;

use num::Complex;

const OUT_FILE_NAME: &'static str = "sample.png";
fn plot(real: &[f64], imag: &[f64], len: usize, height: f32) -> Result<(), Box<dyn Error>>{
    use plotters::prelude::*;
    let root_area = BitMapBackend::new(OUT_FILE_NAME, (1920, 1080)).into_drawing_area();

    root_area.fill(&WHITE)?;

    let x_axis: Vec<f64> = (0..len).map(|x| x as f64).collect();

    let mut cc = ChartBuilder::on(&root_area)
        .margin(5u32)
        .set_all_label_area_size(50u32)
        .build_cartesian_2d(-0.0..len as f32, -height..height)?;

    cc.configure_mesh()
        .x_labels(20)
        .y_labels(10)
        .disable_mesh()
        .x_label_formatter(&|v| format!("{:.1}", v))
        .y_label_formatter(&|v| format!("{:.1}", v))
        .draw()?;

    cc.draw_series(LineSeries::new(x_axis.iter().zip(real.iter()).map(|(&x, &y)| (x as f32, y as f32)), &RED))?
        .label("Real")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
    cc.draw_series(LineSeries::new(x_axis.iter().zip(imag.iter()).map(|(&x, &y)| (x as f32, y as f32)), &BLUE))?
        .label("Imag")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    cc.configure_series_labels().border_style(&BLACK).draw()?;

    root_area.present().expect("Unable to write result to file");
    println!("Result has been saved to {}", OUT_FILE_NAME);
    
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let coefs = firdes::firdes_notch(25, 0.2, 30.0)?;
    let complex_coefs: Vec<Complex<f64>> = coefs.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let mut filter = FIRFilter::new(&coefs, 1.0);
    let mut complex_filter = FIRFilter::<Complex<f64>, f64>::new(&complex_coefs, Complex::new(1.0, 1.0));
    // let complex_iir_filter = IIRFilter::<Complex<f64>, f64>::new(&complex_coefs, &complex_coefs, IIRFilterType::Normal)?;

    let len = 500;
    let ivec: Vec<f64> = (-len/2..len/2).map(|x| (x as f64).cos() * 0.125).collect();
    let qvec: Vec<f64> = (-len/2..len/2).map(|x| (x as f64).sin() * 0.125).collect();
    let complex_vec : Vec<Complex<f64>> = ivec.iter().zip(qvec.iter()).map(|(&x, &y)| Complex::new(x, y)).collect();

    let _circular_buffer = CircularBuffer::from_slice(&complex_vec);
    // println!("{}", circular_buffer);

    let mut agc = AGC::new();
    println!("{}", agc);
    agc.squelch_enable();
    agc.set_bandwidth(0.01)?;
    agc.squelch_set_threshold(-22.0);
    let agc_vec = agc.execute_block(&complex_vec);
    
    let filter_output = filter.execute_block(&agc_vec);
    let _complex_filter_output = complex_filter.execute_block(&ivec);
    let _filter_delay = Filter::group_delay(&complex_filter, 0.125);
    let _freq_response = Filter::frequency_response(&complex_filter, 1000.0);
    // let _iir_filter_delay = Filter::group_delay(&complex_iir_filter, 0.125);
    // let _iir_freq_response = Filter::frequency_response(&complex_iir_filter, 1000.0);

    // println!("{:?}", _complex_filter_output);
    // println!("{_freq_response} {_filter_delay}");
    // println!("{_iir_filter_delay} {_iir_freq_response}");

    println!("{}", filter);
    println!("{}", complex_filter);

    let mut real = vec![];
    let mut imag = vec![];


    let mut auto_corr = AutoCorrelator::<f64>::new(10, 5);
    let _auto_corr_output = auto_corr.execute_block(&filter_output);

    println!("{}", auto_corr);

    let mut nco = NCO::new();
    nco.set_frequency(0.1);
    let mut nco_output = Vec::new();
    for _ in 0..1024 {
        let (r, i) = nco.sincos();
        nco_output.push(Complex::new(r, i));
        nco.step();
    }

    println!("{}", nco);

    let fft_size = 129 * 7;
    let fftf = FFT::new(fft_size, FFTDirection::FORWARD, FFTFlags::ESTIMATE);
    let fftr = FFT::new(fft_size, FFTDirection::REVERSE, FFTFlags::ESTIMATE);
    let fft_output = fftf.execute(&nco_output)?;
    let second_fft_output = fftr.execute(&fft_output)?;

    println!("{}", fftf);
    println!("{}", fftr);

    for num in fft_output.iter() {
        real.push(num.re);
        imag.push(num.im);
    }

    let freq = iirdes::frequency_pre_warp(0.20, 0.25, iirdes::BandType::LOWPASS);
    // let zpkf = iirdes::bilinear_zpkf()

    println!("{freq}");

    let dumb_pll_ff = [6039.61035, 4000.0, -2039.61035];
    let dumb_pll_fb = [4082.63281, -8163.26562, 4080.63281];
    let mut iir_filter = IIRFilter::new(&dumb_pll_ff, &dumb_pll_fb, IIRFilterType::SecondOrder)?;
    let _iir_output = iir_filter.execute_block(&second_fft_output);
    let _iir_filter_delay = iir_filter.group_delay(0.35);
    let _iir_freq_response = iir_filter.frequency_response(0.1);

    // println!("{iir_filter_delay} {iir_freq_response}");

    // plot(&real, &imag, second_fft_output.len(), fft_size as f32)?;

    let polynomial = [6.0, 11.0, -33.0, -33.0, 11.0, 6.0];
    // let _roots = find_roots(&polynomial)?;
    let rec_roots = find_roots_bairstow_persistent(&polynomial, 1.83333333333333333333, -5.5)?;

    println!("{:?}", rec_roots);

    Ok(())
}
