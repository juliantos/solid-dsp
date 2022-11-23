use num_traits::Num;
use solid::filter::Filter;
use solid::filter::firdes::*;
use solid::filter::fir::interp::*;
use solid::filter::iir::*;
use solid::filter::iirdes;
use solid::nco::NCO;

use std::error::Error;

use num::Complex;

#[allow(dead_code)]
fn test_execute<I: Num, O>(mut filter: Box<dyn Filter<I, O>>, input: &[I]) -> (Box<dyn Filter<I, O>>, Vec<O>) {
    let out = filter.execute_block(input);
    (filter, out)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut real = vec![];
    let mut imag = vec![];

    let mut nco = NCO::new();
    nco.set_frequency(0.1);
    let mut nco_output = Vec::new();
    for _ in 0..1024 {
        let (r, i) = nco.sincos();
        nco_output.push(Complex::new(r, i));
        nco.step();
    }

    let filter = iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0)?;
    let mut iir_filter = IIRFilter::<f64, Complex<f64>>::new(&filter.0, &filter.1, IIRFilterType::SecondOrder)?;
    let box_iir = Box::new(iir_filter.clone());
    let iir_output = iir_filter.execute_block(&nco_output);

    for num in iir_output.iter() {
        real.push(num.re);
        imag.push(num.im);
    }

    let coefficients = firdes_kaiser(50, 0.35, 40.0, 0.0)?;
    let mut interpolating_filter = InterpolatingFIRFilter::<f64, Complex<f64>>::new(&coefficients, 7)?;
    let mut copied_filter = interpolating_filter;
    let mut cloned_filter = interpolating_filter.clone();
    let boxed_filter = Box::new(cloned_filter);

    let in_out = interpolating_filter.execute(Complex { re: 10.0, im: 11.0 });
    let co_out = copied_filter.execute(Complex { re: 10.0, im: 11.0 });
    let cl_out = cloned_filter.execute(Complex { re: 10.0, im: 11.0 });
    let (mut boxed_filter, bx_out) = test_execute(boxed_filter, &[Complex { re: 10.0, im: 11.0 }]);
    let rt_out = boxed_filter.execute(Complex { re: 0.0, im: 0.0 });
    let tr_out = interpolating_filter.execute(Complex { re: 0.0, im: 0.0 });

    assert_eq!(in_out, cl_out);
    assert_eq!(co_out, bx_out);
    assert_eq!(rt_out, tr_out);

    let (_, new_iir_output) = test_execute(box_iir, &nco_output);

    dbg!(new_iir_output);
    
    Ok(())
}
