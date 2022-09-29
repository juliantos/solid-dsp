use solid::filter::iir_filter::*;
use solid::filter::iirdes;
use solid::nco::NCO;

use std::error::Error;

use num::Complex;

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
    let mut iir_filter = IIRFilter::new(&filter.0, &filter.1, IIRFilterType::SecondOrder)?;
    let iir_output = iir_filter.execute_block(&nco_output);

    assert!(iirdes::stable(&filter.0, &filter.1)?);

    for num in iir_output.iter() {
        real.push(num.re);
        imag.push(num.im);
    }

    Ok(())
}
