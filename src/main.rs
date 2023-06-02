// use num_traits::Num;
use solid::filter::Filter;
use solid::filter::iir::*;
use solid::filter::iirdes;
use solid::nco::NCO;

// use std::cell::RefCell;
use std::error::Error;
// use std::rc::Rc;

use num::Complex;

// #[warn(dead_code)]
// fn test_execute<I: Num, O>(mut filter: Box<dyn Filter<I, O>>, input: &[I]) -> (Box<dyn Filter<I, O>>, Vec<O>) {
//     let out = filter.execute_block(input);
//     (filter, out)
// }

// #[warn(dead_code)]
// fn test_pass_rc<I: Num, O>(rc_filter: Rc<RefCell<dyn Filter<I, O>>>, input: &[I]) -> Vec<O> {
//     let out = rc_filter.borrow_mut().execute_block(input);
//     out
// }

fn main() -> Result<(), Box<dyn Error>> {
    // let mut real = vec![];
    // let mut imag = vec![];

    let mut nco = NCO::new();
    nco.set_frequency(0.1);
    let mut nco_output = Vec::new();
    for _ in 0..102400 {
        let (r, i) = nco.sincos();
        nco_output.push(Complex::new(r, i));
        nco.step();
    }

    let filter = iirdes::pll::active_lag(0.02, 1.0 / (2f64).sqrt(), 1000.0)?;
    let mut iir_filter = IIRFilter::<f64, Complex<f64>>::new(&filter.0, &filter.1, IIRFilterType::SecondOrder)?;
    let _box_iir = Box::new(iir_filter.clone());
    let _iir_output = iir_filter.execute_block(&nco_output);

    

    Ok(())
}
