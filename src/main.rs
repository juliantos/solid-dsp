use num::Zero;
use num_traits::Num;
use solid::filter::Filter;
use solid::filter::firdes::*;
use solid::filter::fir::interp::*;
use solid::filter::iir::*;
use solid::filter::iirdes;
use solid::nco::NCO;

use std::cell::RefCell;
use std::error::Error;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;

use num::Complex;

fn test_execute<I: Num, O>(mut filter: Box<dyn Filter<I, O>>, input: &[I]) -> (Box<dyn Filter<I, O>>, Vec<O>) {
    let out = filter.execute_block(input);
    (filter, out)
}

fn test_pass_rc<I: Num, O>(rc_filter: Rc<RefCell<dyn Filter<I, O>>>, input: &[I]) -> Vec<O> {
    let out = rc_filter.borrow_mut().execute_block(input);
    out
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut real = vec![];
    let mut imag = vec![];

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
    let box_iir = Box::new(iir_filter.clone());
    let iir_output = iir_filter.execute_block(&nco_output);

    for num in iir_output.iter() {
        real.push(num.re);
        imag.push(num.im);
    }

    let coefficients = firdes_kaiser(50, 0.35, 40.0, 0.0)?;
    let mut interpolating_filter = InterpolatingFIRFilter::<f64, Complex<f64>>::new(&coefficients, 7)?;
    let mut cloned_filter = interpolating_filter.clone();
    let boxed_filter = Box::new(cloned_filter.clone());

    let in_out = interpolating_filter.execute(Complex { re: 10.0, im: 11.0 });
    let co_out = cloned_filter.execute(Complex { re: 10.0, im: 11.0 });
    let cl_out = cloned_filter.execute(Complex { re: 0.0, im: 0.0 });
    let (mut boxed_filter, bx_out) = test_execute(boxed_filter, &[Complex { re: 10.0, im: 11.0 }]);
    let rt_out = boxed_filter.execute(Complex { re: 0.0, im: 0.0 });
    let tr_out = interpolating_filter.execute(Complex { re: 0.0, im: 0.0 });

    assert_eq!(in_out, co_out);
    assert_eq!(in_out, bx_out);
    assert_eq!(cl_out, rt_out);
    assert_eq!(rt_out, tr_out);

    let (_, _new_iir_output) = test_execute(box_iir, &nco_output);

    let rc_filter = RefCell::new(interpolating_filter.clone());
    let rc_out = rc_filter.borrow_mut().execute(Complex::zero());
    let in_out = interpolating_filter.execute(Complex::zero());

    assert_eq!(rc_out, in_out);

    let rc_rc_filter = Rc::new(rc_filter);
    let rr_out = rc_rc_filter.borrow_mut().execute(Complex::new(1.0, 2.0));
    let in_out = interpolating_filter.execute(Complex::new(1.0, 2.0));

    assert_eq!(rr_out, in_out);

    let rf_out = test_pass_rc(rc_rc_filter.clone(), &[Complex::zero()]);
    let in_out = interpolating_filter.execute(Complex::zero());
    let r2_out = rc_rc_filter.borrow_mut().execute(Complex::new(1.0, 2.0));
    let i2_out = interpolating_filter.execute(Complex::new(1.0, 2.0));

    {
        let _c = Rc::clone(&rc_rc_filter);
        assert_eq!(Rc::strong_count(&rc_rc_filter), 2);
    }

    assert_eq!(Rc::strong_count(&rc_rc_filter), 1);

    assert_eq!(rf_out, in_out);
    assert_eq!(r2_out, i2_out);

    // let arc_filter = Arc::new(Mutex::new(rc_filter.clone()));

    // let handles = vec![];
    // for _ in 0..10 {
    //     let arc_filter = Arc::clone(&arc_filter);
    //     let handle = thread::spawn(move || {
    //         let a = arc_filter.lock().unwrap().borrow_mut().execute(Complex::new(10.0, 5.0));
    //     });
    //     handles.push(handle);
    // }

    // for handle in handles {
    //     handle.join().unwrap();
    // }

    Ok(())
}
