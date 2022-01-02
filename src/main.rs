use solid::filter::firdes;
use solid::filter::fir_filter::{FIRFilter};
use solid::filter::firdes::filter_traits::Firdes;

use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {

    
    let coefs = firdes::firdes_notch(25, 0.20, 30.0)?;
    let filter = FIRFilter::new(&coefs, 1.0);

    println!("{}", filter.autocorrelation(0));

    // println!("{}", complex_filter);

    // println!("{:.4}", response);

    // println!("{:.4}", delay);

    Ok(())
}