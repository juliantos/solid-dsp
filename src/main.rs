use solid::filter::firdes;
use solid::filter::fir_filter::{FIRFilter, float_filter::Filter};
use solid::filter::auto_correlator::AutoCorrelator;
use solid::auto_gain_control::AGC;

use std::error::Error;

use num::Complex;

const OUT_FILE_NAME: &'static str = "sample.png";
fn plot(real: &[f64], imag: &[f64], len: usize) -> Result<(), Box<dyn Error>>{
    use plotters::prelude::*;
    let root_area = BitMapBackend::new(OUT_FILE_NAME, (1920, 1080)).into_drawing_area();

    root_area.fill(&WHITE)?;

    let x_axis: Vec<f64> = (0..len).map(|x| x as f64).collect();

    let mut cc = ChartBuilder::on(&root_area)
        .margin(5u32)
        .set_all_label_area_size(50u32)
        .build_cartesian_2d(-0.0..len as f32, -5f32..5f32)?;

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
    let mut filter = FIRFilter::new(&coefs, 1.0);

    let len = 500;
    let ivec: Vec<f64> = (-len/2..len/2).map(|x| (x as f64).cos() * 0.125).collect();
    let qvec: Vec<f64> = (-len/2..len/2).map(|x| (x as f64).sin() * 0.125).collect();
    let complex_vec : Vec<Complex<f64>> = ivec.iter().zip(qvec.iter()).map(|(&x, &y)| Complex::new(x, y)).collect();

    let mut agc = AGC::new();
    agc.squelch_enable();
    agc.set_bandwidth(0.01)?;
    agc.squelch_set_threshold(-22.0);
    let agc_vec = agc.execute_block(&complex_vec);
    
    let filter_output = filter.execute_block(&agc_vec);
    let _filter_delay = Filter::group_delay(&filter, 0.125);

    let mut real = vec![];
    let mut imag = vec![];


    let mut auto_corr = AutoCorrelator::<f64>::new(10, 5);
    let auto_corr_output = auto_corr.execute_block(&filter_output);
    for num in auto_corr_output.iter() {
        real.push(num.re);
        imag.push(num.im);
    }

    // for index in 0..10 {
    //     println!("{}", auto_corr_output[index]);
    // }

    println!("{}", filter);

    plot(&real, &imag, auto_corr_output.len())?;

    Ok(())
}
