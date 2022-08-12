use solid::filter::iirdes;
use solid::filter::iir_filter::*;
use solid::nco::NCO;

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

    assert_eq!(iirdes::stable(&filter.0, &filter.1)?, false);

    for num in iir_output.iter() {
        real.push(num.re);
        imag.push(num.im);
    }

    plot(&real, &imag, iir_output.len(), 1000.0)?;

    Ok(())
}
