use rustfft::{num_complex::Complex64, *};

fn fft(x: &[f64], pad: Option<usize>) -> Vec<Complex64> {
    let len = pad.unwrap_or(x.len());
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(len);

    let mut buf = x
        .iter()
        .copied()
        .chain(std::iter::repeat(0.0))
        .take(len)
        .map(|x| Complex64 { re: x, im: 0.0 })
        .collect::<Vec<_>>();
    fft.process(&mut buf);
    return buf;
}

fn ifft(x: &[Complex64], len: Option<usize>) -> Vec<f64> {
    let len = len.unwrap_or(x.len());
    let mut planner = FftPlanner::<f64>::new();
    let ifft = planner.plan_fft_inverse(x.len());

    let mut buf = x.iter().copied().collect::<Vec<_>>();
    let buflen = buf.len();
    ifft.process(&mut buf);
    return buf
        .into_iter()
        .map(|z| z.re / buflen as f64)
        .take(len)
        .collect();
}

fn next_power_of_2(n: usize) -> usize {
    let mut i = 1;
    while i < n {
        i *= 2;
    }
    return i;
}

pub fn convolution(f: Vec<f64>, g: Vec<f64>) -> Vec<f64> {
    let len = next_power_of_2(f.len() + g.len());
    let fft_f = fft(&f, Some(len));
    let fft_g = fft(&g, Some(len));
    let mut fft_res = Vec::with_capacity(len);

    for (f, g) in fft_f.into_iter().zip(fft_g.into_iter()) {
        fft_res.push(f * g);
    }

    return ifft(&fft_res, Some(f.len() + g.len()));
}

#[cfg(test)]
mod tests {
    use super::convolution;
    use approx::abs_diff_eq;

    #[test]
    fn square_signal() {
        let sigsqr = (0..20)
            .map(|x| if x < 10 { -1.0f64 } else { 1.0f64 })
            .collect::<Vec<_>>();
        let sigir = (0..30)
            .map(|x| if x == 0 { 1.0f64 } else { 0.0 })
            .cycle()
            .take(100)
            .collect::<Vec<_>>();

        let expected = sigsqr.iter().copied().cycle().take(50).collect::<Vec<_>>();
        let x = convolution(sigsqr, sigir);
        abs_diff_eq!(expected, sigsqr);
    }
}
