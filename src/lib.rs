use js_sys::Float64Array;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct OptimizerResult {
    x: Vec<f64>,
    pub f: f64,
    pub iterations: u32,
}

#[wasm_bindgen]
impl OptimizerResult {
    #[wasm_bindgen(getter)]
    pub fn x(&self) -> Float64Array {
        Float64Array::from(&self.x[..])
    }
}

fn project_simplex(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut u = v.to_vec();
    u.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());

    let mut cssv = 0.0_f64;
    let mut rho = 0usize;
    for (i, &ui) in u.iter().enumerate() {
        cssv += ui;
        if ui - (cssv / (i + 1) as f64) > 0.0 {
            rho = i;
        }
    }

    let cssv_rho: f64 = u[..=rho].iter().sum();
    let theta = (cssv_rho - 1.0) / (rho + 1) as f64;

    v.iter().map(|&vi| (vi - theta).max(0.0)).collect()
}

fn portfolio_variance(x: &[f64], sigma: &[f64], n: usize) -> f64 {
    let mut sigma_x = vec![0.0_f64; n];
    for i in 0..n {
        for j in 0..n {
            sigma_x[i] = sigma[i * n + j] * x[j];
        }
    }

    x.iter().zip(&sigma_x).map(|(xi, sx)| 0.5 * xi * sx).sum()
}

fn portfolio_gradient(x: &[f64], sigma: &[f64], n: usize) -> Vec<f64> {
    let mut grad = vec![0.0_f64; n];
    for i in 0..n {
        for j in 0..n {
            grad[i] += sigma[i * n + j] * x[j];
        }
    }
    grad
}

#[wasm_bindgen]
pub fn min_variance(sigma_flat: &[f64], n: usize) -> OptimizerResult {
    const MAX_ITER: u32 = 1000;
    const TOL: f64 = 1e-9;

    const ALPHA: f64 = 0.3;
    const BETA: f64 = 0.5;
    const STEP_INIT: f64 = 1.0;

    let mut x = project_simplex(&vec![1.0 / n as f64; n]);
    let mut iter = 0u32;

    for i in 0..MAX_ITER {
        iter = i + 1;
        let grad = portfolio_gradient(&x, &sigma_flat, n);

        let mut step = STEP_INIT;
        let f_curr = portfolio_variance(&x, &sigma_flat, n);
        let grad_norm_sq: f64 = grad.iter().map(|g| g * g).sum();

        let x_new = loop {
            let candidate: Vec<f64> = x.iter().zip(&grad).map(|(xi, gi)| xi - step * gi).collect();
            let x_proj = project_simplex(&candidate);
            let f_new = portfolio_variance(&x_proj, &sigma_flat, n);

            if f_new <= f_curr - ALPHA * step * grad_norm_sq {
                break x_proj;
            }
            step *= BETA;

            if step < 1e-15 {
                break x_proj;
            }
        };

        let delta: f64 = x_new
            .iter()
            .zip(&x)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        x = x_new;

        if delta < TOL {
            break;
        }
    }

    let f = portfolio_variance(&x, &sigma_flat, n);
    OptimizerResult {
        x,
        f,
        iterations: iter,
    }
}
