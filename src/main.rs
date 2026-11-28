//! kernelwitness CLI — scaffold.

use clap::{Parser, Subcommand};
use kernelwitness::{aggregate_with_warmup, diff, reproducibility, Report, Rig, Trial};

#[derive(Parser)]
#[command(
    name = "kernelwitness",
    version,
    about = "Deterministic GPU kernel benchmarks"
)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Aggregate a trials JSON into a report with rig fingerprint and
    /// percentiles. `--warmup` discards the first N trials per
    /// pre-reg §3 (default 5).
    Aggregate {
        trials: std::path::PathBuf,
        /// Rig fingerprint JSON.
        #[arg(long)]
        rig: std::path::PathBuf,
        #[arg(long)]
        kernel: String,
        #[arg(long)]
        shape: String,
        #[arg(long, default_value_t = 5)]
        warmup: usize,
    },
    /// Diff two reports and attribute to rig fingerprint fields.
    Compare {
        a: std::path::PathBuf,
        b: std::path::PathBuf,
    },
    /// Pre-reg §4 reproducibility verdict: candidate reproduces baseline
    /// iff rigs match AND |delta_mean| <= 3*baseline.stdev. Exit codes:
    /// 0 reproduced, 1 within-bound but cross-rig, 2 out-of-bound.
    Reproducibility {
        baseline: std::path::PathBuf,
        candidate: std::path::PathBuf,
    },
    /// Model-free flip simulation: raw label flips under reduction-order
    /// sized noise versus flips adjudicated genuine by a score-move margin.
    FlipSim {
        #[arg(long, default_value_t = 7)]
        seed: u64,
        #[arg(long, default_value_t = 20000)]
        items: usize,
        #[arg(long)]
        noise_sd: f64,
        #[arg(long)]
        margin: f64,
        #[arg(long)]
        out: std::path::PathBuf,
    },
    /// Model-free screen-threshold sweep: precision of flagged candidate
    /// flips when true behavioural flips mix with numerical-noise flips.
    ScreenSim {
        #[arg(long, default_value_t = 7)]
        seed: u64,
        #[arg(long, default_value_t = 20000)]
        items: usize,
        #[arg(long)]
        noise_sd: f64,
        #[arg(long)]
        true_flip_rate: f64,
        #[arg(long, value_delimiter = ',')]
        thresholds: Vec<f64>,
        #[arg(long)]
        out: std::path::PathBuf,
    },
    /// Model-free sweep of bug detection rate against the number of shapes a
    /// suite samples and the sampling strategy.
    ShapeSim {
        #[arg(long, default_value_t = 7)]
        seed: u64,
        #[arg(long, default_value_t = 2000)]
        bugs_per_class: usize,
        #[arg(long, default_value_t = 256)]
        pool: usize,
        #[arg(long, value_delimiter = ',')]
        counts: Vec<usize>,
        #[arg(long)]
        out: std::path::PathBuf,
    },
    Version,
}

fn load_json<T: serde::de::DeserializeOwned>(p: &std::path::Path) -> anyhow::Result<T> {
    Ok(serde_json::from_str(&std::fs::read_to_string(p)?)?)
}

#[derive(serde::Deserialize)]
struct Trials {
    trials: Vec<Trial>,
}

fn iso_utc(t: u64) -> String {
    let (days, rem) = ((t / 86400) as i64, t % 86400);
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Cmd::Aggregate {
            trials,
            rig,
            kernel,
            shape,
            warmup,
        } => {
            let t: Trials = load_json(&trials)?;
            let r: Rig = load_json(&rig)?;
            let out = aggregate_with_warmup(r, &kernel, &shape, &t.trials, warmup);
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        Cmd::Compare { a, b } => {
            let ra: Report = load_json(&a)?;
            let rb: Report = load_json(&b)?;
            let d = diff(&ra, &rb);
            println!("{}", serde_json::to_string_pretty(&d)?);
            if !d.rig_match {
                println!("note: rigs differ — see each report's `rig` field for attribution");
            }
        }
        Cmd::Reproducibility {
            baseline,
            candidate,
        } => {
            let a: Report = load_json(&baseline)?;
            let b: Report = load_json(&candidate)?;
            let r = reproducibility(&a, &b);
            let reproduced = r.reproduced;
            let within = r.within_bound;
            println!("{}", serde_json::to_string_pretty(&r)?);
            if !reproduced {
                if within {
                    std::process::exit(1); // cross-rig
                } else {
                    std::process::exit(2); // out of 3-sigma band
                }
            }
        }
        Cmd::FlipSim {
            seed,
            items,
            noise_sd,
            margin,
            out,
        } => {
            let t0 = std::time::Instant::now();
            let started = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs();
            let st = kernelwitness::flipsim::simulate(seed, items, noise_sd, margin);
            let mut v = serde_json::to_value(&st)?;
            let o = v.as_object_mut().unwrap();
            o.insert("noise_sd".into(), noise_sd.into());
            o.insert("margin".into(), margin.into());
            o.insert("seed".into(), seed.into());
            o.insert(
                "provenance".into(),
                serde_json::json!({
                    "started_at": iso_utc(started),
                    "seeds": [seed],
                    "command": std::env::args().collect::<Vec<_>>().join(" "),
                    "hardware": format!("{}-{} cpu, no gpu", std::env::consts::OS, std::env::consts::ARCH),
                    "wall_s": t0.elapsed().as_secs_f64(),
                    "data_sources": [{"url": "synthetic: seeded xorshift PRNG inside the component", "license": "MIT (own code, no external data)"}],
                }),
            );
            if let Some(d) = out.parent() {
                std::fs::create_dir_all(d)?;
            }
            std::fs::write(&out, serde_json::to_string_pretty(&v)? + "\n")?;
            println!("{}", serde_json::to_string_pretty(&v)?);
        }
        Cmd::ScreenSim {
            seed,
            items,
            noise_sd,
            true_flip_rate,
            thresholds,
            out,
        } => {
            let t0 = std::time::Instant::now();
            let started = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs();
            let sweep = kernelwitness::flipsim::screen_sweep(seed, items, noise_sd, true_flip_rate, &thresholds);
            let inv = kernelwitness::flipsim::screen_sweep(seed, items, noise_sd, 0.0, &thresholds);
            let v = serde_json::json!({
                "items": items,
                "noise_sd": noise_sd,
                "true_flip_rate": true_flip_rate,
                "seed": seed,
                "threshold_sweep": sweep,
                "batch_invariant_sweep": inv,
                "provenance": {
                    "started_at": iso_utc(started),
                    "seeds": [seed],
                    "command": std::env::args().collect::<Vec<_>>().join(" "),
                    "hardware": format!("{}-{} cpu, no gpu", std::env::consts::OS, std::env::consts::ARCH),
                    "wall_s": t0.elapsed().as_secs_f64(),
                    "data_sources": [{"url": "synthetic: seeded xorshift PRNG inside the component", "license": "MIT (own code, no external data)"}],
                },
            });
            if let Some(d) = out.parent() {
                std::fs::create_dir_all(d)?;
            }
            std::fs::write(&out, serde_json::to_string_pretty(&v)? + "\n")?;
            println!("{}", serde_json::to_string_pretty(&v)?);
        }
        Cmd::ShapeSim {
            seed,
            bugs_per_class,
            pool,
            counts,
            out,
        } => {
            let t0 = std::time::Instant::now();
            let started = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs();
            let r = kernelwitness::shapesim::simulate(seed, bugs_per_class, pool, &counts);
            let mut v = serde_json::to_value(&r)?;
            let o = v.as_object_mut().unwrap();
            o.insert("seed".into(), seed.into());
            o.insert(
                "provenance".into(),
                serde_json::json!({
                    "started_at": iso_utc(started),
                    "seeds": [seed],
                    "command": std::env::args().collect::<Vec<_>>().join(" "),
                    "hardware": format!("{}-{} cpu, no gpu", std::env::consts::OS, std::env::consts::ARCH),
                    "wall_s": t0.elapsed().as_secs_f64(),
                    "data_sources": [{"url": "synthetic: seeded xorshift PRNG inside the component", "license": "MIT (own code, no external data)"}],
                }),
            );
            if let Some(d) = out.parent() {
                std::fs::create_dir_all(d)?;
            }
            std::fs::write(&out, serde_json::to_string_pretty(&v)? + "\n")?;
            println!("{}", serde_json::to_string_pretty(&v)?);
        }
        Cmd::Version => println!("kernelwitness {}", env!("CARGO_PKG_VERSION")),
    }
    Ok(())
}
