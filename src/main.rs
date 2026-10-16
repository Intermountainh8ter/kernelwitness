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
    Version,
}

fn load_json<T: serde::de::DeserializeOwned>(p: &std::path::Path) -> anyhow::Result<T> {
    Ok(serde_json::from_str(&std::fs::read_to_string(p)?)?)
}

#[derive(serde::Deserialize)]
struct Trials {
    trials: Vec<Trial>,
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
        Cmd::Version => println!("kernelwitness {}", env!("CARGO_PKG_VERSION")),
    }
    Ok(())
}
