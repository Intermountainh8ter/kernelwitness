//! kernelwitness CLI — scaffold.

use clap::{Parser, Subcommand};
use kernelwitness::{aggregate, diff, Report, Rig, Trial};

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
    /// percentiles.
    Aggregate {
        trials: std::path::PathBuf,
        /// Rig fingerprint JSON.
        #[arg(long)]
        rig: std::path::PathBuf,
        #[arg(long)]
        kernel: String,
        #[arg(long)]
        shape: String,
    },
    /// Diff two reports and attribute to rig fingerprint fields.
    Compare {
        a: std::path::PathBuf,
        b: std::path::PathBuf,
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
        } => {
            let t: Trials = load_json(&trials)?;
            let r: Rig = load_json(&rig)?;
            let out = aggregate(r, &kernel, &shape, &t.trials);
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
        Cmd::Version => println!("kernelwitness {}", env!("CARGO_PKG_VERSION")),
    }
    Ok(())
}
