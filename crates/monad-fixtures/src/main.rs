//! Command-line entry point for the Monad blockchain fixture runner.

use clap::Parser;
use eyre::Result;
use std::{
    io::{self, Write},
    path::PathBuf,
    process::ExitCode,
};

/// Replay EEST blockchain JSON fixtures with Monad execution rules.
#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    /// JSON files or directories to search recursively.
    #[arg(required = true)]
    paths: Vec<PathBuf>,
    /// Select case names containing this substring.
    #[arg(long)]
    filter: Option<String>,
    /// Write a structured JSON report to stdout.
    #[arg(long)]
    json: bool,
    /// Return a failing exit code if any selected case is unsupported.
    #[arg(long)]
    fail_on_unsupported: bool,
}

fn run(args: Args) -> Result<u8> {
    let report = monad_fixtures::run(&args.paths, args.filter.as_deref())?;
    let mut output = io::stdout().lock();
    if args.json {
        serde_json::to_writer_pretty(&mut output, &report)?;
        writeln!(output)?;
    } else {
        writeln!(output, "Verification: {}", report.verification)?;
        for case in &report.cases {
            let status = match case.status {
                monad_fixtures::Status::Passed => "PASS",
                monad_fixtures::Status::Failed => "FAIL",
                monad_fixtures::Status::Unsupported => "UNSUPPORTED",
            };
            writeln!(
                output,
                "{status} {} :: {} — {}",
                case.file.display(),
                case.name,
                case.message
            )?;
        }
        writeln!(
            output,
            "{} passed, {} failed, {} unsupported ({} total)",
            report.summary.passed,
            report.summary.failed,
            report.summary.unsupported,
            report.summary.total
        )?;
    }
    Ok(report.exit_code(args.fail_on_unsupported))
}

fn main() -> ExitCode {
    match run(Args::parse()) {
        Ok(code) => ExitCode::from(code),
        Err(error) => {
            let _ = writeln!(io::stderr().lock(), "error: {error:#}");
            ExitCode::from(2)
        }
    }
}
