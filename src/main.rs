use std::path::PathBuf;

use indicatif::ProgressBar;
use major_lib::proccessing::{parse_major, AppError};

use opencv::imgcodecs::{imread, IMREAD_COLOR};

#[derive(argh::FromArgs)]
/// Basic Cli
struct Cli {
    #[argh(switch, short = 'v')]
    /// whether to output intermediatry results
    verbosity: bool,
    #[argh(positional)]
    input: PathBuf,
}

fn main() -> Result<(), AppError> {
    let cli: Cli = argh::from_env();

    let input = imread(&cli.input.as_os_str().to_string_lossy(), IMREAD_COLOR)?;

    let bar = ProgressBar::new(100);

    let majors = parse_major(&input, cli.verbosity, bar)?;

    for i in 1..=majors.len() {
        println!("{i:0>2}: {}", majors[i - 1]);
    }

    Ok(())
}
