use std::str::FromStr;

use std::fmt::Display;

#[derive(Debug, strum_macros::Display)]
pub(crate) enum State {
    Id,
    Gender,
    Name,
    Quota,
    #[strum(to_string = "Standards_{0}")]
    Standards(StandardState),
    #[strum(to_string = "Filters_{0}")]
    Filters(FilterState),
    #[strum(to_string = "MinimumRate_{0}")]
    MinimumRate(i32),
    None,
}

#[derive(Debug, strum_macros::Display)]
pub(crate) enum FilterState {
    國文,
    英文,
    數a,
    數b,
    社會,
    自然,
    學測科目組合,
    None,
}

#[derive(Debug, strum_macros::Display)]
pub(crate) enum StandardState {
    國文,
    英文,
    數a,
    數b,
    社會,
    自然,
    英聽,
    None,
}

#[derive(Debug, Default, Clone)]
pub(crate) struct CertificationStandards {
    pub(crate) 國文: Option<Standard>,
    pub(crate) 英文: Option<Standard>,
    pub(crate) 數a: Option<Standard>,
    pub(crate) 數b: Option<Standard>,
    pub(crate) 社會: Option<Standard>,
    pub(crate) 自然: Option<Standard>,
    pub(crate) 英聽: Option<Standard>,
}

impl Display for CertificationStandards {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let p = "_".to_string();
        write!(
            f,
            "{},{},{},{},{},{},{}",
            &self.國文.as_ref().map_or(p.clone(), ToString::to_string),
            &self.英文.as_ref().map_or(p.clone(), ToString::to_string),
            &self.數a.as_ref().map_or(p.clone(), ToString::to_string),
            &self.數b.as_ref().map_or(p.clone(), ToString::to_string),
            &self.社會.as_ref().map_or(p.clone(), ToString::to_string),
            &self.自然.as_ref().map_or(p.clone(), ToString::to_string),
            &self.英聽.as_ref().map_or(p.clone(), ToString::to_string)
        )
    }
}

#[derive(Debug, Default, Clone)]
pub(crate) struct Filters {
    pub(crate) 國文: Option<f64>,
    pub(crate) 英文: Option<f64>,
    pub(crate) 數a: Option<f64>,
    pub(crate) 數b: Option<f64>,
    pub(crate) 社會: Option<f64>,
    pub(crate) 自然: Option<f64>,
    pub(crate) 學測科目組合: Option<f64>,
}
impl Display for Filters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let p = "_".to_string();
        write!(
            f,
            "{},{},{},{},{},{},{}",
            &self.國文.as_ref().map_or(p.clone(), ToString::to_string),
            &self.英文.as_ref().map_or(p.clone(), ToString::to_string),
            &self.數a.as_ref().map_or(p.clone(), ToString::to_string),
            &self.數b.as_ref().map_or(p.clone(), ToString::to_string),
            &self.社會.as_ref().map_or(p.clone(), ToString::to_string),
            &self.自然.as_ref().map_or(p.clone(), ToString::to_string),
            &self
                .學測科目組合
                .as_ref()
                .map_or(p.clone(), ToString::to_string)
        )
    }
}

#[derive(Debug, Clone, strum_macros::Display, strum_macros::EnumString)]
pub(crate) enum Standard {
    頂標,
    前標,
    均標,
    後標,
    底標,
    A,
    B,
    C,
}
