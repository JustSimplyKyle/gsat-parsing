use std::fmt::Display;

use unicode_width::UnicodeWidthStr;

#[derive(Default, Clone, serde::Deserialize, serde::Serialize)]
pub struct Major {
    pub id: String,
    pub gender_requirements: bool,
    pub name: Box<str>,
    pub quota: f64,
    pub certification_standards: CertificationStandards,
    pub filters: Filters,
    pub minimum_rate: [String; 6],
}

impl std::fmt::Display for Major {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let lhs = format!(
            "{}, {}, {}, {}",
            self.id, self.gender_requirements, self.name, self.quota
        );

        // Compute actual display width (handles CJK correctly)
        let width = UnicodeWidthStr::width(lhs.as_str());

        let base = 50_usize;

        let lhs = format!(
            "{lhs}{} [{}]",
            " ".repeat(base.saturating_sub(width)), // adjust padding here
            self.certification_standards,
        );

        let width = UnicodeWidthStr::width(lhs.as_str());
        let lhs = format!(
            "{lhs}{} [{}]",
            " ".repeat((base + 30).saturating_sub(width)), // adjust padding here
            self.filters,
        );

        let width = UnicodeWidthStr::width(lhs.as_str());

        write!(
            f,
            "{lhs}{} [{}]",
            " ".repeat((base + 50).saturating_sub(width)), // adjust padding here
            self.minimum_rate
                .clone()
                .map(|x| if x.is_empty() { "_".to_string() } else { x })
                .join(",")
        )
    }
}

#[derive(Debug, strum_macros::Display)]
pub enum State {
    Id,
    Gender,
    Name,
    Quota,
    #[strum(to_string = "standards-{0}")]
    Standards(StandardState),
    #[strum(to_string = "filters-{0}")]
    Filters(FilterState),
    #[strum(to_string = "minimumRate-{0}")]
    MinimumRate(i32),
    None,
}

#[derive(Debug, strum_macros::Display)]
pub enum FilterState {
    國文,
    英文,
    數a,
    數b,
    社會,
    自然,
    學測科目組合,
    None,
}

#[derive(Debug, strum_macros::Display, strum_macros::EnumIter)]
pub enum StandardState {
    國文,
    英文,
    數a,
    數b,
    社會,
    自然,
    英聽,
    None,
}

#[derive(Debug, Default, Clone, serde::Deserialize, serde::Serialize)]
pub struct CertificationStandards {
    pub 國文: Option<Standard>,
    pub 英文: Option<Standard>,
    pub 數a: Option<Standard>,
    pub 數b: Option<Standard>,
    pub 社會: Option<Standard>,
    pub 自然: Option<Standard>,
    pub 英聽: Option<Standard>,
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

#[derive(Debug, Default, Clone, serde::Deserialize, serde::Serialize)]
pub struct Filters {
    pub 國文: Option<f64>,
    pub 英文: Option<f64>,
    pub 數a: Option<f64>,
    pub 數b: Option<f64>,
    pub 社會: Option<f64>,
    pub 自然: Option<f64>,
    pub 學測科目組合: Option<f64>,
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

#[derive(
    Debug,
    Clone,
    strum_macros::Display,
    strum_macros::EnumString,
    serde::Deserialize,
    serde::Serialize,
)]
pub enum Standard {
    頂標,
    前標,
    均標,
    後標,
    底標,
    A,
    B,
    C,
}
