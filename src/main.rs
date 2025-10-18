use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    str::FromStr,
};

use error_set::error_set;
use image::{DynamicImage, GenericImageView, RgbaImage};
use imageproc::drawing::draw_text_mut;
use opencv::{
    core::*,
    imgcodecs::*,
    imgproc::{self, *},
    prelude::*,
    ximgproc::{self, THINNING_ZHANGSUEN},
};
// use  as OcrPaddleError;
use rust_paddle_ocr::{Det, Rec};
use unicode_width::UnicodeWidthStr;

pub fn remove_text_from_white_background(img: &Mat, threshold: f64) -> Result<Mat, opencv::Error> {
    // Threshold to isolate white background (text on white)
    // White pixels will be 255, everything else will be 0
    let mut thresh = Mat::default();
    imgproc::threshold(&img, &mut thresh, threshold, 255.0, THRESH_BINARY)?;

    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        Size::new(20, 5),
        Point::new(-1, -1),
    )?;
    let mut closed = Mat::default();
    imgproc::morphology_ex(
        &thresh,
        &mut closed,
        imgproc::MORPH_CLOSE,
        &kernel,
        Point::new(-1, -1),
        1,
        BORDER_CONSTANT,
        Scalar::default(),
    )?;

    let mut output = Mat::default();
    bitwise_or(&img, &closed, &mut output, &Mat::default())?;

    Ok(output)
}

error_set! {
    AppError := TableSplitError || IoError || ImageDecodeError || OcrError
    IoError := {
        #[display("An IO error occured, reason: {:#?}", source)]
        (std::io::Error)
    }
    TableSplitError := {
        #[display("OpenCV errored, reason: {:#?}", source)]
        (opencv::Error),
        #[display("Failed to parse the cropped image, reason: {:#?}", source)]
        (image::ImageError)
    } || OcrError
    OcrError := {
        (rust_paddle_ocr::OcrError)
    }
    ImageDecodeError := {
        #[display("Fails to decode image, reason: {:#?}", source)]
        (image::ImageError)
    }
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), AppError> {
    let mut input = imread("011.png", IMREAD_COLOR)?;
    let mut det = Det::from_file("./models/PP-OCRv5_mobile_det.mnn")?
        .with_rect_border_size(12)
        .with_merge_boxes(false)
        .with_merge_threshold(1);
    let mut rec = Rec::from_file(
        "./models/PP-OCRv5_mobile_rec.mnn",
        "./models/ppocr_keys_v5.txt",
    )?
    .with_min_score(0.6)
    .with_punct_min_score(0.1);

    let height = input.rows();
    let width = input.cols();

    let mut gray = Mat::default();
    cvt_color(
        &input,
        &mut gray,
        COLOR_BGR2GRAY,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    let gray = remove_text_from_white_background(&gray, 254.0)?;
    imwrite("cleaned.png", &gray, &Vector::new())?;

    // 反相
    let mut gray_inv = Mat::default();
    bitwise_not(&gray, &mut gray_inv, &no_array())?;

    // 二值化
    let mut binary = Mat::default();
    threshold(&gray_inv, &mut binary, 150.0, 255.0, THRESH_BINARY)?;

    let mut h_lines = detect_h_lines(height, width, &binary)?;

    let mut v_lines = detect_v_lines(height, width, binary)?;

    ximgproc::thinning(&h_lines.clone(), &mut h_lines, THINNING_ZHANGSUEN)?;

    ximgproc::thinning(&v_lines.clone(), &mut v_lines, THINNING_ZHANGSUEN)?;

    let mut intersection_mat = Mat::default();

    bitwise_and(&v_lines, &h_lines, &mut intersection_mat, &no_array())?;

    split_table(&mut det, &mut rec, &gray, &mut input, &intersection_mat)?;

    let mut output = input;

    let green_overlay = Scalar::new(0.0, 255.0, 0.0, 0.0);
    let blue_overlay = Scalar::new(0.0, 0.0, 255.0, 0.0);

    bitwise_or(&output.clone(), &green_overlay, &mut output, &v_lines)?;
    bitwise_or(&output.clone(), &blue_overlay, &mut output, &h_lines)?;

    imwrite("test.png", &output, &Vector::new())?;

    Ok(())
}

fn detect_h_lines(height: i32, width: i32, binary: &Mat) -> Result<Mat, AppError> {
    let kernel_sizes = vec![width / 10, width / 20, width / 30];
    let mut combined_h_lines = Mat::zeros(height, width, CV_8UC1)?.to_mat()?;
    for &kernel_width in &kernel_sizes {
        let horizontal_kernel =
            get_structuring_element(MORPH_RECT, Size::new(kernel_width, 1), Point::new(-1, -1))?;

        let mut detected = Mat::default();
        morphology_ex(
            binary,
            &mut detected,
            MORPH_OPEN,
            &horizontal_kernel,
            Point::new(-1, -1),
            1,
            BORDER_CONSTANT,
            morphology_default_border_value()?,
        )?;
        let mut tmp = combined_h_lines.clone();
        bitwise_or(&combined_h_lines, &detected, &mut tmp, &no_array())?;
        combined_h_lines = tmp;
    }
    Ok(combined_h_lines)
}

fn detect_v_lines(height: i32, width: i32, binary: Mat) -> Result<Mat, AppError> {
    let dilate_kernel = get_structuring_element(MORPH_RECT, Size::new(1, 3), Point::new(-1, -1))?;
    let mut binary_dilated = Mat::default();
    dilate(
        &binary,
        &mut binary_dilated,
        &dilate_kernel,
        Point::new(-1, -1),
        1,
        BORDER_CONSTANT,
        morphology_default_border_value()?,
    )?;
    let kernel_sizes_v = vec![
        height / 15,
        height / 25,
        height / 35,
        height / 50,
        height / 100,
    ];
    let mut combined_v_lines = Mat::zeros(height, width, CV_8UC1)?.to_mat()?;
    for &kernel_height in &kernel_sizes_v {
        let vertical_kernel =
            get_structuring_element(MORPH_RECT, Size::new(1, kernel_height), Point::new(-1, -1))?;
        let mut detected = Mat::default();
        morphology_ex(
            &binary_dilated,
            &mut detected,
            MORPH_OPEN,
            &vertical_kernel,
            Point::new(-1, -1),
            1,
            BORDER_CONSTANT,
            morphology_default_border_value()?,
        )?;
        bitwise_or(
            &combined_v_lines.clone(),
            &detected,
            &mut combined_v_lines,
            &no_array(),
        )?;
    }
    Ok(combined_v_lines)
}

#[derive(Default, Clone)]
struct Major {
    id: String,
    gender_requirements: bool,
    name: Box<str>,
    quota: usize,
    certification_standards: CertificationStandards,
    filters: Filters,
}

impl std::fmt::Display for Major {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let lhs = format!(
            "{}, {}, {}, {}",
            self.id, self.gender_requirements, self.name, self.quota
        );

        // Compute actual display width (handles CJK correctly)
        let width = UnicodeWidthStr::width(lhs.as_str());

        write!(
            f,
            "{lhs}{} [{}]",
            " ".repeat(60usize.saturating_sub(width)), // adjust padding here
            self.certification_standards,
        )
    }
}

#[derive(Debug)]
enum State {
    Id,
    Gender,
    Name,
    Quota,
    Standards(StandardState),
    Filters,
    None,
}

#[derive(Debug)]
enum StandardState {
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
struct CertificationStandards {
    國文: Option<Standard>,
    英文: Option<Standard>,
    數a: Option<Standard>,
    數b: Option<Standard>,
    社會: Option<Standard>,
    自然: Option<Standard>,
    英聽: Option<Standard>,
}

impl Display for CertificationStandards {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let p = "_".to_string();
        write!(
            f,
            "{},{},{},{},{},{},{}",
            &self.國文.as_ref().map_or(p.clone(), |v| v.to_string()),
            &self.英文.as_ref().map_or(p.clone(), |v| v.to_string()),
            &self.數a.as_ref().map_or(p.clone(), |v| v.to_string()),
            &self.數b.as_ref().map_or(p.clone(), |v| v.to_string()),
            &self.社會.as_ref().map_or(p.clone(), |v| v.to_string()),
            &self.自然.as_ref().map_or(p.clone(), |v| v.to_string()),
            &self.英聽.as_ref().map_or(p.clone(), |v| v.to_string())
        )
    }
}

#[derive(Debug, Default, Clone)]
struct Filters {
    國文: Option<f64>,
    英文: Option<f64>,
    數a: Option<f64>,
    數b: Option<f64>,
    社會: Option<f64>,
    自然: Option<f64>,
    英聽: Option<f64>,
}

#[derive(Debug, Clone)]
enum Standard {
    頂標,
    前標,
    均標,
    後標,
    底標,
    A,
    B,
    C,
}

impl Display for Standard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let p = match self {
            Standard::頂標 => "頂標",
            Standard::前標 => "前標",
            Standard::均標 => "均標",
            Standard::後標 => "後標",
            Standard::底標 => "底標",
            Standard::A => "A",
            Standard::B => "B",
            Standard::C => "C",
        };
        write!(f, "{}", p)
    }
}

#[derive(Debug)]
struct Unimplemented;

impl FromStr for Standard {
    type Err = Unimplemented;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let p = match s {
            "頂標" => Self::頂標,
            "前標" => Self::前標,
            "均標" => Self::均標,
            "後標" => Self::後標,
            "底標" => Self::底標,
            "A" => Self::A,
            "B" => Self::B,
            "C" => Self::C,
            _ => return Err(Unimplemented),
        };
        Ok(p)
    }
}

#[allow(clippy::too_many_lines)]
pub fn split_table(
    det: &mut Det,
    rec: &mut Rec,
    gray: &Mat,
    original_image: &mut Mat,
    intersections: &Mat,
) -> Result<(), TableSplitError> {
    let mut points_mat = Mat::default();
    find_non_zero(intersections, &mut points_mat)?;

    let mut intersections: Vec<(i32, i32)> = points_mat
        .iter::<Point>()
        .into_iter()
        .flatten()
        .map(|e| (e.1.y, e.1.x))
        .collect();

    let mut intersections = intersections
        .into_iter()
        .map(|x| (x.1, x.0))
        .collect::<Vec<_>>();
    intersections.sort_unstable();

    let table = intersections.iter().collect::<HashSet<_>>();

    let max_x = intersections.iter().map(|(x, _)| *x).max().unwrap() as usize;
    let unique_y_values: HashSet<_> = intersections.iter().map(|x| x.1).collect();
    let rows_total_count = unique_y_values.len();

    let mut row_counter = 0;
    let mut state = State::None;
    let mut majors = vec![Major::default(); rows_total_count];

    for (i, window) in intersections
        .windows(2)
        .filter(|w| w[0].0 == w[1].0)
        .enumerate()
    {
        let Some(verticies) = extract_verticies(max_x, &table, window) else {
            continue;
        };

        let mut bounding_box = get_biggest_possible_bounding_box(verticies);

        // Expand rect slightly to increase ocr accuracy
        bounding_box.y = (bounding_box.y - 1).max(0);
        bounding_box.x = (bounding_box.x - 1).max(0);
        bounding_box.height = (bounding_box.height + 1).min(gray.rows() - bounding_box.y);
        bounding_box.width = (bounding_box.width + 1).min(gray.cols() - bounding_box.x);

        let cell = Mat::roi(gray, bounding_box)?.clone_pointee();

        // OCR processing
        let mut buf = Vector::new();
        imencode(".png", &cell, &mut buf, &Vector::new())?;
        let img = image::ImageReader::new(std::io::Cursor::new(buf.as_slice()))
            .with_guessed_format()
            .expect("should be guarnateed a .png file")
            .decode()?;
        let ocr = paddle_ocr(det, rec, &img)?;
        let rect = det.find_text_rect(&img)?;
        for rec in rect.clone() {
            let tl = (rec.left(), rec.top());
            let br = (rec.right(), rec.bottom());
            let mut rec = Rect_::from_points(tl.into(), br.into());
            rec.x += bounding_box.x + 1;
            rec.y += bounding_box.y + 1;
            rectangle(
                original_image,
                rec,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                1,
                LINE_8,
                0,
            )?;
        }

        let ocr = if let Some(text_dimensions) = rect.first() {
            imwrite(&format!("outputs/{i}.png"), &cell, &Vector::new())?;
            // Render back the ocr-ed text
            let text_widths = rect.iter().map(imageproc::rect::Rect::width).sum::<u32>();
            let font_scale = if bounding_box.height <= 24 {
                text_dimensions.height() as f32
            } else {
                text_dimensions.height() as f32 - 8.
            };
            let color = image::Rgba::from([255, 255, 255, 255]);
            let mut img_buffer = RgbaImage::new(text_widths, text_dimensions.height());
            let font = ab_glyph::FontRef::try_from_slice(include_bytes!(
                "/usr/share/fonts/noto-cjk/NotoSansCJK.otf"
            ))
            .expect("place your font at /usr/share/fonts/noto-cjk/NotoSansCJK.otf");
            draw_text_mut(&mut img_buffer, color, 0, 0, font_scale, &font, &ocr);

            // Combine cell image and text
            let text_mat = image_to_mat(img_buffer)?;
            let converted_cell = convert_rgba_to_bgr(cell)?;
            let converted_text = convert_rgba_to_bgr(text_mat)?;
            let result = vconcat_with_horizontal_pad(&converted_cell, &converted_text)?;

            imwrite(&format!("outputs/{i}.png"), &result, &Vector::new())?;
            ocr
        } else {
            "--".to_string()
        };

        match state {
            State::Id => {
                majors[row_counter].id = ocr.clone();
            }
            State::Gender => majors[row_counter].gender_requirements = ocr != "無" && ocr != "--",
            State::Name => {
                majors[row_counter].name = ocr.clone().into();
            }
            State::Quota => {
                majors[row_counter].quota = ocr.clone().parse().unwrap_or(0);
            }
            State::Standards(StandardState::None) => {}
            State::Standards(StandardState::國文) if ocr != "檢定標準" && ocr != "國文" => {
                if ocr == "--" {
                    majors[row_counter].certification_standards.國文 = None;
                } else {
                    majors[row_counter].certification_standards.國文 =
                        Some(ocr.parse().unwrap_or(Standard::C));
                }
            }
            State::Standards(StandardState::國文) => {}
            State::Standards(StandardState::英文) if ocr != "檢定標準" && ocr != "英文" => {
                if ocr == "--" {
                    majors[row_counter].certification_standards.英文 = None;
                } else {
                    majors[row_counter].certification_standards.英文 =
                        Some(ocr.parse().unwrap_or(Standard::C));
                }
            }
            State::Standards(StandardState::英文) => {}
            State::Standards(StandardState::數a) if ocr != "檢定標準" && !ocr.contains("數學") => {
                if ocr == "--" {
                    majors[row_counter].certification_standards.數a = None;
                } else {
                    majors[row_counter].certification_standards.數a =
                        Some(ocr.parse().unwrap_or(Standard::C));
                }
            }
            State::Standards(StandardState::數a) => {}
            State::Standards(StandardState::數b) if ocr != "檢定標準" && !ocr.contains("數學") => {
                if ocr == "--" {
                    majors[row_counter].certification_standards.數b = None;
                } else {
                    majors[row_counter].certification_standards.數b =
                        Some(ocr.parse().unwrap_or(Standard::C));
                }
            }
            State::Standards(StandardState::數b) => {}
            State::Standards(StandardState::社會) if ocr != "檢定標準" && ocr != "社會" => {
                if ocr == "--" {
                    majors[row_counter].certification_standards.社會 = None;
                } else {
                    majors[row_counter].certification_standards.社會 =
                        Some(ocr.parse().unwrap_or(Standard::C));
                }
            }
            State::Standards(StandardState::社會) => {}
            State::Standards(StandardState::自然) if ocr != "檢定標準" && ocr != "自然" => {
                if ocr == "--" {
                    majors[row_counter].certification_standards.自然 = None;
                } else {
                    majors[row_counter].certification_standards.自然 =
                        Some(ocr.parse().unwrap_or(Standard::C));
                }
            }
            State::Standards(StandardState::自然) => {}
            State::None => {}
            _ => {
                dbg!(&state);
                todo!();
            }
        }

        row_counter += 1;

        match &*ocr {
            "校系代碼" => {
                if !matches!(state, State::Id) {
                    state = State::Id;
                    row_counter = 0;
                } else {
                    row_counter -= 1;
                }
            }
            "性別要求" => {
                if !matches!(state, State::Gender) {
                    for i in 0..majors.len() {
                        println!("{i}: {}", majors[i]);
                    }
                    state = State::Gender;
                    row_counter = 0;
                } else {
                    row_counter -= 1;
                }
            }
            "校系名稱" => {
                if !matches!(state, State::Name) {
                    for i in 0..majors.len() {
                        println!("{i}: {}", majors[i]);
                    }
                    state = State::Name;
                    row_counter = 0;
                } else {
                    row_counter -= 1;
                }
            }
            "招生名額" => {
                if !matches!(state, State::Quota) {
                    for i in 0..majors.len() {
                        println!("{i}: {}", majors[i]);
                    }
                    state = State::Quota;
                    row_counter = 0;
                } else {
                    row_counter -= 1;
                }
            }
            "檢定標準" => {
                if !matches!(state, State::Standards(_)) {
                    for i in 0..majors.len() {
                        println!("{i}: {}", majors[i]);
                    }
                    state = State::Standards(StandardState::None);
                    row_counter = 0;
                } else {
                    row_counter -= 1;
                }
            }
            "篩選倍率" => {}
            "國文" => {
                if matches!(state, State::Standards(StandardState::None)) {
                    state = State::Standards(StandardState::國文);
                    row_counter = 0;
                } else {
                    row_counter -= 1;
                }
            }
            "英文" => {
                if matches!(state, State::Standards(StandardState::國文)) {
                    for i in 0..majors.len() {
                        println!("{i}: {}", majors[i]);
                    }
                    state = State::Standards(StandardState::英文);
                    row_counter = 0;
                } else {
                    row_counter -= 1;
                }
            }
            x if x.contains("數學") && x.contains('A') => {
                if matches!(state, State::Standards(StandardState::英文)) {
                    for i in 0..majors.len() {
                        println!("{i}: {}", majors[i]);
                    }
                    state = State::Standards(StandardState::數a);
                    row_counter = 0;
                } else {
                    row_counter -= 1;
                }
            }
            x if x.contains("數學") && x.contains('B') => {
                if matches!(state, State::Standards(StandardState::數a)) {
                    for i in 0..majors.len() {
                        println!("{i}: {}", majors[i]);
                    }
                    state = State::Standards(StandardState::數b);
                    row_counter = 0;
                } else {
                    row_counter -= 1;
                }
            }
            "社會" => {
                if matches!(state, State::Standards(StandardState::數b)) {
                    for i in 0..majors.len() {
                        println!("{i}: {}", majors[i]);
                    }
                    state = State::Standards(StandardState::社會);
                    row_counter = 0;
                } else {
                    row_counter -= 1;
                }
            }
            "自然" => {
                if matches!(state, State::Standards(StandardState::社會)) {
                    for i in 0..majors.len() {
                        println!("{i}: {}", majors[i]);
                    }
                    state = State::Standards(StandardState::自然);
                    row_counter = 0;
                } else {
                    row_counter -= 1;
                }
            }
            "英聽" => {
                if matches!(state, State::Standards(StandardState::自然)) {
                    for i in 0..majors.len() {
                        println!("{i}: {}", majors[i]);
                    }
                    state = State::Standards(StandardState::英聽);
                    row_counter = 0;
                } else {
                    row_counter -= 1;
                }
            }
            _ => {}
        }
    }

    Ok(())
}

fn get_biggest_possible_bounding_box(
    [top_left, top_right, bottom_left, bottom_right]: [Point_<i32>; 4],
) -> Rect_<i32> {
    let rect1 = Rect::from_points(
        (bottom_left.x, bottom_left.y).into(),
        (top_right.x, top_right.y).into(),
    );

    let rect2 = Rect::from_points(
        (top_left.x, top_left.y).into(),
        (bottom_right.x, bottom_right.y).into(),
    );

    if rect1 > rect2 { rect1 } else { rect2 }
}

fn extract_verticies(
    max_x: usize,
    table: &HashSet<&(i32, i32)>,
    window: &[(i32, i32)],
) -> Option<[Point_<i32>; 4]> {
    let [top_left, bottom_left] = window else {
        unreachable!()
    };

    let mut top_right = None;
    for x in top_left.0 as usize + 1..max_x {
        if table.contains(&(x as i32, top_left.1)) {
            top_right = Some((x as i32, top_left.1));
            break;
        }
    }
    let mut bottom_right = None;
    for x in bottom_left.0 as usize + 1..max_x {
        if table.contains(&(x as i32, bottom_left.1)) {
            bottom_right = Some((x as i32, bottom_left.1));
            break;
        }
    }
    Some([
        (*top_left).into(),
        top_right?.into(),
        (*bottom_left).into(),
        bottom_right?.into(),
    ])
}

fn convert_rgba_to_bgr(cell: Mat) -> Result<Mat, TableSplitError> {
    let mut converted_cell = Mat::default();
    cvt_color(
        &cell,
        &mut converted_cell,
        COLOR_RGBA2BGR,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    Ok(converted_cell)
}

fn image_to_mat(
    img_buffer: image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
) -> Result<Mat, TableSplitError> {
    let mut text_mat = unsafe {
        Mat::new_rows_cols(
            img_buffer.dimensions().1 as i32,
            img_buffer.dimensions().0 as i32,
            CV_8UC4,
        )?
    };
    text_mat
        .data_bytes_mut()?
        .copy_from_slice(img_buffer.as_raw());
    Ok(text_mat)
}

fn paddle_ocr(
    det: &mut Det,
    rec: &mut Rec,
    image: &DynamicImage,
) -> Result<(String), rust_paddle_ocr::OcrError> {
    let text_images = det.find_text_img(image)?;
    let target = text_images
        .into_iter()
        .map(|x| rec.predict_str(&x))
        .collect::<Result<Vec<_>, rust_paddle_ocr::OcrError>>()?
        .join("");
    Ok(target)
}

fn vconcat_with_horizontal_pad(img1: &Mat, img2: &Mat) -> Result<Mat, opencv::Error> {
    let w1 = img1.cols();
    let w2 = img2.cols();
    let max_width = w1.max(w2);

    let padded1 = if w1 < max_width {
        let left_pad = (max_width - w1) / 2;
        let right_pad = max_width - w1 - left_pad;
        let mut padded = Mat::default();
        opencv::core::copy_make_border(
            img1,
            &mut padded,
            0,
            0,
            left_pad,
            right_pad,
            opencv::core::BORDER_CONSTANT,
            Scalar::all(0.0),
        )?;
        padded
    } else {
        img1.clone()
    };

    let padded2 = if w2 < max_width {
        let left_pad = (max_width - w2) / 2;
        let right_pad = max_width - w2 - left_pad;
        let mut padded = Mat::default();
        opencv::core::copy_make_border(
            img2,
            &mut padded,
            0,
            0,
            left_pad,
            right_pad,
            opencv::core::BORDER_CONSTANT,
            Scalar::all(0.0),
        )?;
        padded
    } else {
        img2.clone()
    };

    let mut result = Mat::default();
    opencv::core::vconcat2(&padded1, &padded2, &mut result)?;
    Ok(result)
}
