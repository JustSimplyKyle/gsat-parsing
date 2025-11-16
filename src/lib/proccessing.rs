use std::{
    collections::{HashMap, HashSet},
    f64::consts::PI,
    fs::{create_dir_all, remove_dir_all},
    path::{Path, PathBuf},
    sync::OnceLock,
};

pub use opencv::imgcodecs::{imread, IMREAD_COLOR};

use error_set::error_set;
use image::{DynamicImage, RgbaImage};
use imageproc::drawing::draw_text_mut;
pub use indicatif::ProgressBar;
use indicatif::{ProgressIterator, ProgressStyle};
use opencv::{
    core::*,
    imgcodecs::*,
    imgproc::{self, *},
};
// use  as OcrPaddleError;
use rust_paddle_ocr::{Det, Rec};
use rusty_tesseract::Args;

use crate::gsat::{self, FilterState, Major, StandardState, State};

pub fn extract_blocks(img: &Mat, limit: f64) -> Result<Vec<Mat>, opencv::Error> {
    // Threshold to isolate white background (text on white)
    // White pixels will be 255, everything else will be 0
    let mut thresh = Mat::default();
    imgproc::threshold(&img, &mut thresh, limit, 255.0, THRESH_BINARY)?;

    let kernel =
        get_structuring_element(imgproc::MORPH_RECT, Size::new(20, 5), Point::new(-1, -1))?;

    let mut closed = Mat::default();

    morphology_ex(
        &thresh,
        &mut closed,
        imgproc::MORPH_CLOSE,
        &kernel,
        Point::new(-1, -1),
        1,
        BORDER_CONSTANT,
        Scalar::default(),
    )?;

    imwrite("morphology_ex.png", &closed, &Vector::new())?;

    let mut output = Mat::default();
    bitwise_or(&img, &closed, &mut output, &Mat::default())?;

    imwrite("morphology_ex-ored.png", &output, &Vector::new())?;

    let mut thresh = Mat::default();
    threshold(&output, &mut thresh, limit, 255.0, THRESH_BINARY)?;
    imwrite("morphology_ex-thershed.png", &output, &Vector::new())?;

    bitwise_not(&thresh.clone(), &mut thresh, &no_array())?;
    imwrite("morphology_ex-noted.png", &thresh, &Vector::new())?;

    let mut contours = Vector::<Vector<Point>>::new();
    let mut hierarchy = Vector::<Vec4i>::new();
    find_contours_with_hierarchy(
        &thresh,
        &mut contours,
        &mut hierarchy,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    )?;

    let mut blocks: Vec<Rect> = contours
        .iter()
        .map(|contour| {
            let rect = imgproc::bounding_rect(&contour).unwrap();
            let area = (rect.width * rect.height) as f64;
            (rect, area)
        })
        .filter(|(rect, area)| {
            // Filter out very small blocks (noise)
            *area > 1000.0 && rect.width > 50 && rect.height > 20
        })
        .map(|(mut rect, _)| {
            rect.x = (rect.x - 8).max(0);
            rect.y = (rect.y - 8).max(0);
            rect.width = (rect.width + 16).min(img.cols() - rect.x);
            rect.height = (rect.height + 16).min(img.rows() - rect.y);
            rect
        })
        .collect();

    // sort by y-values first
    blocks.sort_by(|a, b| a.y.cmp(&b.y).then_with(|| a.x.cmp(&b.x)));

    let blocks_mat = blocks
        .into_iter()
        .map(|rect| Mat::roi(img, rect).map(|x| x.clone_pointee()))
        .collect::<Result<Vec<_>, opencv::Error>>()?;

    imwrite("morphed.png", &thresh, &Vector::new())?;

    Ok(blocks_mat)
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
    } || OcrError || IoError
    OcrError := {
        (rust_paddle_ocr::OcrError),
        (rusty_tesseract::TessError),
        #[display("No intersection points can be found between the horiziontal and veritcal lines.")]
        NoIntersection
    }
    ImageDecodeError := {
        #[display("Fails to decode image, reason: {:#?}", source)]
        (image::ImageError)
    }
}

static VERBOSITY: OnceLock<bool> = OnceLock::new();

pub fn parse_major(input: &Mat, verbosity: bool, bar: ProgressBar) -> Result<Vec<Major>, AppError> {
    set_use_opencl(true)?;

    let _ = VERBOSITY.set(verbosity);

    let mut det = Det::from_bytes(include_bytes!("../../models/PP-OCRv5_mobile_det.mnn"))?
        .with_rect_border_size(12)
        .with_merge_boxes(false)
        .with_merge_threshold(1);
    let mut rec = Rec::from_bytes_with_keys(
        include_bytes!("../../models/PP-OCRv5_mobile_rec.mnn"),
        include_bytes!("../../models/ppocr_keys_v5.txt"),
    )?
    .with_min_score(0.6)
    .with_punct_min_score(0.1);
    let mut gray = Mat::default();
    cvt_color(
        &input,
        &mut gray,
        COLOR_BGR2GRAY,
        0,
        AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    let gray = extract_blocks(&gray, 254.0)?;
    if PathBuf::from(".tmp-files").exists() {
        remove_dir_all(".tmp-files")?;
    }
    let majors = (0..3)
        .map(|i| {
            let folder =
                PathBuf::from(".tmp-files/outputs".to_string() + "-" + &(i + 1).to_string());

            if *VERBOSITY.get().unwrap() {
                create_dir_all(&folder)?;
            }

            write_intermediate(&format!("outputs-{}.png", i + 1), &gray[i])?;

            process_block(&mut det, &mut rec, &gray[i], &folder, bar.clone())
        })
        .collect::<Result<Vec<_>, AppError>>()?
        .into_iter()
        .flatten()
        .filter(|x| x.id != 333.to_string())
        .collect::<Vec<_>>();
    Ok(majors)
}

fn process_block(
    det: &mut Det,
    rec: &mut Rec,
    gray: &Mat,
    folder: &Path,
    bar: ProgressBar,
) -> Result<Vec<Major>, AppError> {
    imwrite("cleaned.png", &gray, &Vector::new())?;
    let mut gray_inv = Mat::default();
    bitwise_not(&gray, &mut gray_inv, &no_array())?;
    let mut binary = Mat::default();
    threshold(&gray_inv, &mut binary, 90.0, 255.0, THRESH_BINARY)?;
    let mut h_lines = detect_h_lines(&binary)?;
    let mut v_lines = detect_v_lines(&binary)?;

    // transpose to make columns continous
    transpose(&h_lines.clone(), &mut h_lines)?;
    for y in 0..h_lines.rows() {
        keep_middle_line(h_lines.row_mut(y)?.data_bytes_mut()?);
    }
    transpose(&h_lines.clone(), &mut h_lines)?;

    for y in 0..v_lines.rows() {
        keep_middle_line(v_lines.row_mut(y)?.data_bytes_mut()?);
    }

    let mut intersection_mat = Mat::default();
    bitwise_and(&v_lines, &h_lines, &mut intersection_mat, &no_array())?;
    let mut test = Mat::default();
    bitwise_or(&v_lines, &h_lines, &mut test, &no_array())?;
    write_intermediate("h_lines.png", &h_lines)?;
    write_intermediate("v_lines.png", &v_lines)?;
    write_intermediate("intersections-or.png", &test)?;
    write_intermediate("intersections-and.png", &intersection_mat)?;
    split_table(det, rec, gray, &intersection_mat, folder, bar).map_err(Into::into)
}

fn write_intermediate(name: &str, mat: &Mat) -> Result<(), AppError> {
    if *VERBOSITY.get().unwrap() {
        let path = PathBuf::from(".tmp-files");

        imwrite(
            path.join(name).to_string_lossy().as_ref(),
            mat,
            &Vector::new(),
        )?;
    }
    Ok(())
}

fn detect_h_lines(binary: &Mat) -> Result<Mat, AppError> {
    let dilate_kernel = get_structuring_element(MORPH_RECT, Size::new(3, 1), Point::new(-1, -1))?;
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

    let kernel_width = 60;

    let horizontal_kernel =
        get_structuring_element(MORPH_RECT, Size::new(kernel_width, 1), Point::new(-1, -1))?;

    let mut detected = Mat::default();
    morphology_ex(
        &binary_dilated,
        &mut detected,
        MORPH_OPEN,
        &horizontal_kernel,
        Point::new(-1, -1),
        1,
        BORDER_CONSTANT,
        morphology_default_border_value()?,
    )?;
    Ok(detected)
}

fn detect_v_lines(binary: &Mat) -> Result<Mat, AppError> {
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
    let kernel_height = 40;
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
    Ok(detected)
}

macro_rules! init_handle {
    ($ocr:expr, $row_counter:expr, $majors:expr) => {
        let ocr = &$ocr;
        let row_counter = $row_counter;
        let majors = &mut $majors;
        // Inner macro - generates match arm using the initialized variables
        macro_rules! handle_standard {
            ($field:ident, $ocr_test:expr) => {
                if ocr.is_some() && $ocr_test {
                    majors[row_counter].certification_standards.$field = None;
                } else {
                    majors[row_counter].certification_standards.$field =
                        ocr.as_ref().and_then(|s| s.parse().ok());
                }
            };
            ($field:ident) => {
                handle_standard!(
                    $field,
                    ocr.as_ref()
                        .map(|s| s == stringify!($field))
                        .unwrap_or(false)
                )
            };
        }
        macro_rules! handle_filters {
            ($field:ident, $ocr_test:expr) => {
                if ocr.is_some() && $ocr_test {
                    majors[row_counter].filters.$field = None;
                } else {
                    majors[row_counter].filters.$field = ocr.as_ref().and_then(|s| s.parse().ok());
                }
            };
            ($field:ident) => {
                handle_filters!(
                    $field,
                    ocr.as_ref()
                        .map(|s| s == stringify!($field))
                        .unwrap_or(false)
                )
            };
        }
    };
}

#[allow(clippy::too_many_lines)]
pub fn split_table(
    det: &mut Det,
    rec: &mut Rec,
    gray: &Mat,
    intersections: &Mat,
    folder: &Path,
    bar: ProgressBar,
) -> Result<Vec<Major>, TableSplitError> {
    let mut points_mat = Mat::default();
    find_non_zero(intersections, &mut points_mat)?;

    let mut intersections = points_mat
        .iter::<Point>()
        .into_iter()
        .flatten()
        .map(|e| (e.1.x, e.1.y))
        .collect::<Vec<_>>();

    intersections.sort_unstable();

    let table = intersections.iter().collect::<HashSet<_>>();

    let max_x = intersections
        .iter()
        .map(|(x, _)| *x)
        .max()
        .ok_or(OcrError::NoIntersection)?;
    let unique_y_values: HashSet<_> = intersections.iter().map(|x| x.1).collect();
    let rows_total_count = unique_y_values.len() - 2;

    let mut row_counter = 0;
    let mut state = gsat::State::None;
    let mut majors = vec![Major::default(); rows_total_count];

    let iter = intersections.windows(2).filter(|w| w[0].0 == w[1].0);
    let count = iter.clone().count();

    bar.set_position(0);
    bar.set_length(count as u64);

    bar.set_style(
        #[allow(clippy::literal_string_with_formatting_args)]
        ProgressStyle::with_template(
            "{msg:.magenta}\n[{human_pos}/{human_len}] [{percent}%] [{elapsed_precise}] {wide_bar:.cyan/blue}",
        )
        .expect("Invalid template(compile time issue)")
        .progress_chars("##-"),
    );

    let iter = iter.clone().progress_with(bar.clone());

    for (i, window) in iter.enumerate() {
        // prospected bounding box verticies, may be incorrect due to overlapping neighbors
        let Some(verticies) = extract_verticies(max_x, &table, window) else {
            continue;
        };

        bar.set_message(state.to_string());

        let mut bounding_box = get_biggest_possible_bounding_box(verticies);

        bounding_box.x += 1;
        bounding_box.y += 1;
        bounding_box.width -= 2;
        bounding_box.height -= 2;

        let mut cell = Mat::roi(gray, bounding_box)?.clone_pointee();

        copy_make_border(
            &cell.clone(),
            &mut cell,
            3, // top border width
            3, // bottom border width
            3, // left border width
            3, // right border width
            BORDER_CONSTANT,
            Scalar::all(255.),
        )?;

        // OCR processing
        let mut buf = Vector::new();
        imencode(".png", &cell, &mut buf, &Vector::new())?;
        let img = image::ImageReader::new(std::io::Cursor::new(buf.as_slice()))
            .with_guessed_format()?
            .decode()?;
        let ocr = paddle_ocr(det, rec, &img)?;
        let rect = det.find_text_rect(&img)?;

        // pads 0 on the left for numbers such as 001
        // let output_name = format!(
        //     "{i:0>width$}-{state}-{row_counter:0>2}",
        //     width = count.ilog10() as usize + 1
        // );
        let mut output_text_ocr = String::new();

        let ocr = if let Some(text_dimensions) = rect.first() {
            output_text_ocr = ocr.clone();
            Some(ocr)
        } else {
            output_text_ocr = String::from("--");
            None
        };

        init_handle!(ocr, row_counter, majors);

        let ocr = ocr.as_deref().filter(|s| !s.is_empty());

        match state {
            gsat::State::Id => {
                let digits = tesseract_digit_parse(&img)?;
                output_text_ocr = digits.clone();
                majors[row_counter].id = digits;
            }
            gsat::State::Gender => {
                majors[row_counter].gender_requirements = ocr.is_some_and(|x| x != "無");
                output_text_ocr = if majors[row_counter].gender_requirements {
                    String::from("有")
                } else {
                    String::from("無")
                };
            }
            gsat::State::Name => {
                majors[row_counter].name = ocr.unwrap_or_default().into();
            }
            gsat::State::Quota => {
                // let digits = tesseract_digit_parse(&img)?;
                let ocr_digit = ocr.and_then(|x| x.parse().ok()).unwrap_or_default();
                // let quota = digits.parse().map_or(ocr_digit, |x| x);
                majors[row_counter].quota = ocr_digit;
            }
            gsat::State::Standards(gsat::StandardState::None) => {}

            _ if ocr == Some("檢定標準") || ocr.is_some_and(|x| x.contains("倍率")) => {}

            gsat::State::Standards(gsat::StandardState::國文) => {
                handle_standard!(國文);
            }
            gsat::State::Standards(gsat::StandardState::英文) => {
                handle_standard!(英文);
            }
            gsat::State::Standards(gsat::StandardState::數a) => {
                handle_standard!(數a, ocr.is_some_and(|x| x.contains("數學")));
            }
            gsat::State::Standards(gsat::StandardState::數b) => {
                handle_standard!(數b, ocr.is_some_and(|x| x.contains("數學")));
            }
            gsat::State::Standards(gsat::StandardState::社會) => {
                handle_standard!(社會);
            }
            gsat::State::Standards(gsat::StandardState::自然) => {
                handle_standard!(自然, ocr.is_some_and(|x| x.contains("然")));
            }
            gsat::State::Standards(gsat::StandardState::英聽) => {
                handle_standard!(英聽);
            }
            gsat::State::Filters(FilterState::國文) => {
                handle_filters!(國文);
            }
            gsat::State::Filters(gsat::FilterState::英文) => {
                handle_filters!(英文);
            }
            gsat::State::Filters(gsat::FilterState::數a) => {
                handle_filters!(數a, ocr.is_some_and(|x| x.contains("數學")));
            }
            gsat::State::Filters(gsat::FilterState::數b) => {
                handle_filters!(數b, ocr.is_some_and(|x| x.contains("數學")));
            }
            gsat::State::Filters(gsat::FilterState::社會) => {
                handle_filters!(社會);
            }
            gsat::State::Filters(gsat::FilterState::自然) => {
                handle_filters!(自然, ocr.is_some_and(|x| x.contains("然")));
            }
            gsat::State::Filters(gsat::FilterState::學測科目組合) => {
                handle_filters!(學測科目組合, ocr.is_some_and(|x| x.contains("學測")));
            }
            gsat::State::MinimumRate(x) if x > 0 => {
                let ocr = ocr.unwrap_or_default().to_string();

                let ocr = ocr.replace(['+', '(', ')', '文', '學', '然', '會', '【', '（'], "");

                output_text_ocr = ocr.clone();
                majors[row_counter].minimum_rate[x as usize - 1] = ocr;
            }
            gsat::State::None
            | gsat::State::Filters(gsat::FilterState::None)
            | gsat::State::MinimumRate(_) => {}
        }

        // Render back the ocr-ed text
        if *VERBOSITY.get().unwrap() {
            // text_img_merger(
            //     gray,
            //     bounding_box,
            //     output_text_ocr,
            //     &folder.join(output_name.clone() + ".png"),
            // )?;
            if matches!(
                state,
                State::Standards(_) | State::Filters(_) | State::Quota | State::MinimumRate(_)
            ) {
                let output_name = format!("{}-{}", majors[row_counter].name, state);
                imwrite(
                    &folder
                        .join(output_name.clone() + ".png")
                        .to_string_lossy()
                        .as_ref(),
                    &cell,
                    &Vector::new(),
                )?;
            }
        }

        // if true {
        // }

        row_counter += 1;

        match ocr {
            Some("校系代碼") => {
                if !matches!(state, State::Id) {
                    state = gsat::State::Id;
                    row_counter = 0;
                }
            }
            Some("性別要求") => {
                if !matches!(state, State::Gender) {
                    state = gsat::State::Gender;
                    row_counter = 0;
                }
            }
            Some("校系名稱") => {
                if !matches!(state, State::Name) {
                    state = gsat::State::Name;
                    row_counter = 0;
                }
            }
            Some("招生名額") => {
                if !matches!(state, State::Quota) {
                    state = gsat::State::Quota;
                    row_counter = 0;
                }
            }
            Some("檢定標準") => {
                if !matches!(state, State::Standards(_)) {
                    state = gsat::State::Standards(gsat::StandardState::None);
                    row_counter = 0;
                }
            }
            Some(x) if x.contains("倍率") && x.contains("級分") => {
                if matches!(state, State::Filters(gsat::FilterState::學測科目組合)) {
                    state = State::MinimumRate(0);
                    row_counter = 0;
                }
            }
            Some(x) if x.contains("倍率") => {
                if matches!(state, State::Standards(StandardState::英聽)) {
                    state = State::Filters(FilterState::None);
                    row_counter = 0;
                }
            }
            Some(x) if x.contains("順序") => {
                if let State::MinimumRate(x) = state {
                    state = State::MinimumRate(x + 1);
                    row_counter = 0;
                }
            }
            Some("國文") => {
                if matches!(state, State::Standards(StandardState::None)) {
                    state = gsat::State::Standards(gsat::StandardState::國文);
                    row_counter = 0;
                } else if matches!(state, State::Filters(FilterState::None)) {
                    state = gsat::State::Filters(gsat::FilterState::國文);
                    row_counter = 0;
                }
            }
            Some("英文") => {
                if matches!(state, State::Standards(_)) {
                    state = gsat::State::Standards(gsat::StandardState::英文);
                    row_counter = 0;
                } else if matches!(state, State::Filters(_)) {
                    state = gsat::State::Filters(gsat::FilterState::英文);
                    row_counter = 0;
                }
            }
            Some(x) if x.contains("數學") && x.contains('A') => {
                if matches!(state, State::Standards(_)) {
                    state = gsat::State::Standards(gsat::StandardState::數a);
                    row_counter = 0;
                } else if matches!(state, State::Filters(_)) {
                    state = gsat::State::Filters(gsat::FilterState::數a);
                    row_counter = 0;
                }
            }
            Some(x) if x.contains("數學") && x.contains('B') => {
                if matches!(state, State::Standards(_)) {
                    state = gsat::State::Standards(gsat::StandardState::數b);
                    row_counter = 0;
                } else if matches!(state, State::Filters(_)) {
                    state = gsat::State::Filters(gsat::FilterState::數b);
                    row_counter = 0;
                }
            }
            Some("社會") => {
                if matches!(state, State::Standards(_)) {
                    state = gsat::State::Standards(gsat::StandardState::社會);
                    row_counter = 0;
                } else if matches!(state, State::Filters(_)) {
                    state = gsat::State::Filters(gsat::FilterState::社會);
                    row_counter = 0;
                }
            }
            Some("自然" | "白然") => {
                if matches!(state, State::Standards(_)) {
                    state = gsat::State::Standards(gsat::StandardState::自然);
                    row_counter = 0;
                } else if matches!(state, State::Filters(_)) {
                    state = gsat::State::Filters(gsat::FilterState::自然);
                    row_counter = 0;
                }
            }
            Some("英聽") => {
                if matches!(state, State::Standards(StandardState::自然)) {
                    state = gsat::State::Standards(gsat::StandardState::英聽);
                    row_counter = 0;
                }
            }
            Some(x) if x.contains("學測") => {
                if matches!(state, State::Filters(FilterState::自然)) {
                    state = gsat::State::Filters(gsat::FilterState::學測科目組合);
                    row_counter = 0;
                }
            }
            _ => {}
        }
    }

    Ok(majors)
}

fn text_img_merger(
    gray: &Mat,
    bounding_box: Rect_<i32>,
    output_text_ocr: String,
    path: &Path,
) -> Result<(), TableSplitError> {
    let color = image::Rgba::from([255, 255, 255, 255]);
    let mut img_buffer =
        RgbaImage::from_pixel(gray.cols() as u32, 60, image::Rgba::from([0, 0, 0, 0]));
    let font = ab_glyph::FontRef::try_from_slice(include_bytes!(
        "/usr/share/fonts/noto-cjk/NotoSansCJK.otf"
    ))
    .expect("place your font at /usr/share/fonts/noto-cjk/NotoSansCJK.otf");
    draw_text_mut(&mut img_buffer, color, 10, 10, 40., &font, &output_text_ocr);
    let text_mat = image_to_mat(img_buffer)?;
    let mut converted_cell = convert_rgba_to_bgr(gray.clone())?;
    rectangle(
        &mut converted_cell,
        bounding_box,
        Scalar::all(0.),
        2,
        LINE_8,
        0,
    )?;
    let converted_text = convert_rgba_to_bgr(text_mat)?;
    let mut result = Mat::default();
    vconcat2(&converted_text, &converted_cell, &mut result)?;
    imwrite(path.to_string_lossy().as_ref(), &result, &Vector::new())?;

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

    if rect1 > rect2 {
        rect1
    } else {
        rect2
    }
}

fn extract_verticies(
    max_x: i32,
    table: &HashSet<&(i32, i32)>,
    window: &[(i32, i32)],
) -> Option<[Point_<i32>; 4]> {
    let [top_left, bottom_left] = window else {
        unreachable!()
    };

    let top_right = (top_left.0 + 1..max_x)
        .find(|e| table.contains(&(*e, top_left.1)))
        .map(|x| (x, top_left.1));

    let bottom_right = (bottom_left.0 + 1..max_x)
        .find(|e| table.contains(&(*e, bottom_left.1)))
        .map(|x| (x, bottom_left.1));

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

fn paddle_ocr(det: &mut Det, rec: &mut Rec, image: &DynamicImage) -> Result<String, OcrError> {
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
            Scalar::all(255.0),
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
            Scalar::all(255.0),
        )?;
        padded
    } else {
        img2.clone()
    };

    let mut result = Mat::default();
    opencv::core::vconcat2(&padded1, &padded2, &mut result)?;
    Ok(result)
}

pub fn tesseract_digit_parse(img: &DynamicImage) -> Result<String, OcrError> {
    let args = Args {
        lang: "eng".to_string(),
        config_variables: HashMap::from([(
            "tessedit_char_whitelist".to_string(),
            "0123456789".to_string(),
        )]),
        psm: Some(7),
        oem: Some(3),
        ..Default::default()
    };

    let img = rusty_tesseract::Image::from_dynamic_image(img)?;
    let raw_text = rusty_tesseract::image_to_string(&img, &args)?;

    let digits_only: String = raw_text.chars().filter(char::is_ascii_digit).collect();

    Ok(digits_only)
}

fn keep_middle_line(line: &mut [u8]) {
    let line_with_index = line.iter().copied().enumerate().collect::<Vec<_>>();
    // if consectuive elements are the all 255, group them together
    let consecutive_runs = line_with_index
        .chunk_by(|(_, lhs), (_, rhs)| *lhs == 255 && *rhs == 255)
        .filter(|x| !x.iter().rev().any(|x| x.1 == 0)); // prevents boundary condition of [255, 0]
    line.fill(0);
    for stride in consecutive_runs {
        line[stride[stride.len() - 1].0] = 255;
    }
}
