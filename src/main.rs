use std::collections::HashMap;

use error_set::error_set;
use image::{DynamicImage, GenericImageView, RgbaImage};
use imageproc::drawing::draw_text_mut;
use opencv::{
    core::*,
    imgcodecs::*,
    imgproc::*,
    prelude::*,
    ximgproc::{self, THINNING_ZHANGSUEN},
};
// use  as OcrPaddleError;
use rust_paddle_ocr::{Det, Rec};
use rusty_tesseract::{Args, Image};

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
    let input = imread("011.png", IMREAD_COLOR)?;
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

    let mut intersection = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let lhs = v_lines.at_2d::<u8>(y, x)?;
            let rhs = h_lines.at_2d::<u8>(y, x)?;
            if lhs == rhs && *lhs == 255 {
                intersection.push((y, x));
            }
        }
    }

    split_table(&mut det, &mut rec, &gray, &mut intersection)?;

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
    let kernel_sizes_v = vec![height / 15, height / 25, height / 35, height / 50];
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

// fn keep_middle_line(line: &mut [u8]) {
//     let line_with_index = line.iter().copied().enumerate().collect::<Vec<_>>();

//     // if consectuive elements are all 255, group them together
//     let consecutive_runs = line_with_index
//         .chunk_by(|(_, lhs), (_, rhs)| lhs == rhs)
//         .filter(|chunk| chunk[0].1 == 255);

//     line.fill(0);

//     for stride in consecutive_runs {
//         line[stride[stride.len() / 2].0] = 255;
//     }
// }

pub fn split_table(
    det: &mut Det,
    rec: &mut Rec,
    img: &Mat,
    intersections: &mut [(i32, i32)],
) -> Result<(), TableSplitError> {
    intersections.sort_unstable();

    let max_y = intersections.iter().map(|(x, _)| *x).max().unwrap() as usize;
    let max_x = intersections.iter().map(|(_, y)| *y).max().unwrap() as usize;

    let mut table = vec![vec![false; max_x + 1]; max_y + 1];
    for a in intersections.iter() {
        table[a.0 as usize][a.1 as usize] = true;
    }

    for (i, window) in intersections
        .windows(2)
        .filter(|w| w[0].0 == w[1].0)
        .enumerate()
    {
        let Some((top_left, top_right, bottom_left, bottom_right)) =
            extract_verticies(max_y, &table, window)
        else {
            continue;
        };

        let mut rect = get_biggest_bounding_box(top_left, top_right, bottom_left, bottom_right);

        // widen a bit to fix ocr inaccuracies
        rect.height += 1;
        rect.y -= 1;
        rect.width += 1;
        rect.x -= 1;

        let cell = Mat::roi(img, rect)?.clone_pointee();
        let mut buf = Vector::new();
        imencode(".png", &cell, &mut buf, &Vector::new())?;
        let img = image::ImageReader::new(std::io::Cursor::new(buf.as_slice()))
            .with_guessed_format()
            .expect("should be guarnateed a .png file")
            .decode()?;
        let ocr = paddle_ocr(det, rec, &img)?;
        let rect = det.find_text_rect(&img)?;
        let Some(text_dimensions) = rect.first() else {
            imwrite(&format!("outputs/{i}.png"), &cell, &Vector::new())?;
            continue;
        };
        let text_widths = rect.iter().map(imageproc::rect::Rect::width).sum::<u32>();

        let font_scale = text_dimensions.height() as f32 - 4.0;
        let color = image::Rgba::from([255, 255, 255, 255]);
        let mut img_buffer = RgbaImage::new(text_widths, text_dimensions.height());
        let font = ab_glyph::FontRef::try_from_slice(include_bytes!(
            "/usr/share/fonts/noto-cjk/NotoSansCJK.otf"
        ))
        .expect("place your font at /usr/share/fonts/noto-cjk/NotoSansCJK.otf");

        draw_text_mut(&mut img_buffer, color, 0, 0, font_scale, &font, &ocr);

        let text_mat = image_to_mat(img_buffer)?;

        let converted_cell = convert_rgba_to_bgr(cell)?;
        let converted_text = convert_rgba_to_bgr(text_mat)?;

        let result = vconcat_with_horizontal_pad(&converted_cell, &converted_text).unwrap();

        imwrite(&format!("outputs/{i}.png"), &result, &Vector::new())?;
    }

    Ok(())
}

fn get_biggest_bounding_box(
    top_left: (i32, i32),
    top_right: (i32, i32),
    bottom_left: (i32, i32),
    bottom_right: (i32, i32),
) -> Rect_<i32> {
    let rect1 = Rect::from_points(
        (bottom_left.1, bottom_left.0).into(),
        (top_right.1, top_right.0).into(),
    );

    let rect2 = Rect::from_points(
        (top_left.1, top_left.0).into(),
        (bottom_right.1, bottom_right.0).into(),
    );

    let rect = if rect1 > rect2 { rect1 } else { rect2 };
    rect
}

fn extract_verticies(
    max_y: usize,
    table: &[Vec<bool>],
    window: &[(i32, i32)],
) -> Option<((i32, i32), (i32, i32), (i32, i32), (i32, i32))> {
    let [top_left, top_right] = window else {
        unreachable!()
    };

    let mut bottom_left = None;
    for y in top_left.0 as usize + 1..max_y {
        if table[y][top_left.1 as usize] {
            bottom_left = Some((y as i32, top_left.1));
            break;
        }
    }
    let mut bottom_right = None;
    for y in top_right.0 as usize + 1..max_y {
        if table[y][top_right.1 as usize] {
            bottom_right = Some((y as i32, top_right.1));
            break;
        }
    }
    Some((*top_left, *top_right, bottom_left?, bottom_right?))
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
