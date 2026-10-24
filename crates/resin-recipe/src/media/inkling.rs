//! Inkling image patches: row-major crops, normalized padding, and a repeated time axis.
//!
//! Geometry follows Transformers' InklingImageProcessor (Apache-2.0), including its additional
//! rightmost patch column when the image width is an exact patch multiple.

use image::imageops::FilterType;
use serde::Deserialize;

use super::{RgbImage, Tensor, resize_filter};
use crate::Error;

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct Size {
    pub height: usize,
    pub width: usize,
}

impl Default for Size {
    fn default() -> Self {
        Self {
            height: 40,
            width: 40,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct Config {
    pub size: Size,
    pub do_resize: bool,
    pub do_normalize: bool,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub resample: u32,
    pub rescale_image_frac: Option<f64>,
    pub rescale_image_max_upscaled_long_edge: Option<usize>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            size: Size::default(),
            do_resize: true,
            do_normalize: true,
            image_mean: [0.481_454_66, 0.457_827_5, 0.408_210_73],
            image_std: [0.268_629_54, 0.261_302_6, 0.275_777_1],
            resample: 1,
            rescale_image_frac: None,
            rescale_image_max_upscaled_long_edge: Some(2048),
        }
    }
}

pub struct Prepared {
    pub pixel_values: Tensor<f32>,
    pub token_counts: Vec<usize>,
}

impl Config {
    fn validate(&self) -> Result<FilterType, Error> {
        if !self.do_resize || self.size.height == 0 || self.size.height != self.size.width {
            return Err(Error::new(
                "Inkling requires do_resize=true and a positive square patch size",
            ));
        }
        if self.image_mean.iter().any(|value| !value.is_finite())
            || self
                .image_std
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(Error::new(
                "Inkling normalization requires finite means and positive finite standard deviations",
            ));
        }
        if self
            .rescale_image_frac
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(Error::new(
                "Inkling rescale_image_frac must be positive and finite",
            ));
        }
        resize_filter(self.resample)
    }

    fn scaled_size(&self, image: &RgbImage) -> Result<(usize, usize), Error> {
        let Some(fraction) = self.rescale_image_frac else {
            return Ok((image.width, image.height));
        };
        let long_edge = image.width.max(image.height) as f64;
        let mut target = long_edge * fraction;
        if let Some(cap) = self.rescale_image_max_upscaled_long_edge {
            target = target.min((cap as f64).max(long_edge));
        }
        let ratio = target / long_edge;
        let dimension = |value: usize| {
            // Python's reference uses floor(x + .5), rather than ties-to-even.
            let scaled = (value as f64 * ratio + 0.5).floor();
            if !scaled.is_finite() || scaled < 1.0 || scaled >= usize::MAX as f64 {
                return Err(Error::new(
                    "Inkling rescaling produced an invalid image size",
                ));
            }
            Ok(scaled as usize)
        };
        Ok((dimension(image.width)?, dimension(image.height)?))
    }

    pub fn process(&self, images: &[RgbImage]) -> Result<Prepared, Error> {
        let filter = self.validate()?;
        let patch = self.size.height;
        let mut data = Vec::new();
        let mut token_counts = Vec::with_capacity(images.len());
        for original in images {
            let (width, height) = self.scaled_size(original)?;
            let resized;
            let image = if (width, height) == (original.width, original.height) {
                original
            } else {
                resized = original.resize(width, height, filter)?;
                &resized
            };
            let rows = height.div_ceil(patch);
            let columns = (width / patch)
                .checked_add(1)
                .ok_or_else(|| Error::new("Inkling patch grid is too large"))?;
            let count = rows
                .checked_mul(columns)
                .ok_or_else(|| Error::new("Inkling patch grid is too large"))?;
            let elements = count
                .checked_mul(2)
                .and_then(|value| value.checked_mul(patch))
                .and_then(|value| value.checked_mul(patch))
                .and_then(|value| value.checked_mul(3))
                .ok_or_else(|| Error::new("Inkling patch tensor is too large"))?;
            data.try_reserve(elements)
                .map_err(|error| Error::new(format!("cannot allocate Inkling patches: {error}")))?;
            token_counts.push(count);
            for row in 0..rows {
                for column in 0..columns {
                    let frame_start = data.len();
                    for y in 0..patch {
                        for x in 0..patch {
                            let (y, x) = (row * patch + y, column * patch + x);
                            for channel in 0..3 {
                                let value = if y < height && x < width {
                                    image.data[(y * width + x) * 3 + channel]
                                } else {
                                    -1.0
                                };
                                data.push(if self.do_normalize {
                                    (value - self.image_mean[channel]) / self.image_std[channel]
                                } else {
                                    value
                                });
                            }
                        }
                    }
                    data.extend_from_within(frame_start..);
                }
            }
        }
        Ok(Prepared {
            pixel_values: Tensor::new(vec![token_counts.iter().sum(), 2, patch, patch, 3], data)?,
            token_counts,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(width: usize, height: usize) -> RgbImage {
        RgbImage::new(width, height, vec![0.5; width * height * 3]).unwrap()
    }

    #[test]
    fn patch_counts_preserve_the_extra_right_column() {
        let images = [image(40, 40), image(80, 40), image(39, 41), image(41, 39)];
        let result = Config::default().process(&images).unwrap();
        assert_eq!(result.token_counts, [2, 3, 2, 2]);
        assert_eq!(result.pixel_values.shape, [9, 2, 40, 40, 3]);
    }

    #[test]
    fn patch_layout_repeats_hwc_pixels_and_normalizes_negative_padding() {
        let config = Config {
            size: Size {
                height: 2,
                width: 2,
            },
            image_mean: [0.5, 0.25, 0.0],
            image_std: [0.5, 0.25, 1.0],
            ..Config::default()
        };
        let source = RgbImage::new(2, 1, vec![0.0, 0.25, 0.5, 0.5, 0.75, 1.0]).unwrap();
        let result = config.process(&[source]).unwrap();
        let first = [
            -1.0, 0.0, 0.5, 0.0, 2.0, 1.0, -3.0, -5.0, -1.0, -3.0, -5.0, -1.0,
        ];
        let padded = [-3.0, -5.0, -1.0].repeat(4);
        assert_eq!(result.token_counts, [2]);
        assert_eq!(result.pixel_values.shape, [2, 2, 2, 2, 3]);
        assert_eq!(
            result.pixel_values.data,
            [first.as_slice(), &first, &padded, &padded].concat()
        );
    }

    #[test]
    fn optional_scaling_uses_half_up_rounding_and_only_caps_growth() {
        let mut config = Config {
            rescale_image_frac: Some(0.5),
            ..Config::default()
        };
        assert_eq!(config.scaled_size(&image(3, 5)).unwrap(), (2, 3));
        config.rescale_image_frac = Some(2.0);
        config.rescale_image_max_upscaled_long_edge = Some(10);
        assert_eq!(config.scaled_size(&image(6, 8)).unwrap(), (8, 10));
        assert_eq!(config.scaled_size(&image(12, 8)).unwrap(), (12, 8));
    }

    #[test]
    fn explicit_no_normalization_preserves_padding_and_input_values() {
        let config = Config {
            size: Size {
                height: 1,
                width: 1,
            },
            do_normalize: false,
            ..Config::default()
        };
        let result = config.process(&[image(1, 1)]).unwrap();
        assert_eq!(
            result.pixel_values.data,
            [vec![0.5; 6], vec![-1.0; 6]].concat()
        );
    }

    #[test]
    fn scaled_images_are_patched_at_their_resized_dimensions() {
        let config = Config {
            size: Size {
                height: 2,
                width: 2,
            },
            rescale_image_frac: Some(0.5),
            do_normalize: false,
            ..Config::default()
        };
        let result = config.process(&[image(3, 5)]).unwrap();
        assert_eq!(result.token_counts, [4]);
        assert_eq!(result.pixel_values.shape, [4, 2, 2, 2, 3]);
        for value in result.pixel_values.data {
            assert!(value == -1.0 || (value - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn invalid_patch_sizes_and_scales_are_rejected() {
        for size in [
            Size {
                height: 0,
                width: 0,
            },
            Size {
                height: 2,
                width: 3,
            },
        ] {
            assert!(
                Config {
                    size,
                    ..Config::default()
                }
                .process(&[])
                .is_err()
            );
        }
        for fraction in [0.0, -1.0, f64::INFINITY, f64::NAN] {
            assert!(
                Config {
                    rescale_image_frac: Some(fraction),
                    ..Config::default()
                }
                .process(&[])
                .is_err()
            );
        }
        assert!(
            Config {
                do_resize: false,
                ..Config::default()
            }
            .process(&[])
            .is_err()
        );
    }
}
