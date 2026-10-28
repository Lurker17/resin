//! Gemma 4 image/video patchification. Geometry and layouts follow the Apache-2.0
//! Transformers Gemma4 image/video processors; resizing uses our shared Rust filter.

use serde::{Deserialize, Deserializer};

use super::{RgbImage, Source, Tensor, Video, resize_filter};
use crate::Error;

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct PatchConfig {
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub max_soft_tokens: Option<usize>,
    pub do_resize: bool,
    pub do_normalize: Option<bool>,
    #[serde(deserialize_with = "channels")]
    pub image_mean: [f32; 3],
    #[serde(deserialize_with = "channels")]
    pub image_std: [f32; 3],
    pub resample: u32,
}
impl Default for PatchConfig {
    fn default() -> Self {
        Self {
            patch_size: 16,
            pooling_kernel_size: 3,
            max_soft_tokens: None,
            do_resize: true,
            do_normalize: None,
            image_mean: [0.0; 3],
            image_std: [1.0; 3],
            resample: 3,
        }
    }
}
fn channels<'de, D: Deserializer<'de>>(deserializer: D) -> Result<[f32; 3], D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Channels {
        Scalar(f32),
        Rgb([f32; 3]),
    }
    Ok(match Channels::deserialize(deserializer)? {
        Channels::Scalar(value) => [value; 3],
        Channels::Rgb(values) => values,
    })
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct VideoConfig {
    #[serde(flatten)]
    pub patches: PatchConfig,
    pub do_sample_frames: bool,
    pub num_frames: Option<usize>,
    pub fps: Option<f64>,
}
impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            patches: PatchConfig::default(),
            do_sample_frames: true,
            num_frames: Some(32),
            fps: None,
        }
    }
}
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub image_processor: PatchConfig,
    pub video_processor: VideoConfig,
}

#[derive(Clone, Debug)]
pub struct Frame {
    pub timestamp: Option<String>,
    pub token_count: usize,
}
#[derive(Clone, Debug)]
pub struct Expansion {
    pub is_video: bool,
    pub frames: Vec<Frame>,
}
#[derive(Clone, Debug, Default)]
pub struct Prepared {
    pub pixel_values: Option<Tensor<f32>>,
    pub image_position_ids: Option<Tensor<i64>>,
    pub pixel_values_videos: Option<Tensor<f32>>,
    pub video_position_ids: Option<Tensor<i64>>,
    /// One expansion per source, retaining mixed image/video ordering.
    pub expansions: Vec<Expansion>,
}

impl Config {
    pub fn process(&self, sources: &[Source]) -> Result<Prepared, Error> {
        let mut prepared = Prepared::default();
        for source in sources {
            match source {
                Source::Image(image) => {
                    let (pixels, positions, token_count) =
                        self.image_processor.process(image, false)?;
                    push(&mut prepared.pixel_values, pixels)?;
                    push(&mut prepared.image_position_ids, positions)?;
                    prepared.expansions.push(Expansion {
                        is_video: false,
                        frames: vec![Frame {
                            timestamp: None,
                            token_count,
                        }],
                    });
                }
                Source::Video(video) => {
                    let indices = self.video_processor.select(video)?;
                    let mut pixels = None;
                    let mut positions = None;
                    let mut frames = Vec::with_capacity(indices.len());
                    for index in indices {
                        let (frame_pixels, frame_positions, token_count) = self
                            .video_processor
                            .patches
                            .process(&video.frames[index], true)?;
                        push(&mut pixels, frame_pixels)?;
                        push(&mut positions, frame_positions)?;
                        frames.push(Frame {
                            timestamp: Some(timestamp(video.frame_indices[index], video.fps)?),
                            token_count,
                        });
                    }
                    push(&mut prepared.pixel_values_videos, pixels.unwrap())?;
                    push(&mut prepared.video_position_ids, positions.unwrap())?;
                    prepared.expansions.push(Expansion {
                        is_video: true,
                        frames,
                    });
                }
            }
        }
        Ok(prepared)
    }
}

/// Stack a source without re-copying its flattened storage or losing shape checks.
fn push<T>(batch: &mut Option<Tensor<T>>, mut source: Tensor<T>) -> Result<(), Error> {
    if let Some(batch) = batch {
        if batch.shape[1..] != source.shape {
            return Err(Error::new(
                "Gemma media tensors in a batch must have matching shapes",
            ));
        }
        batch.shape[0] += 1;
        batch.data.append(&mut source.data);
    } else {
        source.shape.insert(0, 1);
        *batch = Some(source);
    }
    Ok(())
}
fn product(values: &[usize]) -> Result<usize, Error> {
    values
        .iter()
        .try_fold(1usize, |n, value| n.checked_mul(*value))
        .ok_or_else(|| Error::new("Gemma patch geometry overflows addressable storage"))
}

impl PatchConfig {
    fn max_patches(&self, is_video: bool) -> Result<usize, Error> {
        let tokens = self
            .max_soft_tokens
            .unwrap_or(if is_video { 70 } else { 280 });
        if ![70, 140, 280, 560, 1120].contains(&tokens) {
            return Err(Error::new(
                "Gemma max_soft_tokens must be 70, 140, 280, 560, or 1120",
            ));
        }
        if self.patch_size == 0 || self.pooling_kernel_size == 0 {
            return Err(Error::new("Gemma patch and pooling sizes must be positive"));
        }
        product(&[tokens, self.pooling_kernel_size, self.pooling_kernel_size])
    }
    fn target_size(
        &self,
        height: usize,
        width: usize,
        max_patches: usize,
    ) -> Result<(usize, usize), Error> {
        let target_pixels = product(&[max_patches, self.patch_size, self.patch_size])?;
        let side = product(&[self.patch_size, self.pooling_kernel_size])?;
        let factor = (target_pixels as f64 / (height as f64 * width as f64)).sqrt();
        let mut target_height = ((factor * height as f64 / side as f64).floor() as usize) * side;
        let mut target_width = ((factor * width as f64 / side as f64).floor() as usize) * side;
        let max_side =
            max_patches / product(&[self.pooling_kernel_size, self.pooling_kernel_size])? * side;
        if target_height == 0 && target_width == 0 {
            return Err(Error::new("Gemma resized image would have no patches"));
        }
        if target_height == 0 {
            target_height = side;
            target_width = product(&[width / height, side])?.min(max_side);
        } else if target_width == 0 {
            target_width = side;
            target_height = product(&[height / width, side])?.min(max_side);
        }
        if product(&[target_height, target_width])? > target_pixels {
            return Err(Error::new("Gemma resized image exceeds the patch budget"));
        }
        Ok((target_height, target_width))
    }
    fn process(
        &self,
        image: &RgbImage,
        is_video: bool,
    ) -> Result<(Tensor<f32>, Tensor<i64>, usize), Error> {
        let max_patches = self.max_patches(is_video)?;
        let resized = if self.do_resize {
            let (height, width) = self.target_size(image.height, image.width, max_patches)?;
            if (height, width) != (image.height, image.width) {
                Some(image.resize(width, height, resize_filter(self.resample)?)?)
            } else {
                None
            }
        } else {
            None
        };
        let image = resized.as_ref().unwrap_or(image);
        let patch = self.patch_size;
        if !image.height.is_multiple_of(patch) || !image.width.is_multiple_of(patch) {
            return Err(Error::new(
                "Gemma image dimensions must be divisible by patch_size",
            ));
        }
        let normalize = self.do_normalize.unwrap_or(is_video);
        if normalize
            && (self.image_mean.iter().any(|v| !v.is_finite())
                || self.image_std.iter().any(|v| !v.is_finite() || *v <= 0.0))
        {
            return Err(Error::new(
                "Gemma normalization needs finite means and positive standard deviations",
            ));
        }
        let (rows, cols) = (image.height / patch, image.width / patch);
        let count = product(&[rows, cols])?;
        let padded = count.max(max_patches);
        let patch_width = product(&[patch, patch, 3])?;
        let mut pixels = vec![0.0; product(&[padded, patch_width])?];
        let mut positions = vec![-1; product(&[padded, 2])?];
        for py in 0..rows {
            for px in 0..cols {
                let index = py * cols + px;
                positions[2 * index] = px as i64;
                positions[2 * index + 1] = py as i64;
                for y in 0..patch {
                    for x in 0..patch {
                        for channel in 0..3 {
                            let value = image.data
                                [((py * patch + y) * image.width + px * patch + x) * 3 + channel];
                            pixels[index * patch_width + (y * patch + x) * 3 + channel] =
                                if normalize {
                                    (value - self.image_mean[channel]) / self.image_std[channel]
                                } else {
                                    value
                                };
                        }
                    }
                }
            }
        }
        Ok((
            Tensor::new(vec![padded, patch_width], pixels)?,
            Tensor::new(vec![padded, 2], positions)?,
            count / product(&[self.pooling_kernel_size, self.pooling_kernel_size])?,
        ))
    }
}

impl VideoConfig {
    fn select(&self, video: &Video) -> Result<Vec<usize>, Error> {
        let total = video.frames.len();
        if total == 0 || video.frame_indices.len() != total {
            return Err(Error::new(
                "Gemma video needs nonempty frames and one original index per frame",
            ));
        }
        if !video.fps.is_finite() || video.fps <= 0.0 {
            return Err(Error::new("Gemma video fps must be finite and positive"));
        }
        let first = &video.frames[0];
        if video
            .frames
            .iter()
            .any(|f| (f.width, f.height) != (first.width, first.height))
        {
            return Err(Error::new(
                "Gemma video frames must have matching image dimensions",
            ));
        }
        if !self.do_sample_frames {
            return Ok((0..total).collect());
        }
        let count = match (self.num_frames, self.fps) {
            (Some(_), Some(_)) => {
                return Err(Error::new(
                    "Gemma num_frames and fps sampling are mutually exclusive",
                ));
            }
            (Some(n), None) => n,
            (None, Some(fps)) if fps.is_finite() && fps > 0.0 => {
                (total as f64 / video.fps * fps).floor() as usize
            }
            (None, Some(_)) => {
                return Err(Error::new("Gemma sampling fps must be finite and positive"));
            }
            (None, None) => total,
        };
        if count == 0 || count > total {
            return Err(Error::new(
                "Gemma sampled frame count must be positive and no larger than the video",
            ));
        }
        Ok((0..count)
            .map(|i| ((i as u128 * total as u128) / count as u128) as usize)
            .collect())
    }
}
fn timestamp(index: usize, fps: f64) -> Result<String, Error> {
    let seconds = (index as f64 / fps).floor();
    if !seconds.is_finite() || seconds >= u64::MAX as f64 {
        return Err(Error::new(
            "Gemma video timestamp is outside the supported range",
        ));
    }
    let seconds = seconds as u64;
    Ok(format!("{:02}:{:02}", seconds / 60, seconds % 60))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn config() -> Config {
        serde_json::from_value(json!({
            "image_processor": {"patch_size":2,"pooling_kernel_size":1,"max_soft_tokens":70,"do_resize":false},
            "video_processor": {"patch_size":2,"pooling_kernel_size":1,"max_soft_tokens":70,"do_resize":false,"num_frames":2}
        })).unwrap()
    }
    fn image(width: usize, height: usize) -> RgbImage {
        let n = width * height * 3;
        RgbImage::new(width, height, (0..n).map(|i| i as f32 / n as f32).collect()).unwrap()
    }
    #[test]
    fn patches_preserve_rgb_pixel_order_positions_normalization_and_padding() {
        let mut cfg = config();
        cfg.image_processor.do_normalize = Some(true);
        cfg.image_processor.image_mean = [0.1, 0.2, 0.3];
        cfg.image_processor.image_std = [0.5, 0.25, 2.0];
        let rgb = image(4, 2);
        let prepared = cfg.process(&[Source::Image(rgb.clone())]).unwrap();
        let pixels = prepared.pixel_values.unwrap();
        assert_eq!(pixels.shape, [1, 70, 12]);
        let indices = [
            0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23,
        ];
        for (actual, source) in pixels.data.iter().zip(indices) {
            let channel = source % 3;
            assert_eq!(
                *actual,
                (rgb.data[source] - cfg.image_processor.image_mean[channel])
                    / cfg.image_processor.image_std[channel]
            );
        }
        assert!(pixels.data[24..].iter().all(|v| *v == 0.0));
        let positions = prepared.image_position_ids.unwrap();
        assert_eq!(&positions.data[..4], [0, 0, 1, 0]);
        assert!(positions.data[4..].iter().all(|v| *v == -1));
        assert_eq!(prepared.expansions[0].frames[0].token_count, 2);
    }
    #[test]
    fn mixed_sources_keep_expansion_order_video_sampling_and_original_timestamps() {
        let frames = (0..5).map(|_| image(2, 2)).collect();
        let sources = [
            Source::Image(image(2, 2)),
            Source::Video(Video {
                frames,
                fps: 2.0,
                frame_indices: vec![120, 121, 124, 125, 130],
            }),
            Source::Image(image(2, 2)),
        ];
        let result = config().process(&sources).unwrap();
        assert_eq!(result.pixel_values.unwrap().shape, [2, 70, 12]);
        assert_eq!(result.pixel_values_videos.unwrap().shape, [1, 2, 70, 12]);
        assert_eq!(result.video_position_ids.unwrap().shape, [1, 2, 70, 2]);
        assert_eq!(
            result
                .expansions
                .iter()
                .map(|e| e.is_video)
                .collect::<Vec<_>>(),
            [false, true, false]
        );
        assert_eq!(
            result.expansions[1]
                .frames
                .iter()
                .map(|f| f.timestamp.as_deref().unwrap())
                .collect::<Vec<_>>(),
            ["01:00", "01:02"]
        );
    }
    #[test]
    fn aspect_geometry_matches_reference_including_extreme_ratios() {
        let cfg = PatchConfig::default();
        let max = cfg.max_patches(false).unwrap();
        for (height, width, target) in [
            (100, 100, (768, 768)),
            (640, 960, (624, 960)),
            (1, 10000, (48, 13440)),
            (10000, 1, (13440, 48)),
        ] {
            assert_eq!(cfg.target_size(height, width, max).unwrap(), target);
        }
        assert_eq!(cfg.max_patches(true).unwrap(), 630);
    }
    #[test]
    fn partial_configs_keep_distinct_image_and_video_defaults() {
        let cfg: Config = serde_json::from_value(json!({
            "image_processor": {"patch_size":1,"pooling_kernel_size":1,"do_resize":false,"image_mean":0.5},
            "video_processor": {"patch_size":1,"pooling_kernel_size":1,"do_resize":false,"image_mean":0.5,"do_sample_frames":false}
        })).unwrap();
        let image = RgbImage::new(1, 1, vec![0.25; 3]).unwrap();
        let result = cfg
            .process(&[
                Source::Image(image.clone()),
                Source::Video(Video {
                    frames: vec![image],
                    fps: 24.0,
                    frame_indices: vec![0],
                }),
            ])
            .unwrap();
        let images = result.pixel_values.unwrap();
        let videos = result.pixel_values_videos.unwrap();
        assert_eq!(images.shape, [1, 280, 3]);
        assert_eq!(videos.shape, [1, 1, 70, 3]);
        assert_eq!(&images.data[..3], &[0.25; 3]);
        assert_eq!(&videos.data[..3], &[-0.25; 3]);
    }
    #[test]
    fn malformed_geometry_video_metadata_and_sampling_fail_before_work() {
        let cfg = config();
        assert!(
            cfg.process(&[Source::Video(Video {
                frames: vec![],
                fps: 24.0,
                frame_indices: vec![]
            })])
            .is_err()
        );
        assert!(
            cfg.process(&[Source::Video(Video {
                frames: vec![image(2, 2)],
                fps: 24.0,
                frame_indices: vec![0]
            })])
            .is_err()
        );
        let mut bad = cfg.clone();
        bad.image_processor.patch_size = 0;
        assert!(bad.process(&[Source::Image(image(2, 2))]).is_err());
    }
}
