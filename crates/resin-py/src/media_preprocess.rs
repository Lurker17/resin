//! Native recipe media preparation. Python is only the input/output buffer boundary.

use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::{
    prelude::*,
    types::{PyDict, PyTuple},
};
use pythonize::depythonize;
use resin_recipe::{
    Encoder, Error, MediaConfig,
    media::{self, RgbImage, Source, Video},
};
use serde_json::{Value, json};

use crate::{
    media_payload::{
        DeepSeekV41Image, DeepSeekV41Media, Gemma4Media, InklingMedia, Qwen35Media, error,
    },
    recipe_media::MediaTokens,
    recipe_types::MediaSources,
};

enum Payload {
    DeepSeek(media::deepseek::Prepared, u32),
    Gemma(media::gemma::Prepared, u32, u32),
    Inkling(media::inkling::Prepared, u32),
    Qwen(media::qwen::Prepared, usize, u32),
}
pub(crate) struct Prepared {
    pub(crate) expansions: Vec<MediaTokens>,
    payload: Payload,
}

pub(crate) fn config_of(owner: &Bound<'_, PyAny>, name: &str) -> PyResult<Value> {
    let Some(processor) = crate::recipe::optional_attr(owner, name)? else {
        return Ok(json!({}));
    };
    if processor.is_none() || !processor.hasattr("to_dict")? {
        return Ok(json!({}));
    }
    Ok(depythonize(&processor.call_method0("to_dict")?)?)
}
pub(crate) fn process(
    config: &MediaConfig,
    sources: Vec<Source>,
    encoder: &dyn Encoder,
) -> Result<Prepared, Error> {
    let mut expansions = Vec::new();
    if let MediaConfig::Gemma(config) = config {
        let prepared = config.process(&sources)?;
        let image_id = encoder.special_id("<|image|>")?;
        let video_id = encoder.special_id("<|video|>")?;
        let boi = encoder.special_id("<|image>")?;
        let eoi = encoder.special_id("<image|>")?;
        for expansion in &prepared.expansions {
            let mut ids = Vec::new();
            let mut prefix = String::new();
            for (index, frame) in expansion.frames.iter().enumerate() {
                if let Some(timestamp) = &frame.timestamp {
                    let text = format!("{timestamp} ");
                    if index == 0 {
                        prefix = text;
                    } else {
                        ids.extend(encoder.encode_text(&format!(" {text}"))?);
                    }
                }
                ids.push(boi);
                ids.extend(std::iter::repeat_n(
                    if expansion.is_video {
                        video_id
                    } else {
                        image_id
                    },
                    frame.token_count,
                ));
                ids.push(eoi);
            }
            expansions.push(MediaTokens::from_parts(
                if expansion.is_video {
                    "<|video|>"
                } else {
                    "<|image|>"
                },
                ids,
                prefix,
                String::new(),
            ));
        }
        return Ok(Prepared {
            expansions,
            payload: Payload::Gemma(prepared, image_id, video_id),
        });
    }
    let images = sources
        .into_iter()
        .map(|source| match source {
            Source::Image(image) => Ok(image),
            Source::Video(_) => Err(Error::new("this recipe does not support video input")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (counts, marker, payload) = match config {
        MediaConfig::DeepSeek(config, expected) => {
            let id = encoder.special_id("<｜deepseek_image｜>")?;
            if expected.is_some_and(|v| v != id) {
                return Err(Error::new(
                    "tokenizer image token does not match checkpoint image_token_id",
                ));
            }
            let prepared = config.process(&images)?;
            (
                prepared.token_counts(),
                "<｜deepseek_image｜>",
                Payload::DeepSeek(prepared, id),
            )
        }
        MediaConfig::Inkling(config) => {
            let id = encoder.special_id("<|unused_200054|>")?;
            let prepared = config.process(&images)?;
            (
                prepared.token_counts.clone(),
                "<|unused_200054|>",
                Payload::Inkling(prepared, id),
            )
        }
        MediaConfig::Qwen(config) => {
            let id = encoder.special_id("<|image_pad|>")?;
            let prepared = config.process(&images)?;
            (
                prepared.token_counts.clone(),
                "<|image_pad|>",
                Payload::Qwen(prepared, config.merge_size, id),
            )
        }
        MediaConfig::Gemma(_) => unreachable!(),
    };
    let id = encoder.special_id(marker)?;
    for count in counts {
        expansions.push(MediaTokens::from_parts(
            marker,
            vec![id; count],
            String::new(),
            String::new(),
        ));
    }
    Ok(Prepared {
        expansions,
        payload,
    })
}
impl Prepared {
    pub(crate) fn bind(
        self,
        py: Python<'_>,
        token_ids: &[u32],
        spans: &[(usize, usize)],
    ) -> PyResult<Py<PyAny>> {
        let tokens = token_ids.to_vec();
        Ok(match self.payload {
            Payload::Inkling(data, marker) => Py::new(
                py,
                InklingMedia::build(py, tokens, marker, Some(data.pixel_values))?,
            )?
            .into_any(),
            Payload::Qwen(data, merge, marker) => {
                let positions = spans.iter().flat_map(|&(p, n)| p..p + n).collect();
                Py::new(
                    py,
                    Qwen35Media::build(
                        py,
                        tokens,
                        positions,
                        data.pixel_values,
                        data.image_grid_thw,
                        merge,
                        marker,
                    )?,
                )?
                .into_any()
            }
            Payload::Gemma(data, image_id, video_id) => {
                let mut types = vec![0u8; tokens.len()];
                for &(position, length) in spans {
                    for i in position..position + length {
                        types[i] = if tokens[i] == image_id {
                            1
                        } else if tokens[i] == video_id {
                            2
                        } else {
                            0
                        };
                    }
                }
                Py::new(
                    py,
                    Gemma4Media::build(
                        py,
                        tokens,
                        types,
                        data.pixel_values,
                        data.image_position_ids,
                        data.pixel_values_videos,
                        data.video_position_ids,
                    )?,
                )?
                .into_any()
            }
            Payload::DeepSeek(data, marker) => {
                let images = data
                    .images
                    .into_iter()
                    .zip(spans)
                    .map(|(image, &(start, _))| {
                        Py::new(
                            py,
                            DeepSeekV41Image::build(
                                py,
                                start,
                                image.patches,
                                image.n_vit_h,
                                image.n_vit_w,
                                image.types,
                            )?,
                        )
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                Py::new(py, DeepSeekV41Media::build(py, tokens, images, marker)?)?.into_any()
            }
        })
    }
}

pub(crate) fn image_source(py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<RgbImage> {
    if let Ok(value) = source.extract::<String>() {
        return py.detach(|| RgbImage::data_url(&value)).map_err(error);
    }
    if let Ok(pair) = source.cast::<PyTuple>()
        && pair.len() == 2
    {
        let data: Vec<u8> = pair.get_item(0)?.extract()?;
        let mime: String = pair.get_item(1)?.extract()?;
        return py.detach(|| RgbImage::decode(&data, &mime)).map_err(error);
    }
    let array = source
        .extract::<PyReadonlyArrayDyn<'_, f32>>()
        .map_err(|_| error("images must be numpy arrays with dtype float32"))?;
    let shape = array.shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(error("images must have RGB/HWC shape [height, width, 3]"));
    }
    RgbImage::new(
        shape[1],
        shape[0],
        array.as_array().iter().copied().collect(),
    )
    .map_err(error)
}
fn video_source(py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<Video> {
    let (frames, fps, indices) = if let Ok(dict) = source.cast::<PyDict>() {
        (
            dict.get_item("frames")?
                .ok_or_else(|| error("video requires decoded frames"))?,
            dict.get_item("fps")?
                .map(|v| v.extract::<f64>())
                .transpose()?
                .unwrap_or(24.0),
            dict.get_item("frame_indices")?
                .map(|v| v.extract::<Vec<usize>>())
                .transpose()?,
        )
    } else {
        (source.clone(), 24.0, None)
    };
    if frames.is_instance_of::<pyo3::types::PyString>()
        || frames.is_instance_of::<pyo3::types::PyBytes>()
    {
        return Err(error("video content must contain decoded frames"));
    }
    let mut images = Vec::new();
    for frame in frames.try_iter()? {
        let frame = frame?;
        if let Ok(array) = frame.extract::<PyReadonlyArrayDyn<'_, u8>>() {
            let shape = array.shape();
            if shape.len() != 3 || shape[2] != 3 {
                return Err(error("video frames must be RGB/HWC"));
            }
            images.push(
                RgbImage::new(
                    shape[1],
                    shape[0],
                    array
                        .as_array()
                        .iter()
                        .map(|&v| f32::from(v) / 255.)
                        .collect(),
                )
                .map_err(error)?,
            );
        } else {
            images.push(image_source(py, &frame)?);
        }
    }
    let frame_indices = indices.unwrap_or_else(|| (0..images.len()).collect());
    if images.is_empty() || !fps.is_finite() || fps <= 0.0 || frame_indices.len() != images.len() {
        return Err(error("video frames, fps and frame indices are invalid"));
    }
    Ok(Video {
        frames: images,
        fps,
        frame_indices,
    })
}
pub(crate) fn sources(py: Python<'_>, sources: MediaSources) -> PyResult<Vec<Source>> {
    sources
        .into_iter()
        .map(|(kind, source)| match kind {
            "image" => image_source(py, source.bind(py)).map(Source::Image),
            "video" => video_source(py, source.bind(py)).map(Source::Video),
            _ => Err(error("this recipe does not support audio input")),
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (height, width) -> "tuple[int, ...]")]
pub(crate) fn image_token_types(
    py: Python<'_>,
    height: usize,
    width: usize,
) -> PyResult<Py<PyTuple>> {
    // Each cell yields one type, each row a separator, plus image boundaries.
    if height == 0
        || width == 0
        || width
            .checked_add(1)
            .and_then(|w| height.checked_mul(w))
            .and_then(|n| n.checked_add(2))
            .is_none_or(|n| n > 1_048_576)
    {
        return Err(error(
            "image grid must be positive and contain at most 1048576 tokens",
        ));
    }
    Ok(PyTuple::new(py, media::deepseek::image_token_types(height, width))?.unbind())
}
