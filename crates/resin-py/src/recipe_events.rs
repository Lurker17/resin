//! Typed response events; native streaming never materializes Python dictionaries.

use pyo3::{prelude::*, types::PyTuple};
use resin_recipe::{AttributedEvent, Event};

type SpanTuple = (u32, u32, u32, u32, u32);

/// A semantic parser event with token-to-text byte provenance.
///
/// Empty token_spans identify synthetic text or unknown provenance.
#[pyclass(frozen, subclass, eq, module = "resin._resin", skip_from_py_object)]
#[derive(PartialEq, Eq)]
pub(crate) struct ParserEvent {
    pub(crate) inner: AttributedEvent,
}

#[pymethods]
impl ParserEvent {
    #[getter]
    fn r#type(&self) -> &'static str {
        match self.inner.event {
            Event::TextDelta { .. } => "text_delta",
            Event::ReasoningDelta { .. } => "reasoning_delta",
            Event::ToolCallStart { .. } => "tool_call_start",
            Event::ToolCallArgumentsDelta { .. } => "tool_call_arguments_delta",
        }
    }

    #[getter]
    fn token_spans(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let spans = self.inner.spans.iter().map(|span| {
            (
                span.token,
                span.source.start,
                span.source.end,
                span.text.start,
                span.text.end,
            )
        });
        Ok(PyTuple::new(py, spans)?.unbind())
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

macro_rules! event_class {
    ($class:ident { $($field:ident: $field_type:ty),+ }) => {
        #[pyclass(frozen, extends=ParserEvent, module="resin._resin")]
        pub(crate) struct $class;

        #[pymethods]
        impl $class {
            #[new]
            #[pyo3(signature=($($field),+, *, token_spans=Vec::new()))]
            fn new($($field: $field_type),+, token_spans: Vec<SpanTuple>) -> PyClassInitializer<Self> {
                PyClassInitializer::from(ParserEvent {
                    inner: AttributedEvent {
                        event: Event::$class { $($field),+ },
                        spans: token_spans.into_iter().map(Into::into).collect(),
                    },
                }).add_subclass(Self)
            }

            $(
                #[getter]
                fn $field(slf: PyRef<'_, Self>) -> $field_type {
                    match &slf.into_super().inner.event {
                        Event::$class { $field, .. } => $field.clone(),
                        _ => unreachable!("native event subtype matches its variant"),
                    }
                }
            )+
        }
    };
}

event_class!(TextDelta { delta: String });
event_class!(ReasoningDelta { delta: String });
event_class!(ToolCallStart {
    index: usize,
    id: String,
    name: String
});
event_class!(ToolCallArgumentsDelta {
    index: usize,
    delta: String
});

pub(crate) fn to_python(
    py: Python<'_>,
    events: Vec<AttributedEvent>,
) -> PyResult<Vec<Py<ParserEvent>>> {
    events
        .into_iter()
        .map(|inner| {
            // Inspect before moving so every string/span buffer is transferred without cloning.
            let kind = match inner.event {
                Event::TextDelta { .. } => 0,
                Event::ReasoningDelta { .. } => 1,
                Event::ToolCallStart { .. } => 2,
                Event::ToolCallArgumentsDelta { .. } => 3,
            };
            let base = PyClassInitializer::from(ParserEvent { inner });
            Ok(match kind {
                0 => Bound::new(py, base.add_subclass(TextDelta))?
                    .into_super()
                    .unbind(),
                1 => Bound::new(py, base.add_subclass(ReasoningDelta))?
                    .into_super()
                    .unbind(),
                2 => Bound::new(py, base.add_subclass(ToolCallStart))?
                    .into_super()
                    .unbind(),
                _ => Bound::new(py, base.add_subclass(ToolCallArgumentsDelta))?
                    .into_super()
                    .unbind(),
            })
        })
        .collect()
}

pub(crate) fn from_python(events: &Bound<'_, PyAny>) -> PyResult<Vec<AttributedEvent>> {
    let py = events.py();
    Ok(events
        .extract::<Vec<Py<ParserEvent>>>()?
        .iter()
        .map(|event| event.borrow(py).inner.clone())
        .collect())
}
