//! The shared HTTP frontend keeps client channels; replica sessions own inference.

use std::{collections::HashMap, thread, time::Duration};

use pyo3::prelude::*;
use resin_http::{GenerationEvent, GenerationJob};
use tokio::sync::{mpsc, watch};

pub(crate) fn run(
    coordinator: Py<PyAny>,
    mut jobs: mpsc::Receiver<GenerationJob>,
    shutdown: watch::Receiver<bool>,
    replicas: usize,
) -> Result<(), String> {
    let mut clients = HashMap::<String, mpsc::Sender<GenerationEvent>>::new();
    let mut disconnected = false;
    let mut canceled_next = Vec::new();
    loop {
        let mut admissions = Vec::new();
        let mut canceled = std::mem::take(&mut canceled_next);
        canceled.extend(
            clients
                .iter()
                .filter_map(|(id, client)| client.is_closed().then_some(id.clone())),
        );
        for id in &canceled {
            clients.remove(id);
        }
        for _ in 0..super::http::ADMISSIONS_PER_STEP.saturating_mul(replicas) {
            if *shutdown.borrow()
                || clients.len() >= super::http::MAX_ACTIVE_REQUESTS.saturating_mul(replicas)
            {
                break;
            }
            match jobs.try_recv() {
                Ok(job) if !job.events.is_closed() => {
                    clients.insert(job.request.id.clone(), job.events);
                    admissions.push(job.request);
                }
                Ok(_) => {}
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }
        let stopping = disconnected || *shutdown.borrow();
        let requests = serde_json::to_string(&admissions).map_err(|error| error.to_string())?;
        let result = Python::attach(|py| {
            let object = coordinator.bind(py);
            let output: String = object
                .call_method1("round", (requests, canceled, stopping))?
                .extract()?;
            let closed: bool = object.getattr("closed")?.extract()?;
            Ok::<_, PyErr>((output, closed))
        });
        let (output, closed) = match result {
            Ok(output) => output,
            Err(error) => {
                let message = format!("distributed HTTP execution failed: {error}");
                for client in clients.values() {
                    let _ = client.try_send(GenerationEvent::Error {
                        message: message.clone(),
                    });
                }
                return Err(message);
            }
        };
        let events: Vec<(String, GenerationEvent)> = serde_json::from_str(&output)
            .map_err(|error| format!("invalid replica HTTP events: {error}"))?;
        for (id, event) in events {
            let terminal = matches!(
                event,
                GenerationEvent::Finished { .. } | GenerationEvent::Error { .. }
            );
            if let Some(client) = clients.get(&id) {
                // Backpressure remains local to one client. Cancel its owner
                // on the next round instead of delaying every active replica.
                if client.try_send(event).is_err() {
                    clients.remove(&id);
                    if !terminal {
                        canceled_next.push(id.clone());
                    }
                }
            }
            if terminal {
                clients.remove(&id);
            }
        }
        if closed {
            return Ok(());
        }
        if clients.is_empty() {
            thread::sleep(Duration::from_millis(1));
        }
    }
}
