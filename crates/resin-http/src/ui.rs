pub const INDEX_HTML: &str = include_str!("index.html");

pub fn index_html(routed_dp_size: usize) -> String {
    INDEX_HTML.replace(
        "name=\"resin-dp-size\" content=\"1\"",
        &format!(
            "name=\"resin-dp-size\" content=\"{}\"",
            routed_dp_size.max(1)
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::INDEX_HTML;

    #[test]
    fn demo_can_send_openai_image_content_parts() {
        assert!(INDEX_HTML.contains("id=\"image-input\""));
        assert!(INDEX_HTML.contains("type:'image_url'"));
        assert!(INDEX_HTML.contains("image_url:{url:image.url}"));
    }
}
