pub const INDEX_HTML: &str = include_str!("index.html");

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
