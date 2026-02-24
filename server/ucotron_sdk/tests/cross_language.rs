//! Cross-language SDK integration tests.
//!
//! These tests run against a live Ucotron server. Set `UCOTRON_TEST_SERVER_URL`
//! to the server URL (e.g., `http://127.0.0.1:8420`). If the env var is not
//! set, tests are skipped.

fn server_url() -> Option<String> {
    std::env::var("UCOTRON_TEST_SERVER_URL").ok()
}

macro_rules! skip_if_no_server {
    () => {
        match server_url() {
            Some(url) => url,
            None => {
                eprintln!("UCOTRON_TEST_SERVER_URL not set — skipping integration test");
                return;
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_cross_language_health() {
    let url = skip_if_no_server!();
    let client = ucotron_sdk::UcotronClient::new(url);

    let health = client.health().await.expect("health check failed");
    assert_eq!(health.status, "ok");
    assert!(!health.version.is_empty());
    assert!(!health.instance_id.is_empty());
}

#[tokio::test]
async fn test_cross_language_metrics() {
    let url = skip_if_no_server!();
    let client = ucotron_sdk::UcotronClient::new(url);

    let metrics = client.metrics().await.expect("metrics failed");
    assert!(!metrics.instance_id.is_empty());
}

#[tokio::test]
async fn test_cross_language_add_memory_and_search() {
    let url = skip_if_no_server!();
    let client = ucotron_sdk::UcotronClient::new(url);

    let ns = format!("rust_test_{}", std::process::id());
    let opts = ucotron_sdk::AddMemoryOptions {
        namespace: Some(ns.clone()),
        ..Default::default()
    };

    let result = client
        .add_memory("Rust SDK test: The capital of France is Paris.", opts)
        .await
        .expect("add_memory failed");

    // Verify response structure has expected fields
    let _ = result.chunk_node_ids;
    let _ = result.edges_created;

    // Search for the memory
    let search_opts = ucotron_sdk::SearchOptions {
        limit: Some(5),
        namespace: Some(ns.clone()),
        ..Default::default()
    };

    let search_result = client
        .search("capital of France", search_opts)
        .await
        .expect("search failed");

    assert!(!search_result.query.is_empty());
}

#[tokio::test]
async fn test_cross_language_augment() {
    let url = skip_if_no_server!();
    let client = ucotron_sdk::UcotronClient::new(url);

    let ns = format!("rust_aug_{}", std::process::id());
    let opts = ucotron_sdk::AugmentOptions {
        namespace: Some(ns),
        ..Default::default()
    };

    let result = client
        .augment("Tell me about artificial intelligence", opts)
        .await
        .expect("augment failed");

    // Augment should always return a context_text (possibly empty)
    let _ = result.context_text;
}

#[tokio::test]
async fn test_cross_language_learn() {
    let url = skip_if_no_server!();
    let client = ucotron_sdk::UcotronClient::new(url);

    let ns = format!("rust_learn_{}", std::process::id());
    let opts = ucotron_sdk::LearnOptions {
        namespace: Some(ns),
        ..Default::default()
    };

    let result = client
        .learn(
            "The user mentioned they prefer dark mode and use VSCode.",
            opts,
        )
        .await
        .expect("learn failed");

    let _ = result.memories_created;
    let _ = result.entities_found;
}

#[tokio::test]
async fn test_cross_language_list_memories() {
    let url = skip_if_no_server!();
    let client = ucotron_sdk::UcotronClient::new(url);

    let result = client
        .list_memories(None, None, None, None)
        .await
        .expect("list_memories failed");

    // Should return a vec (possibly empty)
    let _ = result.len();
}

#[tokio::test]
async fn test_cross_language_list_entities() {
    let url = skip_if_no_server!();
    let client = ucotron_sdk::UcotronClient::new(url);

    let opts = ucotron_sdk::EntityOptions::default();
    let result = client
        .list_entities(None, None, opts)
        .await
        .expect("list_entities failed");

    // Should return a vec (possibly empty)
    let _ = result.len();
}
