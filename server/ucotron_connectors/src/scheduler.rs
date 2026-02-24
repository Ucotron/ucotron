//! Connector sync scheduling with cron-based periodic execution.
//!
//! The scheduler manages periodic and on-demand sync operations for
//! configured connectors. It supports cron-based scheduling, manual
//! triggers, and tracks sync history.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio::task::JoinHandle;

use crate::connector::{ConnectorId, ContentItem, SyncCursor, SyncResult, WebhookPayload};

/// Schedule configuration for a connector's sync operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncSchedule {
    /// Connector instance ID.
    pub connector_id: ConnectorId,
    /// Cron expression for periodic sync (e.g., "0 */6 * * *" for every 6 hours).
    pub cron_expression: Option<String>,
    /// Whether this schedule is active.
    pub enabled: bool,
    /// Maximum duration for a single sync operation (seconds).
    pub timeout_secs: u64,
    /// Number of retries on sync failure.
    pub max_retries: u32,
}

impl SyncSchedule {
    /// Creates a new schedule with defaults.
    pub fn new(connector_id: impl Into<ConnectorId>) -> Self {
        Self {
            connector_id: connector_id.into(),
            cron_expression: None,
            enabled: true,
            timeout_secs: 300,
            max_retries: 3,
        }
    }

    /// Sets the cron expression.
    pub fn cron(mut self, expr: impl Into<String>) -> Self {
        self.cron_expression = Some(expr.into());
        self
    }

    /// Sets the timeout.
    pub fn timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Sets max retries.
    pub fn retries(mut self, max: u32) -> Self {
        self.max_retries = max;
        self
    }
}

/// Status of a sync operation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SyncStatus {
    /// Sync completed successfully.
    Success,
    /// Sync failed with an error.
    Failed { error: String },
    /// Sync is currently running.
    Running,
    /// Sync was cancelled.
    Cancelled,
}

/// Record of a completed sync operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncRecord {
    /// Connector instance ID.
    pub connector_id: ConnectorId,
    /// When the sync started (Unix seconds).
    pub started_at: u64,
    /// When the sync finished (Unix seconds).
    pub finished_at: Option<u64>,
    /// Number of items fetched.
    pub items_fetched: usize,
    /// Number of items skipped.
    pub items_skipped: usize,
    /// Sync status.
    pub status: SyncStatus,
}

/// In-memory scheduler state.
///
/// Tracks schedules and sync history for all connectors.
pub struct Scheduler {
    schedules: HashMap<ConnectorId, SyncSchedule>,
    history: Vec<SyncRecord>,
}

impl Scheduler {
    /// Creates a new empty scheduler.
    pub fn new() -> Self {
        Self {
            schedules: HashMap::new(),
            history: Vec::new(),
        }
    }

    /// Registers a sync schedule for a connector.
    pub fn add_schedule(&mut self, schedule: SyncSchedule) {
        self.schedules
            .insert(schedule.connector_id.clone(), schedule);
    }

    /// Removes a schedule.
    pub fn remove_schedule(&mut self, connector_id: &str) -> Option<SyncSchedule> {
        self.schedules.remove(connector_id)
    }

    /// Gets a schedule by connector ID.
    pub fn get_schedule(&self, connector_id: &str) -> Option<&SyncSchedule> {
        self.schedules.get(connector_id)
    }

    /// Lists all registered schedules.
    pub fn list_schedules(&self) -> Vec<&SyncSchedule> {
        self.schedules.values().collect()
    }

    /// Records a completed sync operation.
    pub fn record_sync(&mut self, record: SyncRecord) {
        self.history.push(record);
    }

    /// Gets sync history for a connector (most recent first).
    pub fn get_history(&self, connector_id: &str) -> Vec<&SyncRecord> {
        let mut records: Vec<_> = self
            .history
            .iter()
            .filter(|r| r.connector_id == connector_id)
            .collect();
        records.sort_by(|a, b| b.started_at.cmp(&a.started_at));
        records
    }

    /// Gets the most recent sync record for a connector.
    pub fn last_sync(&self, connector_id: &str) -> Option<&SyncRecord> {
        self.history
            .iter()
            .filter(|r| r.connector_id == connector_id)
            .max_by_key(|r| r.started_at)
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Validates a cron expression string.
///
/// Returns `Ok(())` if the expression is valid, or an error message.
pub fn validate_cron_expression(expr: &str) -> Result<(), String> {
    cron::Schedule::from_str(expr)
        .map(|_| ())
        .map_err(|e| format!("Invalid cron expression '{}': {}", expr, e))
}

/// Computes the next fire time for a cron expression as a Unix timestamp.
///
/// Returns `None` if no future occurrence exists (shouldn't happen for normal cron expressions).
pub fn next_fire_time(expr: &str) -> Option<u64> {
    let schedule = cron::Schedule::from_str(expr).ok()?;
    let next = schedule.upcoming(chrono::Utc).next()?;
    Some(next.timestamp() as u64)
}

/// Command sent to the cron scheduler loop.
#[derive(Debug)]
enum SchedulerCommand {
    /// Trigger an immediate sync for a connector.
    TriggerSync { connector_id: ConnectorId },
    /// Shut down the scheduler.
    Shutdown,
}

/// Configuration for the cron scheduler.
#[derive(Debug, Clone)]
pub struct CronSchedulerConfig {
    /// How often to check for due cron jobs (seconds).
    pub check_interval_secs: u64,
}

impl Default for CronSchedulerConfig {
    fn default() -> Self {
        Self {
            check_interval_secs: 60,
        }
    }
}

/// A type-erased sync function that performs incremental sync for a connector.
///
/// Takes a `SyncCursor` and returns a `SyncResult` or an error.
/// This avoids the dyn-incompatibility of `async fn in trait` (`Connector`).
pub type SyncFn = Arc<
    dyn Fn(SyncCursor) -> Pin<Box<dyn Future<Output = anyhow::Result<SyncResult>> + Send>>
        + Send
        + Sync,
>;

/// A type-erased webhook handler function for a connector.
///
/// Takes a `WebhookPayload` and returns parsed `ContentItem`s or an error.
/// This mirrors `SyncFn` but for webhook-driven real-time sync.
pub type WebhookFn = Arc<
    dyn Fn(WebhookPayload) -> Pin<Box<dyn Future<Output = anyhow::Result<Vec<ContentItem>>> + Send>>
        + Send
        + Sync,
>;

/// Registration info for a connector in the scheduler.
struct ConnectorEntry {
    /// The sync function (type-erased closure over Connector + ConnectorConfig).
    sync_fn: SyncFn,
    /// The webhook handler function (type-erased closure over Connector + ConnectorConfig).
    webhook_fn: Option<WebhookFn>,
    /// Whether the connector is enabled.
    enabled: bool,
}

/// Async cron-based scheduler that runs connector syncs on schedule.
///
/// The `CronScheduler` spawns a background tokio task that periodically
/// evaluates registered cron expressions and triggers connector syncs
/// when they are due. It also supports manual trigger via [`trigger_sync`].
pub struct CronScheduler {
    /// Shared state: schedules + history.
    state: Arc<RwLock<Scheduler>>,
    /// Connector registry: connector_id → entry.
    connectors: Arc<RwLock<HashMap<ConnectorId, ConnectorEntry>>>,
    /// Per-connector sync cursors (persisted across syncs).
    cursors: Arc<RwLock<HashMap<ConnectorId, SyncCursor>>>,
    /// Channel to send commands to the background loop.
    command_tx: mpsc::Sender<SchedulerCommand>,
    /// Handle to the background task (for join on shutdown).
    task_handle: Mutex<Option<JoinHandle<()>>>,
}

impl CronScheduler {
    /// Creates and starts a new cron scheduler.
    ///
    /// The scheduler starts a background tokio task that checks for due
    /// cron jobs at the configured interval.
    pub fn new(config: CronSchedulerConfig) -> Self {
        let state = Arc::new(RwLock::new(Scheduler::new()));
        let connectors: Arc<RwLock<HashMap<ConnectorId, ConnectorEntry>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let cursors = Arc::new(RwLock::new(HashMap::new()));
        let (command_tx, command_rx) = mpsc::channel(64);

        let task_handle = Self::spawn_loop(
            state.clone(),
            connectors.clone(),
            cursors.clone(),
            command_rx,
            config.check_interval_secs,
        );

        Self {
            state,
            connectors,
            cursors,
            command_tx,
            task_handle: Mutex::new(Some(task_handle)),
        }
    }

    /// Registers a connector with a type-erased sync function and schedule.
    ///
    /// The `sync_fn` encapsulates the connector and its config, avoiding
    /// dyn-compatibility issues with the `Connector` trait's async methods.
    pub async fn register(
        &self,
        connector_id: impl Into<ConnectorId>,
        sync_fn: SyncFn,
        enabled: bool,
        schedule: SyncSchedule,
    ) {
        let id = connector_id.into();
        self.connectors.write().await.insert(
            id.clone(),
            ConnectorEntry { sync_fn, webhook_fn: None, enabled },
        );
        self.state.write().await.add_schedule(schedule);
    }

    /// Registers a webhook handler for an existing connector.
    ///
    /// The `webhook_fn` encapsulates `Connector::handle_webhook()` with
    /// the connector's config pre-bound, allowing type-erased invocation.
    pub async fn register_webhook(
        &self,
        connector_id: &str,
        webhook_fn: WebhookFn,
    ) -> Result<(), String> {
        let mut connectors = self.connectors.write().await;
        if let Some(entry) = connectors.get_mut(connector_id) {
            entry.webhook_fn = Some(webhook_fn);
            Ok(())
        } else {
            Err(format!("Connector '{}' not registered", connector_id))
        }
    }

    /// Handles an incoming webhook for a connector.
    ///
    /// Looks up the registered webhook handler and invokes it with the payload.
    /// Returns the parsed content items or an error.
    pub async fn handle_webhook(
        &self,
        connector_id: &str,
        payload: WebhookPayload,
    ) -> anyhow::Result<Vec<ContentItem>> {
        let connectors = self.connectors.read().await;
        let entry = connectors.get(connector_id).ok_or_else(|| {
            anyhow::anyhow!("Connector '{}' not found", connector_id)
        })?;

        if !entry.enabled {
            return Err(anyhow::anyhow!(
                "Connector '{}' is disabled",
                connector_id
            ));
        }

        let webhook_fn = entry.webhook_fn.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "Connector '{}' does not have a webhook handler registered",
                connector_id
            )
        })?;

        let wh = webhook_fn.clone();
        // Drop the read lock before invoking the async function.
        drop(connectors);
        (wh)(payload).await
    }

    /// Checks whether a connector has a webhook handler registered.
    pub async fn has_webhook(&self, connector_id: &str) -> bool {
        self.connectors
            .read()
            .await
            .get(connector_id)
            .map(|e| e.webhook_fn.is_some())
            .unwrap_or(false)
    }

    /// Unregisters a connector.
    pub async fn unregister(&self, connector_id: &str) {
        self.connectors.write().await.remove(connector_id);
        self.state.write().await.remove_schedule(connector_id);
        self.cursors.write().await.remove(connector_id);
    }

    /// Triggers an immediate sync for a connector.
    ///
    /// Returns an error if the command channel is closed (scheduler shut down).
    pub async fn trigger_sync(&self, connector_id: &str) -> Result<(), String> {
        self.command_tx
            .send(SchedulerCommand::TriggerSync {
                connector_id: connector_id.to_string(),
            })
            .await
            .map_err(|_| "Scheduler is shut down".to_string())
    }

    /// Gets the shared scheduler state for querying schedules and history.
    pub fn state(&self) -> &Arc<RwLock<Scheduler>> {
        &self.state
    }

    /// Gets the shared cursors for inspection.
    pub fn cursors(&self) -> &Arc<RwLock<HashMap<ConnectorId, SyncCursor>>> {
        &self.cursors
    }

    /// Shuts down the scheduler gracefully.
    pub async fn shutdown(&self) {
        let _ = self.command_tx.send(SchedulerCommand::Shutdown).await;
        let mut handle = self.task_handle.lock().await;
        if let Some(h) = handle.take() {
            let _ = h.await;
        }
    }

    /// Spawns the background scheduler loop.
    fn spawn_loop(
        state: Arc<RwLock<Scheduler>>,
        connectors: Arc<RwLock<HashMap<ConnectorId, ConnectorEntry>>>,
        cursors: Arc<RwLock<HashMap<ConnectorId, SyncCursor>>>,
        mut command_rx: mpsc::Receiver<SchedulerCommand>,
        check_interval_secs: u64,
    ) -> JoinHandle<()> {
        // Track last fire time per connector to avoid double-firing.
        let last_fired: Arc<RwLock<HashMap<ConnectorId, u64>>> =
            Arc::new(RwLock::new(HashMap::new()));

        tokio::spawn(async move {
            let interval = tokio::time::Duration::from_secs(check_interval_secs);
            let mut ticker = tokio::time::interval(interval);
            // First tick fires immediately but we skip it to allow registration time.
            ticker.tick().await;

            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        Self::check_and_run_due(
                            &state,
                            &connectors,
                            &cursors,
                            &last_fired,
                        ).await;
                    }
                    cmd = command_rx.recv() => {
                        match cmd {
                            Some(SchedulerCommand::TriggerSync { connector_id }) => {
                                Self::run_sync(
                                    &connector_id,
                                    &state,
                                    &connectors,
                                    &cursors,
                                ).await;
                            }
                            Some(SchedulerCommand::Shutdown) | None => {
                                tracing::info!("Connector scheduler shutting down");
                                break;
                            }
                        }
                    }
                }
            }
        })
    }

    /// Checks all schedules and runs syncs that are due.
    async fn check_and_run_due(
        state: &Arc<RwLock<Scheduler>>,
        connectors: &Arc<RwLock<HashMap<ConnectorId, ConnectorEntry>>>,
        cursors: &Arc<RwLock<HashMap<ConnectorId, SyncCursor>>>,
        last_fired: &Arc<RwLock<HashMap<ConnectorId, u64>>>,
    ) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Collect due connector IDs under a brief read lock.
        let due_ids: Vec<ConnectorId> = {
            let sched = state.read().await;
            let fired = last_fired.read().await;
            sched
                .list_schedules()
                .iter()
                .filter(|s| s.enabled && s.cron_expression.is_some())
                .filter_map(|s| {
                    let expr = s.cron_expression.as_ref()?;
                    let last = fired.get(&s.connector_id).copied().unwrap_or(0);
                    if is_cron_due(expr, now, last) {
                        Some(s.connector_id.clone())
                    } else {
                        None
                    }
                })
                .collect()
        };

        for connector_id in due_ids {
            // Update last_fired BEFORE running to avoid double-fire.
            last_fired.write().await.insert(connector_id.clone(), now);
            Self::run_sync(&connector_id, state, connectors, cursors).await;
        }
    }

    /// Executes a single sync for a connector, recording the result.
    async fn run_sync(
        connector_id: &str,
        state: &Arc<RwLock<Scheduler>>,
        connectors: &Arc<RwLock<HashMap<ConnectorId, ConnectorEntry>>>,
        cursors: &Arc<RwLock<HashMap<ConnectorId, SyncCursor>>>,
    ) {
        let now_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Look up the connector entry.
        let (sync_fn, enabled) = {
            let conns = connectors.read().await;
            match conns.get(connector_id) {
                Some(entry) => (entry.sync_fn.clone(), entry.enabled),
                None => {
                    tracing::warn!("Sync trigger for unknown connector: {}", connector_id);
                    return;
                }
            }
        };

        if !enabled {
            tracing::debug!("Skipping disabled connector: {}", connector_id);
            return;
        }

        // Get the timeout and retries from the schedule.
        let (timeout_secs, max_retries) = {
            let sched = state.read().await;
            let schedule = sched.get_schedule(connector_id);
            (
                schedule.map(|s| s.timeout_secs).unwrap_or(300),
                schedule.map(|s| s.max_retries).unwrap_or(3),
            )
        };

        // Get or create cursor for incremental sync.
        let cursor = {
            let cur = cursors.read().await;
            cur.get(connector_id).cloned().unwrap_or_default()
        };

        tracing::info!("Starting sync for connector: {}", connector_id);

        // Record sync as running.
        {
            let mut sched = state.write().await;
            sched.record_sync(SyncRecord {
                connector_id: connector_id.to_string(),
                started_at: now_ts,
                finished_at: None,
                items_fetched: 0,
                items_skipped: 0,
                status: SyncStatus::Running,
            });
        }

        // Execute the sync with retries and timeout.
        let mut last_error = None;
        let mut items_fetched = 0;
        let mut items_skipped = 0;
        let mut new_cursor = cursor.clone();

        for attempt in 0..=max_retries {
            if attempt > 0 {
                tracing::info!(
                    "Retrying sync for {} (attempt {}/{})",
                    connector_id,
                    attempt,
                    max_retries
                );
                let backoff = tokio::time::Duration::from_secs(1 << attempt.min(5));
                tokio::time::sleep(backoff).await;
            }

            let timeout = tokio::time::Duration::from_secs(timeout_secs);
            match tokio::time::timeout(timeout, (sync_fn)(cursor.clone())).await {
                Ok(Ok(result)) => {
                    items_fetched = result.items.len();
                    items_skipped = result.skipped;
                    new_cursor = result.cursor;
                    last_error = None;
                    break;
                }
                Ok(Err(e)) => {
                    last_error = Some(format!("{}", e));
                    tracing::warn!(
                        "Sync error for {} (attempt {}): {}",
                        connector_id,
                        attempt,
                        e
                    );
                }
                Err(_) => {
                    last_error = Some("Sync timed out".to_string());
                    tracing::warn!(
                        "Sync timeout for {} after {}s (attempt {})",
                        connector_id,
                        timeout_secs,
                        attempt,
                    );
                }
            }
        }

        let finished_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let status = match &last_error {
            Some(err) => SyncStatus::Failed {
                error: err.clone(),
            },
            None => SyncStatus::Success,
        };

        // Update cursor on success.
        if last_error.is_none() {
            cursors
                .write()
                .await
                .insert(connector_id.to_string(), new_cursor);
        }

        // Record the result.
        {
            let mut sched = state.write().await;
            sched.record_sync(SyncRecord {
                connector_id: connector_id.to_string(),
                started_at: now_ts,
                finished_at: Some(finished_ts),
                items_fetched,
                items_skipped,
                status: status.clone(),
            });
        }

        match &status {
            SyncStatus::Success => {
                tracing::info!(
                    "Sync completed for {}: {} fetched, {} skipped",
                    connector_id,
                    items_fetched,
                    items_skipped
                );
            }
            SyncStatus::Failed { error } => {
                tracing::error!("Sync failed for {}: {}", connector_id, error);
            }
            _ => {}
        }
    }
}

/// Checks whether a cron expression is due based on the current time
/// and the last time it fired.
fn is_cron_due(expr: &str, now_secs: u64, last_fired_secs: u64) -> bool {
    let schedule = match cron::Schedule::from_str(expr) {
        Ok(s) => s,
        Err(_) => return false,
    };

    // Find next occurrence after `last_fired` and check if it's <= now.
    let after = chrono::DateTime::from_timestamp(last_fired_secs as i64, 0).unwrap_or_default();
    if let Some(next) = schedule.after(&after).next() {
        let next_ts = next.timestamp() as u64;
        next_ts <= now_secs
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_schedule_builder() {
        let schedule = SyncSchedule::new("conn-1")
            .cron("0 */6 * * *")
            .timeout(600)
            .retries(5);

        assert_eq!(schedule.connector_id, "conn-1");
        assert_eq!(
            schedule.cron_expression,
            Some("0 */6 * * *".to_string())
        );
        assert_eq!(schedule.timeout_secs, 600);
        assert_eq!(schedule.max_retries, 5);
        assert!(schedule.enabled);
    }

    #[test]
    fn test_scheduler_add_remove() {
        let mut scheduler = Scheduler::new();
        scheduler.add_schedule(SyncSchedule::new("conn-1"));
        scheduler.add_schedule(SyncSchedule::new("conn-2"));

        assert_eq!(scheduler.list_schedules().len(), 2);
        assert!(scheduler.get_schedule("conn-1").is_some());

        scheduler.remove_schedule("conn-1");
        assert!(scheduler.get_schedule("conn-1").is_none());
        assert_eq!(scheduler.list_schedules().len(), 1);
    }

    #[test]
    fn test_sync_history() {
        let mut scheduler = Scheduler::new();

        scheduler.record_sync(SyncRecord {
            connector_id: "conn-1".to_string(),
            started_at: 1000,
            finished_at: Some(1010),
            items_fetched: 50,
            items_skipped: 2,
            status: SyncStatus::Success,
        });

        scheduler.record_sync(SyncRecord {
            connector_id: "conn-1".to_string(),
            started_at: 2000,
            finished_at: Some(2005),
            items_fetched: 10,
            items_skipped: 0,
            status: SyncStatus::Success,
        });

        scheduler.record_sync(SyncRecord {
            connector_id: "conn-2".to_string(),
            started_at: 1500,
            finished_at: Some(1520),
            items_fetched: 100,
            items_skipped: 5,
            status: SyncStatus::Failed {
                error: "timeout".to_string(),
            },
        });

        let history = scheduler.get_history("conn-1");
        assert_eq!(history.len(), 2);
        // Most recent first
        assert_eq!(history[0].started_at, 2000);
        assert_eq!(history[1].started_at, 1000);

        let last = scheduler.last_sync("conn-1").unwrap();
        assert_eq!(last.started_at, 2000);
        assert_eq!(last.items_fetched, 10);
    }

    #[test]
    fn test_sync_status_variants() {
        assert_eq!(SyncStatus::Success, SyncStatus::Success);
        assert_ne!(
            SyncStatus::Success,
            SyncStatus::Failed {
                error: "err".to_string()
            }
        );
    }

    #[test]
    fn test_schedule_serialization() {
        let schedule = SyncSchedule::new("conn-1").cron("*/30 * * * *");
        let json = serde_json::to_string(&schedule).unwrap();
        let deserialized: SyncSchedule = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.connector_id, "conn-1");
        assert_eq!(
            deserialized.cron_expression,
            Some("*/30 * * * *".to_string())
        );
    }

    #[test]
    fn test_validate_cron_expression_valid() {
        assert!(validate_cron_expression("0 */6 * * * *").is_ok());
        assert!(validate_cron_expression("0 0 * * * *").is_ok());
        assert!(validate_cron_expression("0 30 9 * * Mon-Fri").is_ok());
    }

    #[test]
    fn test_validate_cron_expression_invalid() {
        assert!(validate_cron_expression("not a cron").is_err());
        assert!(validate_cron_expression("").is_err());
    }

    #[test]
    fn test_next_fire_time() {
        // "every minute" should always have a next fire time
        let next = next_fire_time("0 * * * * *");
        assert!(next.is_some());
        let ts = next.unwrap();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        // Next fire should be within 60 seconds from now
        assert!(ts > now || ts == now);
        assert!(ts <= now + 61);
    }

    #[test]
    fn test_next_fire_time_invalid() {
        assert!(next_fire_time("garbage").is_none());
    }

    #[test]
    fn test_cron_scheduler_config_defaults() {
        let config = CronSchedulerConfig::default();
        assert_eq!(config.check_interval_secs, 60);
    }

    #[test]
    fn test_is_cron_due() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Every second — should be due if last_fired is old enough.
        assert!(is_cron_due("* * * * * *", now, now - 60));

        // If last_fired is now, next fire is in the future.
        assert!(!is_cron_due("* * * * * *", now, now));
    }

    /// Helper to create a mock sync function for testing.
    fn mock_sync_fn(items: usize, cursor_val: &str) -> SyncFn {
        let cursor_val = cursor_val.to_string();
        Arc::new(move |_cursor: SyncCursor| {
            let cursor_val = cursor_val.clone();
            Box::pin(async move {
                Ok(SyncResult {
                    items: (0..items)
                        .map(|i| crate::connector::ContentItem {
                            content: format!("item-{}", i),
                            source: crate::connector::SourceMetadata {
                                connector_type: "mock".to_string(),
                                connector_id: "test".to_string(),
                                source_id: format!("{}", i),
                                source_url: None,
                                author: None,
                                created_at: None,
                                extra: HashMap::new(),
                            },
                            media: None,
                        })
                        .collect(),
                    cursor: SyncCursor {
                        value: Some(cursor_val),
                        last_sync: Some(12345),
                    },
                    skipped: 0,
                })
            })
        })
    }

    /// Helper to create a failing sync function for testing.
    fn failing_sync_fn(msg: &str) -> SyncFn {
        let msg = msg.to_string();
        Arc::new(move |_cursor: SyncCursor| {
            let msg = msg.clone();
            Box::pin(async move { Err(anyhow::anyhow!("{}", msg)) })
        })
    }

    #[tokio::test]
    async fn test_cron_scheduler_register_unregister() {
        let scheduler = CronScheduler::new(CronSchedulerConfig {
            check_interval_secs: 3600,
        });

        let sync_fn = mock_sync_fn(1, "cursor-1");
        let schedule = SyncSchedule::new("test-conn").cron("0 * * * * *");

        scheduler
            .register("test-conn", sync_fn, true, schedule)
            .await;

        {
            let state = scheduler.state().read().await;
            assert!(state.get_schedule("test-conn").is_some());
        }

        {
            let conns = scheduler.connectors.read().await;
            assert!(conns.contains_key("test-conn"));
        }

        scheduler.unregister("test-conn").await;

        {
            let state = scheduler.state().read().await;
            assert!(state.get_schedule("test-conn").is_none());
        }

        scheduler.shutdown().await;
    }

    #[tokio::test]
    async fn test_cron_scheduler_manual_trigger() {
        let scheduler = CronScheduler::new(CronSchedulerConfig {
            check_interval_secs: 3600,
        });

        let sync_fn = mock_sync_fn(1, "mock-cursor-1");
        let schedule = SyncSchedule::new("manual-conn");

        scheduler
            .register("manual-conn", sync_fn, true, schedule)
            .await;

        // Trigger a manual sync.
        scheduler.trigger_sync("manual-conn").await.unwrap();

        // Give the background task time to process.
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        // Check that a sync was recorded.
        let state = scheduler.state().read().await;
        let history = state.get_history("manual-conn");
        assert!(!history.is_empty());

        scheduler.shutdown().await;
    }

    #[tokio::test]
    async fn test_cron_scheduler_shutdown() {
        let scheduler = CronScheduler::new(CronSchedulerConfig {
            check_interval_secs: 1,
        });

        scheduler.shutdown().await;

        // Triggering after shutdown should fail.
        let result = scheduler.trigger_sync("nonexistent").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_cron_scheduler_trigger_unknown_connector() {
        let scheduler = CronScheduler::new(CronSchedulerConfig {
            check_interval_secs: 3600,
        });

        // Trigger for a connector that doesn't exist — should not panic.
        scheduler.trigger_sync("unknown").await.unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let state = scheduler.state().read().await;
        let history = state.get_history("unknown");
        assert!(history.is_empty());

        scheduler.shutdown().await;
    }

    #[tokio::test]
    async fn test_cron_scheduler_cursor_persistence() {
        let scheduler = CronScheduler::new(CronSchedulerConfig {
            check_interval_secs: 3600,
        });

        let sync_fn = mock_sync_fn(1, "mock-cursor-1");
        let schedule = SyncSchedule::new("cursor-conn");

        scheduler.register("cursor-conn", sync_fn, true, schedule).await;

        scheduler.trigger_sync("cursor-conn").await.unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        let cursors = scheduler.cursors().read().await;
        let cursor = cursors.get("cursor-conn");
        assert!(cursor.is_some());
        assert_eq!(
            cursor.unwrap().value,
            Some("mock-cursor-1".to_string())
        );

        scheduler.shutdown().await;
    }

    #[tokio::test]
    async fn test_cron_scheduler_disabled_connector_skipped() {
        let scheduler = CronScheduler::new(CronSchedulerConfig {
            check_interval_secs: 3600,
        });

        let sync_fn = mock_sync_fn(1, "cursor");
        let schedule = SyncSchedule::new("disabled-conn");

        scheduler.register("disabled-conn", sync_fn, false, schedule).await;

        scheduler.trigger_sync("disabled-conn").await.unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        let state = scheduler.state().read().await;
        let history = state.get_history("disabled-conn");
        assert!(history.is_empty());

        scheduler.shutdown().await;
    }

    #[tokio::test]
    async fn test_cron_scheduler_sync_failure_recorded() {
        let scheduler = CronScheduler::new(CronSchedulerConfig {
            check_interval_secs: 3600,
        });

        let sync_fn = failing_sync_fn("connection refused");
        let schedule = SyncSchedule::new("fail-conn").retries(0);

        scheduler.register("fail-conn", sync_fn, true, schedule).await;

        scheduler.trigger_sync("fail-conn").await.unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        let state = scheduler.state().read().await;
        let history = state.get_history("fail-conn");
        // Should have Running + Failed records
        assert!(!history.is_empty());
        // At least one record should be Failed
        let has_failed = history.iter().any(|r| matches!(r.status, SyncStatus::Failed { .. }));
        assert!(has_failed, "Expected at least one Failed record");

        scheduler.shutdown().await;
    }

    /// Helper to create a mock webhook handler function.
    fn mock_webhook_fn(items: usize) -> WebhookFn {
        Arc::new(move |_payload: WebhookPayload| {
            Box::pin(async move {
                Ok((0..items)
                    .map(|i| crate::connector::ContentItem {
                        content: format!("webhook-item-{}", i),
                        source: crate::connector::SourceMetadata {
                            connector_type: "mock".to_string(),
                            connector_id: "test".to_string(),
                            source_id: format!("wh-{}", i),
                            source_url: None,
                            author: None,
                            created_at: None,
                            extra: HashMap::new(),
                        },
                        media: None,
                    })
                    .collect())
            })
        })
    }

    /// Helper to create a failing webhook handler.
    fn failing_webhook_fn(msg: &str) -> WebhookFn {
        let msg = msg.to_string();
        Arc::new(move |_payload: WebhookPayload| {
            let msg = msg.clone();
            Box::pin(async move { Err(anyhow::anyhow!("{}", msg)) })
        })
    }

    #[tokio::test]
    async fn test_webhook_register_and_handle() {
        let scheduler = CronScheduler::new(CronSchedulerConfig {
            check_interval_secs: 3600,
        });

        let sync_fn = mock_sync_fn(1, "cursor");
        let schedule = SyncSchedule::new("wh-conn");
        scheduler.register("wh-conn", sync_fn, true, schedule).await;

        // Register a webhook handler.
        let wh_fn = mock_webhook_fn(3);
        scheduler.register_webhook("wh-conn", wh_fn).await.unwrap();

        assert!(scheduler.has_webhook("wh-conn").await);

        // Handle a webhook.
        let payload = WebhookPayload {
            body: b"{}".to_vec(),
            headers: HashMap::new(),
            content_type: Some("application/json".to_string()),
        };
        let items = scheduler.handle_webhook("wh-conn", payload).await.unwrap();
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].content, "webhook-item-0");

        scheduler.shutdown().await;
    }

    #[tokio::test]
    async fn test_webhook_unknown_connector() {
        let scheduler = CronScheduler::new(CronSchedulerConfig {
            check_interval_secs: 3600,
        });

        let payload = WebhookPayload {
            body: b"{}".to_vec(),
            headers: HashMap::new(),
            content_type: None,
        };
        let result = scheduler.handle_webhook("nonexistent", payload).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));

        scheduler.shutdown().await;
    }

    #[tokio::test]
    async fn test_webhook_disabled_connector() {
        let scheduler = CronScheduler::new(CronSchedulerConfig {
            check_interval_secs: 3600,
        });

        let sync_fn = mock_sync_fn(1, "cursor");
        let schedule = SyncSchedule::new("disabled-wh");
        scheduler.register("disabled-wh", sync_fn, false, schedule).await;

        let wh_fn = mock_webhook_fn(1);
        scheduler.register_webhook("disabled-wh", wh_fn).await.unwrap();

        let payload = WebhookPayload {
            body: b"{}".to_vec(),
            headers: HashMap::new(),
            content_type: None,
        };
        let result = scheduler.handle_webhook("disabled-wh", payload).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("disabled"));

        scheduler.shutdown().await;
    }

    #[tokio::test]
    async fn test_webhook_no_handler_registered() {
        let scheduler = CronScheduler::new(CronSchedulerConfig {
            check_interval_secs: 3600,
        });

        let sync_fn = mock_sync_fn(1, "cursor");
        let schedule = SyncSchedule::new("no-wh");
        scheduler.register("no-wh", sync_fn, true, schedule).await;

        assert!(!scheduler.has_webhook("no-wh").await);

        let payload = WebhookPayload {
            body: b"{}".to_vec(),
            headers: HashMap::new(),
            content_type: None,
        };
        let result = scheduler.handle_webhook("no-wh", payload).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("webhook handler"));

        scheduler.shutdown().await;
    }

    #[tokio::test]
    async fn test_webhook_register_unknown_connector() {
        let scheduler = CronScheduler::new(CronSchedulerConfig {
            check_interval_secs: 3600,
        });

        let wh_fn = mock_webhook_fn(1);
        let result = scheduler.register_webhook("nonexistent", wh_fn).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not registered"));

        scheduler.shutdown().await;
    }

    #[tokio::test]
    async fn test_webhook_handler_failure() {
        let scheduler = CronScheduler::new(CronSchedulerConfig {
            check_interval_secs: 3600,
        });

        let sync_fn = mock_sync_fn(1, "cursor");
        let schedule = SyncSchedule::new("fail-wh");
        scheduler.register("fail-wh", sync_fn, true, schedule).await;

        let wh_fn = failing_webhook_fn("invalid payload");
        scheduler.register_webhook("fail-wh", wh_fn).await.unwrap();

        let payload = WebhookPayload {
            body: b"bad".to_vec(),
            headers: HashMap::new(),
            content_type: None,
        };
        let result = scheduler.handle_webhook("fail-wh", payload).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("invalid payload"));

        scheduler.shutdown().await;
    }

    #[tokio::test]
    async fn test_webhook_with_headers() {
        let scheduler = CronScheduler::new(CronSchedulerConfig {
            check_interval_secs: 3600,
        });

        let sync_fn = mock_sync_fn(1, "cursor");
        let schedule = SyncSchedule::new("header-wh");
        scheduler.register("header-wh", sync_fn, true, schedule).await;

        // Webhook handler that inspects headers.
        let wh_fn: WebhookFn = Arc::new(|payload: WebhookPayload| {
            Box::pin(async move {
                let event = payload.headers.get("x-github-event")
                    .cloned()
                    .unwrap_or_default();
                Ok(vec![crate::connector::ContentItem {
                    content: format!("event: {}", event),
                    source: crate::connector::SourceMetadata {
                        connector_type: "github".to_string(),
                        connector_id: "header-wh".to_string(),
                        source_id: "1".to_string(),
                        source_url: None,
                        author: None,
                        created_at: None,
                        extra: HashMap::new(),
                    },
                    media: None,
                }])
            })
        });
        scheduler.register_webhook("header-wh", wh_fn).await.unwrap();

        let mut headers = HashMap::new();
        headers.insert("x-github-event".to_string(), "push".to_string());
        let payload = WebhookPayload {
            body: b"{}".to_vec(),
            headers,
            content_type: Some("application/json".to_string()),
        };
        let items = scheduler.handle_webhook("header-wh", payload).await.unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].content, "event: push");

        scheduler.shutdown().await;
    }
}
