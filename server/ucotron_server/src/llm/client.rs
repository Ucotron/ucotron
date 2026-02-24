//! LLM Client Abstraction
//!
//! Provides a unified interface for interacting with different LLM providers.
//! Supports OpenAI, Anthropic, Fireworks, and Custom providers.

use std::pin::Pin;

use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use reqwest::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::error::AppError;

const OPENAI_API_BASE: &str = "https://api.openai.com/v1";
const ANTHROPIC_API_BASE: &str = "https://api.anthropic.com/v1";
const FIREWORKS_API_BASE: &str = "https://api.fireworks.ai/inference/v1";
const ANTHROPIC_API_VERSION: &str = "2023-06-01";

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct LLMResponse {
    pub content: String,
    pub model: String,
    pub provider: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct StreamingChunk {
    pub content: String,
    pub delta: String,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CompletionRequest {
    pub messages: Vec<ChatMessage>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Serialize)]
struct OpenAIMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

#[derive(Debug, Serialize)]
struct OpenAIChatRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct OpenAIChatResponse {
    id: String,
    model: String,
    choices: Vec<OpenAIChoice>,
    usage: OpenAIUsage,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessageContent,
    finish_reason: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIMessageContent {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct OpenAISSEvent {
    #[serde(rename = "choices")]
    choices: Vec<OpenAISSEChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAISSEChoice {
    delta: OpenAISSDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAISSDelta {
    content: Option<String>,
}

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: u32,
    temperature: Option<f32>,
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    id: String,
    type_: String,
    model: String,
    content: Vec<AnthropicContentBlock>,
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
struct AnthropicContentBlock {
    #[serde(rename = "type")]
    type_: String,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct AnthropicSSEvent {
    #[serde(rename = "type")]
    type_: String,
    delta: Option<AnthropicDelta>,
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
struct AnthropicDelta {
    #[serde(rename = "text")]
    text: Option<String>,
}

pub type BoxFuture<'a, T> = Pin<Box<dyn futures_util::Future<Output = T> + Send + 'a>>;
pub type BoxStream<'a, T> = Pin<Box<dyn Stream<Item = T> + Send + 'a>>;

#[async_trait]
pub trait LLMClient: Send + Sync {
    async fn complete(&self, request: CompletionRequest) -> Result<LLMResponse, AppError>;
    async fn complete_streaming(
        &self,
        request: CompletionRequest,
    ) -> Result<BoxStream<'static, Result<StreamingChunk, AppError>>, AppError>;
}

pub struct OpenAIClient {
    api_key: String,
    api_base_url: String,
    default_model: String,
    client: Client,
}

impl OpenAIClient {
    pub fn new(api_key: String, api_base_url: String, default_model: String) -> Self {
        let base_url = if api_base_url.is_empty() {
            OPENAI_API_BASE.to_string()
        } else {
            api_base_url
        };
        Self {
            api_key,
            api_base_url: base_url,
            default_model,
            client: Client::new(),
        }
    }
}

#[async_trait]
impl LLMClient for OpenAIClient {
    async fn complete(&self, request: CompletionRequest) -> Result<LLMResponse, AppError> {
        let model = if request.model.is_empty() {
            self.default_model.clone()
        } else {
            request.model.clone()
        };

        let messages: Vec<OpenAIMessage> = request
            .messages
            .into_iter()
            .map(|m| OpenAIMessage {
                role: m.role,
                content: m.content,
                name: m.name,
            })
            .collect();

        let chat_request = OpenAIChatRequest {
            model: model.clone(),
            messages,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: false,
        };

        let url = format!("{}/chat/completions", self.api_base_url);
        let response = self
            .client
            .post(&url)
            .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
            .header(CONTENT_TYPE, "application/json")
            .json(&chat_request)
            .send()
            .await
            .map_err(|e| AppError::internal(format!("OpenAI request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(AppError::internal(format!(
                "OpenAI API error ({}): {}",
                status, body
            )));
        }

        let chat_response: OpenAIChatResponse = response
            .json()
            .await
            .map_err(|e| AppError::internal(format!("Failed to parse OpenAI response: {}", e)))?;

        let choice = chat_response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| AppError::internal("No completion choices returned"))?;

        Ok(LLMResponse {
            content: choice.message.content,
            model,
            provider: "openai".to_string(),
            input_tokens: chat_response.usage.prompt_tokens,
            output_tokens: chat_response.usage.completion_tokens,
            finish_reason: choice.finish_reason,
        })
    }

    async fn complete_streaming(
        &self,
        request: CompletionRequest,
    ) -> Result<BoxStream<'static, Result<StreamingChunk, AppError>>, AppError> {
        let model = if request.model.is_empty() {
            self.default_model.clone()
        } else {
            request.model.clone()
        };

        let messages: Vec<OpenAIMessage> = request
            .messages
            .into_iter()
            .map(|m| OpenAIMessage {
                role: m.role,
                content: m.content,
                name: m.name,
            })
            .collect();

        let chat_request = OpenAIChatRequest {
            model: model.clone(),
            messages,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: true,
        };

        let url = format!("{}/chat/completions", self.api_base_url);
        let response = self
            .client
            .post(&url)
            .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
            .header(CONTENT_TYPE, "application/json")
            .json(&chat_request)
            .send()
            .await
            .map_err(|e| AppError::internal(format!("OpenAI streaming request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(AppError::internal(format!(
                "OpenAI streaming API error ({}): {}",
                status, body
            )));
        }

        let stream = response.bytes_stream().map(move |chunk_result: Result<bytes::Bytes, reqwest::Error>| {
            chunk_result
                .map_err(|e| AppError::internal(format!("OpenAI stream read error: {}", e)))
                .and_then(|bytes: bytes::Bytes| {
                    let text = String::from_utf8_lossy(&bytes);
                    let mut content = String::new();
                    let mut finish_reason = None;

                    for line in text.lines() {
                        if line.starts_with("data: ") {
                            let data = &line[6..];
                            if data == "[DONE]" {
                                finish_reason = Some("stop".to_string());
                                continue;
                            }
                            if let Ok(event) = serde_json::from_str::<OpenAISSEvent>(data) {
                                if let Some(choice) = event.choices.into_iter().next() {
                                    if let Some(c) = choice.delta.content {
                                        content.push_str(&c);
                                    }
                                    if choice.finish_reason.is_some() {
                                        finish_reason = choice.finish_reason;
                                    }
                                }
                            }
                        }
                    }

                    Ok(StreamingChunk {
                        content: content.clone(),
                        delta: content,
                        finish_reason,
                    })
                })
        });

        Ok(Box::pin(stream))
    }
}

pub struct AnthropicClient {
    api_key: String,
    default_model: String,
    client: Client,
}

impl AnthropicClient {
    pub fn new(api_key: String, default_model: String) -> Self {
        Self {
            api_key,
            default_model,
            client: Client::new(),
        }
    }
}

#[async_trait]
impl LLMClient for AnthropicClient {
    async fn complete(&self, request: CompletionRequest) -> Result<LLMResponse, AppError> {
        let model = if request.model.is_empty() {
            self.default_model.clone()
        } else {
            request.model.clone()
        };

        let max_tokens = request.max_tokens.unwrap_or(1024);

        let messages: Vec<AnthropicMessage> = request
            .messages
            .into_iter()
            .map(|m| AnthropicMessage {
                role: m.role,
                content: m.content,
            })
            .collect();

        let anthropic_request = AnthropicRequest {
            model: model.clone(),
            messages,
            max_tokens,
            temperature: request.temperature,
            stream: false,
        };

        let url = format!("{}/messages", ANTHROPIC_API_BASE);
        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_API_VERSION)
            .header(CONTENT_TYPE, "application/json")
            .json(&anthropic_request)
            .send()
            .await
            .map_err(|e| AppError::internal(format!("Anthropic request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(AppError::internal(format!(
                "Anthropic API error ({}): {}",
                status, body
            )));
        }

        let anthropic_response: AnthropicResponse = response
            .json()
            .await
            .map_err(|e| AppError::internal(format!("Failed to parse Anthropic response: {}", e)))?;

        let content = anthropic_response
            .content
            .into_iter()
            .filter_map(|block| block.text)
            .collect::<Vec<_>>()
            .join("");

        Ok(LLMResponse {
            content,
            model,
            provider: "anthropic".to_string(),
            input_tokens: anthropic_response.usage.input_tokens,
            output_tokens: anthropic_response.usage.output_tokens,
            finish_reason: "stop".to_string(),
        })
    }

    async fn complete_streaming(
        &self,
        request: CompletionRequest,
    ) -> Result<BoxStream<'static, Result<StreamingChunk, AppError>>, AppError> {
        let model = if request.model.is_empty() {
            self.default_model.clone()
        } else {
            request.model.clone()
        };

        let max_tokens = request.max_tokens.unwrap_or(1024);

        let messages: Vec<AnthropicMessage> = request
            .messages
            .into_iter()
            .map(|m| AnthropicMessage {
                role: m.role,
                content: m.content,
            })
            .collect();

        let anthropic_request = AnthropicRequest {
            model: model.clone(),
            messages,
            max_tokens,
            temperature: request.temperature,
            stream: true,
        };

        let url = format!("{}/messages", ANTHROPIC_API_BASE);
        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_API_VERSION)
            .header(CONTENT_TYPE, "application/json")
            .json(&anthropic_request)
            .send()
            .await
            .map_err(|e| AppError::internal(format!("Anthropic streaming request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(AppError::internal(format!(
                "Anthropic streaming API error ({}): {}",
                status, body
            )));
        }

        let stream = response.bytes_stream().map(move |chunk_result: Result<bytes::Bytes, reqwest::Error>| {
            chunk_result
                .map_err(|e| AppError::internal(format!("Anthropic stream read error: {}", e)))
                .and_then(|bytes: bytes::Bytes| {
                    let text = String::from_utf8_lossy(&bytes);
                    let mut content = String::new();
                    let mut finish_reason = None;

                    for line in text.lines() {
                        if line.starts_with("data: ") {
                            let data = &line[6..];
                            if let Ok(event) = serde_json::from_str::<AnthropicSSEvent>(data) {
                                if event.type_ == "message_delta" {
                                    if let Some(usage) = event.usage {
                                        finish_reason = Some("end_turn".to_string());
                                        let _ = usage;
                                    }
                                } else if event.type_ == "content_block_delta" {
                                    if let Some(delta) = event.delta {
                                        if let Some(text) = delta.text {
                                            content.push_str(&text);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    Ok(StreamingChunk {
                        content: content.clone(),
                        delta: content,
                        finish_reason,
                    })
                })
        });

        Ok(Box::pin(stream))
    }
}

pub struct FireworksClient {
    api_key: String,
    api_base_url: String,
    default_model: String,
    client: Client,
}

impl FireworksClient {
    pub fn new(api_key: String, api_base_url: String, default_model: String) -> Self {
        let base_url = if api_base_url.is_empty() {
            FIREWORKS_API_BASE.to_string()
        } else {
            api_base_url
        };
        Self {
            api_key,
            api_base_url: base_url,
            default_model,
            client: Client::new(),
        }
    }
}

#[async_trait]
impl LLMClient for FireworksClient {
    async fn complete(&self, request: CompletionRequest) -> Result<LLMResponse, AppError> {
        let model = if request.model.is_empty() {
            self.default_model.clone()
        } else {
            request.model.clone()
        };

        let messages: Vec<OpenAIMessage> = request
            .messages
            .into_iter()
            .map(|m| OpenAIMessage {
                role: m.role,
                content: m.content,
                name: m.name,
            })
            .collect();

        let chat_request = OpenAIChatRequest {
            model: model.clone(),
            messages,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: false,
        };

        let url = format!("{}/chat/completions", self.api_base_url);
        let response = self
            .client
            .post(&url)
            .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
            .header(CONTENT_TYPE, "application/json")
            .json(&chat_request)
            .send()
            .await
            .map_err(|e| AppError::internal(format!("Fireworks request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(AppError::internal(format!(
                "Fireworks API error ({}): {}",
                status, body
            )));
        }

        let chat_response: OpenAIChatResponse = response
            .json()
            .await
            .map_err(|e| AppError::internal(format!("Failed to parse Fireworks response: {}", e)))?;

        let choice = chat_response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| AppError::internal("No completion choices returned"))?;

        Ok(LLMResponse {
            content: choice.message.content,
            model,
            provider: "fireworks".to_string(),
            input_tokens: chat_response.usage.prompt_tokens,
            output_tokens: chat_response.usage.completion_tokens,
            finish_reason: choice.finish_reason,
        })
    }

    async fn complete_streaming(
        &self,
        request: CompletionRequest,
    ) -> Result<BoxStream<'static, Result<StreamingChunk, AppError>>, AppError> {
        let model = if request.model.is_empty() {
            self.default_model.clone()
        } else {
            request.model.clone()
        };

        let messages: Vec<OpenAIMessage> = request
            .messages
            .into_iter()
            .map(|m| OpenAIMessage {
                role: m.role,
                content: m.content,
                name: m.name,
            })
            .collect();

        let chat_request = OpenAIChatRequest {
            model: model.clone(),
            messages,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: true,
        };

        let url = format!("{}/chat/completions", self.api_base_url);
        let response = self
            .client
            .post(&url)
            .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
            .header(CONTENT_TYPE, "application/json")
            .json(&chat_request)
            .send()
            .await
            .map_err(|e| AppError::internal(format!("Fireworks streaming request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(AppError::internal(format!(
                "Fireworks streaming API error ({}): {}",
                status, body
            )));
        }

        let stream = response.bytes_stream().map(move |chunk_result: Result<bytes::Bytes, reqwest::Error>| {
            chunk_result
                .map_err(|e| AppError::internal(format!("Fireworks stream read error: {}", e)))
                .and_then(|bytes: bytes::Bytes| {
                    let text = String::from_utf8_lossy(&bytes);
                    let mut content = String::new();
                    let mut finish_reason = None;

                    for line in text.lines() {
                        if line.starts_with("data: ") {
                            let data = &line[6..];
                            if data == "[DONE]" {
                                finish_reason = Some("stop".to_string());
                                continue;
                            }
                            if let Ok(event) = serde_json::from_str::<OpenAISSEvent>(data) {
                                if let Some(choice) = event.choices.into_iter().next() {
                                    if let Some(c) = choice.delta.content {
                                        content.push_str(&c);
                                    }
                                    if choice.finish_reason.is_some() {
                                        finish_reason = choice.finish_reason;
                                    }
                                }
                            }
                        }
                    }

                    Ok(StreamingChunk {
                        content: content.clone(),
                        delta: content,
                        finish_reason,
                    })
                })
        });

        Ok(Box::pin(stream))
    }
}

pub struct CustomClient {
    api_key: String,
    api_base_url: String,
    default_model: String,
    client: Client,
}

impl CustomClient {
    pub fn new(api_key: String, api_base_url: String, default_model: String) -> Self {
        Self {
            api_key,
            api_base_url,
            default_model,
            client: Client::new(),
        }
    }
}

#[async_trait]
impl LLMClient for CustomClient {
    async fn complete(&self, request: CompletionRequest) -> Result<LLMResponse, AppError> {
        let model = if request.model.is_empty() {
            self.default_model.clone()
        } else {
            request.model.clone()
        };

        let messages: Vec<OpenAIMessage> = request
            .messages
            .into_iter()
            .map(|m| OpenAIMessage {
                role: m.role,
                content: m.content,
                name: m.name,
            })
            .collect();

        let chat_request = OpenAIChatRequest {
            model: model.clone(),
            messages,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: false,
        };

        let url = format!("{}/chat/completions", self.api_base_url);
        let response = self
            .client
            .post(&url)
            .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
            .header(CONTENT_TYPE, "application/json")
            .json(&chat_request)
            .send()
            .await
            .map_err(|e| AppError::internal(format!("Custom provider request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(AppError::internal(format!(
                "Custom provider API error ({}): {}",
                status, body
            )));
        }

        let chat_response: OpenAIChatResponse = response
            .json()
            .await
            .map_err(|e| AppError::internal(format!("Failed to parse custom provider response: {}", e)))?;

        let choice = chat_response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| AppError::internal("No completion choices returned"))?;

        Ok(LLMResponse {
            content: choice.message.content,
            model,
            provider: "custom".to_string(),
            input_tokens: chat_response.usage.prompt_tokens,
            output_tokens: chat_response.usage.completion_tokens,
            finish_reason: choice.finish_reason,
        })
    }

    async fn complete_streaming(
        &self,
        request: CompletionRequest,
    ) -> Result<BoxStream<'static, Result<StreamingChunk, AppError>>, AppError> {
        let model = if request.model.is_empty() {
            self.default_model.clone()
        } else {
            request.model.clone()
        };

        let messages: Vec<OpenAIMessage> = request
            .messages
            .into_iter()
            .map(|m| OpenAIMessage {
                role: m.role,
                content: m.content,
                name: m.name,
            })
            .collect();

        let chat_request = OpenAIChatRequest {
            model: model.clone(),
            messages,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: true,
        };

        let url = format!("{}/chat/completions", self.api_base_url);
        let response = self
            .client
            .post(&url)
            .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
            .header(CONTENT_TYPE, "application/json")
            .json(&chat_request)
            .send()
            .await
            .map_err(|e| AppError::internal(format!("Custom provider streaming request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(AppError::internal(format!(
                "Custom provider streaming API error ({}): {}",
                status, body
            )));
        }

        let stream = response.bytes_stream().map(move |chunk_result: Result<bytes::Bytes, reqwest::Error>| {
            chunk_result
                .map_err(|e| AppError::internal(format!("Custom provider stream read error: {}", e)))
                .and_then(|bytes: bytes::Bytes| {
                    let text = String::from_utf8_lossy(&bytes);
                    let mut content = String::new();
                    let mut finish_reason = None;

                    for line in text.lines() {
                        if line.starts_with("data: ") {
                            let data = &line[6..];
                            if data == "[DONE]" {
                                finish_reason = Some("stop".to_string());
                                continue;
                            }
                            if let Ok(event) = serde_json::from_str::<OpenAISSEvent>(data) {
                                if let Some(choice) = event.choices.into_iter().next() {
                                    if let Some(c) = choice.delta.content {
                                        content.push_str(&c);
                                    }
                                    if choice.finish_reason.is_some() {
                                        finish_reason = choice.finish_reason;
                                    }
                                }
                            }
                        }
                    }

                    Ok(StreamingChunk {
                        content: content.clone(),
                        delta: content,
                        finish_reason,
                    })
                })
        });

        Ok(Box::pin(stream))
    }
}
