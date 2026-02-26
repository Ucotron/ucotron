# Bugs Pendientes — OSS QA

**Fecha:** 2026-02-26
**Branch:** `ralph/oss-qa`

---

## Bugs Fixeados en este QA

### BUG-2: GET devuelve 200 para memorias borradas (FIXED)
- **Archivo:** `ucotron_server/src/handlers/mod.rs`
- **Fix:** Se agregó helper `node_is_deleted()` que chequea `metadata.deleted` flag
- **Commit:** `b4f8b4a`

### BUG-8: Whisper token decoder produce basura base64 (FIXED)
- **Archivo:** `ucotron_extraction/src/audio.rs`
- **Fix:** Se agregó decodificación base64 para el formato sherpa-onnx two-column
- **Commit:** pendiente (cambios locales sin push)

### BUG-9: Whisper pipeline no se inicializa (FIXED)
- **Archivo:** `ucotron_server/src/main.rs`
- **Fix:** Se agregó `try_init_whisper()` (mismo patrón que `try_init_clip()`)
- **Commit:** pendiente (cambios locales sin push)

### BUG-10: Video pipeline no se inicializa (FIXED)
- **Archivo:** `ucotron_server/src/main.rs`
- **Fix:** Se agregó `try_init_video()` y `AppState::with_all_pipelines_full()`
- **Commit:** pendiente (cambios locales sin push)

---

## Bugs Pendientes (Pre-Launch)

### BUG-P1: Feature `llm` no implementado — LLM inference inactivo
- **Severidad:** ALTA
- **Impacto:** `/augment` es solo retrieval puro (sin LLM). Relation extraction cae a co-occurrence. Los modelos GGUF (Qwen3-4B, Qwen3-0.6B) se descargan pero nunca se cargan.
- **Causa raíz:** El feature flag `llm` en `ucotron_extraction/Cargo.toml` es un stub vacío (`llm = []`). No hay dependencia de `llama-cpp-2` ni implementación de `LlmRelationExtractor`.
- **Archivos a modificar:**
  1. `ucotron_extraction/Cargo.toml` — agregar `llm = ["dep:llama-cpp-2"]` y la dependencia
  2. `ucotron_extraction/src/relations.rs` — implementar `LlmRelationExtractor` dentro de `#[cfg(feature = "llm")]`
  3. `ucotron_server/Cargo.toml` — exponer feature `llm = ["ucotron-extraction/llm"]`
  4. Opcionalmente: agregar LLM-augmented `/augment` que use el modelo para refinar respuestas
- **Para testear:** `cargo build --release --features llm` y re-correr benchmarks QA-011 y QA-014

### BUG-P2: NER model no se carga (ner_loaded=false)
- **Severidad:** ALTA
- **Impacto:** Entity extraction no funciona. El grafo tiene 0 edges y 0 entity nodes. Todas las entidades están vacías.
- **Causa raíz:** A pesar de que gliner_small-v2.1 está descargado (583MB), el health endpoint muestra `ner_loaded=false`. Falta investigar si es un problema de inicialización o de formato del modelo.
- **Archivos a investigar:**
  1. `ucotron_extraction/src/ner.rs` — verificar que la carga del modelo no falla silenciosamente
  2. `ucotron_server/src/main.rs` — verificar que se intenta inicializar NER

### BUG-P3: Corrupted media devuelve 500 en vez de 400
- **Severidad:** BAJA
- **Impacto:** Archivos corruptos (WAV, MP4, PNG) devuelven HTTP 500 Internal Server Error en vez de 400 Bad Request. El server no crashea pero el status code es incorrecto.
- **Archivos a modificar:**
  1. `ucotron_server/src/handlers/audio.rs` — catch parse errors y devolver 400
  2. `ucotron_server/src/handlers/image.rs` — mismo
  3. `ucotron_server/src/handlers/video.rs` — mismo

### BUG-P4: API keys runtime son efímeras
- **Severidad:** MEDIA
- **Impacto:** Keys creadas via `POST /auth/keys` se pierden al reiniciar el server. Solo las keys en `ucotron.toml` persisten.
- **Fix propuesto:** Persistir keys en LMDB o un archivo de estado

### BUG-P5: Chunking demasiado agresivo
- **Severidad:** BAJA
- **Impacto:** Un texto de 21.5KB produce 180 chunks (~120 bytes promedio). Genera demasiados nodos para documentos grandes.
- **Fix propuesto:** Hacer el chunk size configurable en `ucotron.toml` o aumentar defaults

---

## Resumen de Prioridades

| Bug | Severidad | Bloquea Launch? | Esfuerzo |
|-----|-----------|-----------------|----------|
| BUG-P1 (LLM feature) | ALTA | Si se publicita LLM | 2-3 días |
| BUG-P2 (NER no carga) | ALTA | Si se publicita NER/entities | 1 día |
| BUG-P3 (500 vs 400) | BAJA | No | 2 horas |
| BUG-P4 (keys efímeras) | MEDIA | No (se documenta) | 1 día |
| BUG-P5 (chunking) | BAJA | No (se documenta) | 2 horas |
