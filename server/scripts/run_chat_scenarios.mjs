#!/usr/bin/env node
/**
 * run_chat_scenarios.mjs
 *
 * Parses CHAT_SCENARIOS.md at the workspace root, sends each scenario turn
 * to the Next.js chat API, rate-limits to 2 req/s, and logs results to
 * test_results/chat_logs/ as JSON files.
 *
 * Usage:
 *   node run_chat_scenarios.mjs [--dry-run] [--scenario N]
 *
 * Flags:
 *   --dry-run      Print parsed scenarios and planned requests without sending
 *   --scenario N   Only run scenario number N (1-based)
 */

import { readFileSync, mkdirSync, writeFileSync, existsSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const __dirname = dirname(fileURLToPath(import.meta.url));
// Workspace root is two levels up from memory_arena/scripts/
const WORKSPACE_ROOT = join(__dirname, "..", "..");
const SCENARIOS_FILE = join(WORKSPACE_ROOT, "CHAT_SCENARIOS.md");
const CHAT_API_URL = "http://localhost:3002/api/chat";
const OUTPUT_DIR = join(WORKSPACE_ROOT, "test_results", "chat_logs");
const RATE_LIMIT_MS = 500; // 2 req/s = 500ms between requests

// ---------------------------------------------------------------------------
// CLI args
// ---------------------------------------------------------------------------

const args = process.argv.slice(2);
const DRY_RUN = args.includes("--dry-run");
const scenarioFlagIdx = args.indexOf("--scenario");
const SCENARIO_FILTER =
  scenarioFlagIdx !== -1 ? parseInt(args[scenarioFlagIdx + 1], 10) : null;

// ---------------------------------------------------------------------------
// Markdown parser
// ---------------------------------------------------------------------------

/**
 * @typedef {Object} ScenarioTurn
 * @property {"user"|"assistant"} role
 * @property {string} content
 */

/**
 * @typedef {Object} Scenario
 * @property {number} index        1-based scenario number
 * @property {string} name
 * @property {string} namespace
 * @property {ScenarioTurn[]} turns
 */

/**
 * Parse CHAT_SCENARIOS.md.
 *
 * Expected format:
 *
 * ## Scenario 1: <name>
 * **Namespace:** <namespace>
 *
 * **User:** <message>
 * **Assistant:** <expected response (optional, for reference)>
 *
 * @param {string} filePath
 * @returns {Scenario[]}
 */
function parseScenarios(filePath) {
  if (!existsSync(filePath)) {
    console.error(`CHAT_SCENARIOS.md not found at: ${filePath}`);
    console.error(
      "Create the file at the workspace root with scenario definitions."
    );
    process.exit(1);
  }

  const raw = readFileSync(filePath, "utf-8");
  const lines = raw.split("\n");

  const scenarios = [];
  let current = null;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Start of a new scenario: ## Scenario N: Title
    const scenarioMatch = line.match(/^##\s+Scenario\s+(\d+)[:\s]+(.+)/i);
    if (scenarioMatch) {
      if (current) scenarios.push(current);
      current = {
        index: parseInt(scenarioMatch[1], 10),
        name: scenarioMatch[2].trim(),
        namespace: "default",
        turns: [],
      };
      continue;
    }

    if (!current) continue;

    // Namespace line: **Namespace:** value
    const nsMatch = line.match(/^\*\*Namespace\*\*[:\s]+(.+)/i);
    if (nsMatch) {
      current.namespace = nsMatch[1].trim();
      continue;
    }

    // User turn: **User:** message (possibly multi-line until next ** or blank)
    const userMatch = line.match(/^\*\*User\*\*[:\s]+(.+)?/i);
    if (userMatch) {
      let content = (userMatch[1] || "").trim();
      // Collect continuation lines
      while (i + 1 < lines.length) {
        const next = lines[i + 1];
        if (next.match(/^\*\*\w/) || next.match(/^##/)) break;
        i++;
        const trimmed = next.trim();
        if (trimmed) content += (content ? " " : "") + trimmed;
      }
      if (content) current.turns.push({ role: "user", content });
      continue;
    }

    // Assistant turn: **Assistant:** message (optional reference)
    const assistantMatch = line.match(/^\*\*Assistant\*\*[:\s]+(.+)?/i);
    if (assistantMatch) {
      let content = (assistantMatch[1] || "").trim();
      while (i + 1 < lines.length) {
        const next = lines[i + 1];
        if (next.match(/^\*\*\w/) || next.match(/^##/)) break;
        i++;
        const trimmed = next.trim();
        if (trimmed) content += (content ? " " : "") + trimmed;
      }
      if (content) current.turns.push({ role: "assistant", content });
      continue;
    }
  }

  if (current) scenarios.push(current);
  return scenarios;
}

// ---------------------------------------------------------------------------
// Streaming response parser (Vercel AI SDK format)
// ---------------------------------------------------------------------------

/**
 * Parse Vercel AI SDK streaming response.
 * Lines are formatted as: 0:"text chunk"
 * Other line types (data, error, etc.) are noted but not included in text.
 *
 * @param {string} rawBody
 * @returns {{ text: string, chunks: string[], rawLines: string[] }}
 */
function parseStreamingResponse(rawBody) {
  const rawLines = rawBody.split("\n").filter((l) => l.trim());
  const chunks = [];
  let text = "";

  for (const line of rawLines) {
    // Text chunk: 0:"..."
    const textMatch = line.match(/^0:"((?:[^"\\]|\\.)*)"/);
    if (textMatch) {
      // Unescape JSON string sequences
      const chunk = JSON.parse(`"${textMatch[1]}"`);
      chunks.push(chunk);
      text += chunk;
      continue;
    }

    // Finished message: d:{...} — ignore for text extraction
    // Error: 3:"..." — could log
    const errorMatch = line.match(/^3:"((?:[^"\\]|\\.)*)"/);
    if (errorMatch) {
      console.warn("  Stream error chunk:", errorMatch[1]);
    }
  }

  return { text, chunks, rawLines };
}

// ---------------------------------------------------------------------------
// Rate limiter
// ---------------------------------------------------------------------------

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------------------
// API call
// ---------------------------------------------------------------------------

/**
 * Send a single turn to the chat API.
 *
 * @param {string} namespace
 * @param {{ role: string; content: string }[]} messages
 * @returns {Promise<{ ok: boolean; status: number; text: string; rawBody: string; durationMs: number }>}
 */
async function sendChatRequest(namespace, messages) {
  const start = Date.now();
  let ok = false;
  let status = 0;
  let rawBody = "";
  let text = "";

  try {
    const response = await fetch(CHAT_API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages, namespace }),
    });

    status = response.status;
    ok = response.ok;
    rawBody = await response.text();

    if (ok) {
      const parsed = parseStreamingResponse(rawBody);
      text = parsed.text;
    } else {
      text = rawBody;
    }
  } catch (err) {
    text = `Network error: ${err.message}`;
    rawBody = text;
  }

  return { ok, status, text, rawBody, durationMs: Date.now() - start };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  console.log("Ucotron Chat Scenario Runner");
  console.log("============================");
  console.log(`Scenarios file: ${SCENARIOS_FILE}`);
  console.log(`API URL:        ${CHAT_API_URL}`);
  console.log(`Dry run:        ${DRY_RUN}`);
  console.log(`Scenario filter: ${SCENARIO_FILTER ?? "all"}`);
  console.log();

  // Parse scenarios
  const allScenarios = parseScenarios(SCENARIOS_FILE);
  console.log(`Parsed ${allScenarios.length} scenario(s) from CHAT_SCENARIOS.md`);

  // Apply filter
  const scenarios =
    SCENARIO_FILTER !== null
      ? allScenarios.filter((s) => s.index === SCENARIO_FILTER)
      : allScenarios;

  if (scenarios.length === 0) {
    console.error(
      SCENARIO_FILTER !== null
        ? `Scenario ${SCENARIO_FILTER} not found.`
        : "No scenarios found in CHAT_SCENARIOS.md."
    );
    process.exit(1);
  }

  if (DRY_RUN) {
    console.log("\n--- DRY RUN: planned requests ---\n");
    for (const scenario of scenarios) {
      console.log(
        `Scenario ${scenario.index}: "${scenario.name}" [namespace: ${scenario.namespace}]`
      );
      const userTurns = scenario.turns.filter((t) => t.role === "user");
      for (let i = 0; i < userTurns.length; i++) {
        const turn = userTurns[i];
        const preview =
          turn.content.length > 80
            ? turn.content.slice(0, 77) + "..."
            : turn.content;
        console.log(`  Turn ${i + 1}: POST ${CHAT_API_URL}`);
        console.log(`           messages=[{role:"user", content:"${preview}"}]`);
        console.log(`           namespace="${scenario.namespace}"`);
      }
      console.log();
    }
    console.log("Dry run complete — no requests sent.");
    return;
  }

  // Ensure output directory exists
  mkdirSync(OUTPUT_DIR, { recursive: true });
  console.log(`Output directory: ${OUTPUT_DIR}\n`);

  const globalResults = [];

  for (const scenario of scenarios) {
    console.log(
      `Running Scenario ${scenario.index}: "${scenario.name}" [namespace: ${scenario.namespace}]`
    );

    const scenarioLog = {
      scenario_index: scenario.index,
      scenario_name: scenario.name,
      namespace: scenario.namespace,
      started_at: new Date().toISOString(),
      turns: [],
    };

    // Build conversation history (including reference assistant turns)
    const conversationHistory = [];

    let turnIndex = 0;
    for (const turn of scenario.turns) {
      if (turn.role === "user") {
        turnIndex++;
        console.log(`  Turn ${turnIndex}: sending user message...`);

        // Include history up to this user turn
        const messages = [
          ...conversationHistory,
          { role: "user", content: turn.content },
        ];

        const result = await sendChatRequest(scenario.namespace, messages);

        const turnLog = {
          turn_index: turnIndex,
          user_message: turn.content,
          response_text: result.text,
          status_code: result.status,
          ok: result.ok,
          duration_ms: result.durationMs,
          raw_body: result.rawBody,
          timestamp: new Date().toISOString(),
        };

        scenarioLog.turns.push(turnLog);
        conversationHistory.push({ role: "user", content: turn.content });

        if (result.ok) {
          const preview =
            result.text.length > 100
              ? result.text.slice(0, 97) + "..."
              : result.text;
          console.log(
            `    OK (${result.status}) ${result.durationMs}ms — "${preview}"`
          );
          conversationHistory.push({ role: "assistant", content: result.text });
        } else {
          console.warn(
            `    ERROR (${result.status}) ${result.durationMs}ms — ${result.text}`
          );
          // Push empty assistant message so conversation remains coherent
          conversationHistory.push({ role: "assistant", content: "" });
        }

        // Rate limit: 2 req/s
        await sleep(RATE_LIMIT_MS);
      } else if (turn.role === "assistant") {
        // Reference assistant turn — add to history for context but don't send
        conversationHistory.push({ role: "assistant", content: turn.content });
      }
    }

    scenarioLog.finished_at = new Date().toISOString();
    scenarioLog.total_turns = turnIndex;
    scenarioLog.success_count = scenarioLog.turns.filter((t) => t.ok).length;
    scenarioLog.error_count = scenarioLog.turns.filter((t) => !t.ok).length;

    // Write per-scenario JSON log
    const safeName = scenario.name.toLowerCase().replace(/[^a-z0-9]+/g, "_");
    const filename = `scenario_${String(scenario.index).padStart(3, "0")}_${safeName}.json`;
    const outputPath = join(OUTPUT_DIR, filename);
    writeFileSync(outputPath, JSON.stringify(scenarioLog, null, 2), "utf-8");

    console.log(
      `  Saved: ${outputPath} (${turnIndex} turns, ${scenarioLog.success_count} ok, ${scenarioLog.error_count} errors)`
    );
    console.log();

    globalResults.push({
      scenario_index: scenario.index,
      scenario_name: scenario.name,
      total_turns: turnIndex,
      success_count: scenarioLog.success_count,
      error_count: scenarioLog.error_count,
      output_file: filename,
    });
  }

  // Write summary
  const summaryPath = join(OUTPUT_DIR, "summary.json");
  const summary = {
    generated_at: new Date().toISOString(),
    total_scenarios: scenarios.length,
    results: globalResults,
  };
  writeFileSync(summaryPath, JSON.stringify(summary, null, 2), "utf-8");

  console.log("============================");
  console.log(
    `Done. ${scenarios.length} scenario(s) processed. Summary: ${summaryPath}`
  );
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
