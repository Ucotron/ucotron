"use client";

import { useState } from "react";
import { Settings } from "lucide-react";
import { Card } from "@/components/card";

export default function SettingsPage() {
  const [apiUrl, setApiUrl] = useState(
    process.env.NEXT_PUBLIC_UCOTRON_API_URL || "http://localhost:8420"
  );

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Settings</h1>

      <Card title="API Connection">
        <div className="space-y-4">
          <div>
            <label className="mb-1.5 block text-sm text-muted-foreground">
              Ucotron API URL
            </label>
            <input
              type="text"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-primary"
            />
            <p className="mt-1 text-xs text-muted-foreground">
              Set via <code className="rounded bg-muted px-1">NEXT_PUBLIC_UCOTRON_API_URL</code> environment variable
            </p>
          </div>
        </div>
      </Card>

      <Card title="About">
        <div className="flex items-start gap-3">
          <Settings className="mt-0.5 h-5 w-5 text-muted-foreground" />
          <div className="text-sm">
            <p>Ucotron Dashboard v0.1.0</p>
            <p className="mt-1 text-muted-foreground">
              Memory graph administration interface. Connects to the Ucotron REST API
              for managing memories, entities, and monitoring server health.
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
}
