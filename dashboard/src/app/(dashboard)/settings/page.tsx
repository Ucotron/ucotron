"use client";

import { useState } from "react";
import {
  Settings,
  User,
  Key,
  Bell,
  Palette,
  Sun,
  Moon,
  Monitor,
  Globe,
  Trash2,
  Plus,
  Copy,
  Check,
} from "lucide-react";
import { Card, Button, Input } from "@ucotron/ui";
import { useTheme } from "@/components/theme-provider";
import { useLocale } from "@/components/locale-provider";
import { useTranslation } from "@/components/use-translation";
import { cn } from "@/lib/utils";

type TabId = "profile" | "api-keys" | "notifications" | "preferences";

interface Tab {
  id: TabId;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}

const TABS: Tab[] = [
  { id: "profile", label: "Profile", icon: User },
  { id: "api-keys", label: "API Keys", icon: Key },
  { id: "notifications", label: "Notifications", icon: Bell },
  { id: "preferences", label: "Preferences", icon: Palette },
];

const LANGUAGES = [
  { code: "en", label: "English" },
  { code: "es", label: "Español" },
];

function ProfileTab() {
  const [name, setName] = useState("Admin User");
  const [email, setEmail] = useState("admin@ucotron.local");
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="space-y-6">
      <Card title="Profile Information">
        <div className="space-y-4">
          <div>
            <label className="mb-1.5 block text-sm text-muted-foreground">
              Display Name
            </label>
            <Input
              value={name}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setName(e.target.value)}
              placeholder="Your display name"
            />
          </div>
          <div>
            <label className="mb-1.5 block text-sm text-muted-foreground">
              Email Address
            </label>
            <Input
              type="email"
              value={email}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setEmail(e.target.value)}
              placeholder="your@email.com"
            />
          </div>
          <div className="flex items-center gap-3">
            <Button onClick={handleSave}>
              {saved ? "Saved!" : "Save Changes"}
            </Button>
            {saved && (
              <span className="text-sm text-green-500">Profile updated</span>
            )}
          </div>
        </div>
      </Card>

      <Card title="Account Security">
        <div className="space-y-4">
          <div>
            <label className="mb-1.5 block text-sm text-muted-foreground">
              Current Password
            </label>
            <Input
              type="password"
              placeholder="Enter current password"
            />
          </div>
          <div>
            <label className="mb-1.5 block text-sm text-muted-foreground">
              New Password
            </label>
            <Input
              type="password"
              placeholder="Enter new password"
            />
          </div>
          <Button variant="outline">Update Password</Button>
        </div>
      </Card>
    </div>
  );
}

function ApiKeysTab() {
  const [keys, setKeys] = useState([
    { id: "1", name: "Production Key", prefix: "uc_sk_****...****3f2a", created: "2026-01-15" },
    { id: "2", name: "Development Key", prefix: "uc_sk_****...****7b1c", created: "2026-02-01" },
  ]);
  const [showNewKey, setShowNewKey] = useState(false);
  const [newKeyName, setNewKeyName] = useState("");
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const handleCopy = (key: { prefix: string; id: string }) => {
    navigator.clipboard.writeText(key.prefix);
    setCopiedId(key.id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleDelete = (id: string) => {
    setKeys(keys.filter((k) => k.id !== id));
  };

  const handleCreate = () => {
    if (!newKeyName.trim()) return;
    const newKey = {
      id: Date.now().toString(),
      name: newKeyName,
      prefix: `uc_sk_****...****${Math.random().toString(16).slice(2, 6)}`,
      created: new Date().toISOString().split("T")[0],
    };
    setKeys([...keys, newKey]);
    setNewKeyName("");
    setShowNewKey(false);
  };

  return (
    <div className="space-y-6">
      <Card title="API Keys">
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Manage your API keys for programmatic access to the Ucotron API.
          </p>
          
          {keys.map((key) => (
            <div
              key={key.id}
              className="flex items-center justify-between rounded-lg border border-border p-3"
            >
              <div>
                <p className="font-medium">{key.name}</p>
                <p className="font-mono text-sm text-muted-foreground">{key.prefix}</p>
                <p className="text-xs text-muted-foreground">Created: {key.created}</p>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => handleCopy(key)}
                  title="Copy key"
                >
                  {copiedId === key.id ? (
                    <Check className="h-4 w-4 text-green-500" />
                  ) : (
                    <Copy className="h-4 w-4" />
                  )}
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => handleDelete(key.id)}
                  title="Delete key"
                >
                  <Trash2 className="h-4 w-4 text-red-500" />
                </Button>
              </div>
            </div>
          ))}

          {showNewKey ? (
            <div className="flex items-center gap-3">
              <Input
                value={newKeyName}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNewKeyName(e.target.value)}
                placeholder="Key name (e.g., Production)"
                className="flex-1"
              />
              <Button onClick={handleCreate} disabled={!newKeyName.trim()}>
                <Plus className="mr-2 h-4 w-4" />
                Create
              </Button>
              <Button variant="outline" onClick={() => setShowNewKey(false)}>
                Cancel
              </Button>
            </div>
          ) : (
            <Button onClick={() => setShowNewKey(true)}>
              <Plus className="mr-2 h-4 w-4" />
              Create New Key
            </Button>
          )}
        </div>
      </Card>
    </div>
  );
}

function NotificationsTab() {
  const [settings, setSettings] = useState({
    emailDigest: true,
    securityAlerts: true,
    productUpdates: false,
    memoryAlerts: true,
    connectorStatus: true,
    weeklyReport: false,
  });

  const toggle = (key: keyof typeof settings) => {
    setSettings({ ...settings, [key]: !settings[key] });
  };

  return (
    <div className="space-y-6">
      <Card title="Notification Preferences">
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Choose how you want to receive notifications.
          </p>

          <div className="space-y-3">
            {[
              { key: "emailDigest", label: "Email Digest", desc: "Receive daily summary of activity" },
              { key: "securityAlerts", label: "Security Alerts", desc: "Get notified about security events" },
              { key: "productUpdates", label: "Product Updates", desc: "Learn about new features and improvements" },
              { key: "memoryAlerts", label: "Memory Alerts", desc: "Notifications about memory operations" },
              { key: "connectorStatus", label: "Connector Status", desc: "Updates on connector sync status" },
              { key: "weeklyReport", label: "Weekly Report", desc: "Receive weekly usage statistics" },
            ].map((item) => (
              <div key={item.key} className="flex items-center justify-between">
                <div>
                  <p className="font-medium">{item.label}</p>
                  <p className="text-sm text-muted-foreground">{item.desc}</p>
                </div>
                <button
                  onClick={() => toggle(item.key as keyof typeof settings)}
                  className={cn(
                    "relative h-6 w-11 rounded-full transition-colors",
                    settings[item.key as keyof typeof settings]
                      ? "bg-primary"
                      : "bg-muted"
                  )}
                >
                  <span
                    className={cn(
                      "absolute left-0.5 top-0.5 h-5 w-5 rounded-full bg-white transition-transform",
                      settings[item.key as keyof typeof settings] && "translate-x-5"
                    )}
                  />
                </button>
              </div>
            ))}
          </div>
        </div>
      </Card>
    </div>
  );
}

function PreferencesTab() {
  const { theme, setTheme } = useTheme();
  const { locale, setLocale } = useLocale();
  const { t } = useTranslation();
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="space-y-6">
      <Card title={t("settings.appearance")}>
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Customize the look and feel of the dashboard.
          </p>

          <div>
            <label className="mb-3 block text-sm text-muted-foreground">
              {t("settings.theme")}
            </label>
            <div className="flex items-center gap-2 rounded-lg border border-border p-1">
              {[
                { value: "light" as const, icon: Sun, label: t("settings.light") },
                { value: "dark" as const, icon: Moon, label: t("settings.dark") },
                { value: "system" as const, icon: Monitor, label: t("settings.system") },
              ].map(({ value, icon: Icon, label }) => (
                <button
                  key={value}
                  onClick={() => setTheme(value)}
                  className={cn(
                    "flex flex-1 items-center justify-center gap-2 rounded-md py-2 transition-colors",
                    theme === value
                      ? "bg-accent text-accent-foreground"
                      : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  <Icon className="h-4 w-4" />
                  <span className="text-sm">{label}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </Card>

      <Card title={t("settings.languageRegion")}>
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Set your preferred language and regional settings.
          </p>

          <div>
            <label className="mb-1.5 block text-sm text-muted-foreground">
              <Globe className="mr-2 inline h-4 w-4" />
              {t("settings.language")}
            </label>
            <select
              value={locale}
              onChange={(e) => setLocale(e.target.value as "en" | "es")}
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
            >
              {LANGUAGES.map((lang) => (
                <option key={lang.code} value={lang.code}>
                  {lang.label}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-3">
            <Button onClick={handleSave}>
              {saved ? t("settings.saved") : t("settings.saveChanges")}
            </Button>
            {saved && (
              <span className="text-sm text-green-500">{t("settings.preferencesUpdated")}</span>
            )}
          </div>
        </div>
      </Card>

      <Card title={t("settings.about")}>
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

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState<TabId>("profile");
  const { t } = useTranslation();

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">{t("settings.title")}</h1>

      <div className="flex gap-1 border-b border-border">
        {TABS.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={cn(
                "flex items-center gap-2 border-b-2 px-4 py-2.5 text-sm font-medium transition-colors",
                activeTab === tab.id
                  ? "border-primary text-foreground"
                  : "border-transparent text-muted-foreground hover:text-foreground"
              )}
            >
              <Icon className="h-4 w-4" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {activeTab === "profile" && <ProfileTab />}
      {activeTab === "api-keys" && <ApiKeysTab />}
      {activeTab === "notifications" && <NotificationsTab />}
      {activeTab === "preferences" && <PreferencesTab />}
    </div>
  );
}
