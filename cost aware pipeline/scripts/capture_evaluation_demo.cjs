const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

const root = path.resolve(__dirname, "..");
const outputDir = path.join(root, "docs", "demo", "captures");
const edgePath = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";

function readEnv(filePath) {
  const values = {};
  for (const rawLine of fs.readFileSync(filePath, "utf8").split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#") || !line.includes("=")) continue;
    const separator = line.indexOf("=");
    values[line.slice(0, separator).trim()] = line.slice(separator + 1).trim().replace(/^['"]|['"]$/g, "");
  }
  return values;
}

function adminKey(env) {
  const configured = env.APP_API_KEYS || "";
  const admin = configured.split(",").map((item) => item.trim()).find((item) => item.startsWith("admin:"));
  if (!admin) throw new Error("APP_API_KEYS does not contain an admin key");
  return admin.slice(admin.indexOf(":") + 1);
}

async function main() {
  fs.mkdirSync(outputDir, { recursive: true });
  const env = readEnv(path.join(root, ".env"));
  const browser = await chromium.launch({ executablePath: edgePath, headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
  const page = await context.newPage();

  const login = await context.request.post("http://localhost:8000/login", {
    form: { api_key: adminKey(env) },
  });
  if (!login.ok()) throw new Error(`Local application login failed with HTTP ${login.status()}`);
  await page.goto("http://localhost:8000/evaluation", { waitUntil: "networkidle" });

  const completedRun = page.locator(".history-run").filter({ hasText: "completed" }).first();
  if (await completedRun.count()) {
    await completedRun.click();
  } else {
    const summary = (await page.locator("body").innerText()).slice(0, 500).replace(/\s+/g, " ");
    throw new Error(`No completed evaluation was visible at ${page.url()}. Page summary: ${summary}`);
  }
  await page.locator("#evaluation-result .evaluation-card").waitFor({ state: "visible" });

  const card = page.locator("#evaluation-result .evaluation-card");
  const selector = page.getByLabel("Context precision view");
  await selector.selectOption("claim");
  await card.screenshot({ path: path.join(outputDir, "evaluation-claim-aware.png") });

  await selector.selectOption("ragas");
  await card.screenshot({ path: path.join(outputDir, "evaluation-original-ragas.png") });

  await selector.selectOption("compare");
  await card.screenshot({ path: path.join(outputDir, "evaluation-compare.png") });

  const audit = card.locator("details").filter({ hasText: "Per-excerpt retrieval audit" }).first();
  await audit.locator("summary").click();
  await audit.screenshot({ path: path.join(outputDir, "evaluation-retrieval-audit.png") });

  console.log("Captured evaluation demo views without running research or evaluation.");
  await browser.close();
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
