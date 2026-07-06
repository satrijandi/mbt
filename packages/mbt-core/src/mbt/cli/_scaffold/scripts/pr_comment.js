// mbt PR comment bot (FR-STATE-05): metrics vs champion, gate table,
// retrained nodes, cost estimate. Sourced from run_results.json and
// state_diff.json ONLY - no re-computation in workflow scripts (S7-04).
const fs = require('fs');

const MARKER = '<!-- mbt-pr-comment -->';
const RUNNER_USD_PER_MINUTE = 0.008; // GitHub-hosted ubuntu-latest

function loadJson(path) {
  try {
    return JSON.parse(fs.readFileSync(path, 'utf8'));
  } catch (e) {
    return null;
  }
}

function gateRow(nodeId, g) {
  const status = g.passed ? 'PASS' : '**FAIL**';
  if (g.kind === 'champion') {
    const champ = g.champion_version
      ? `v${g.champion_version} = ${g.champion_value?.toFixed(4)}`
      : 'none (bootstrap)';
    const delta = g.actual_delta == null ? '-' : g.actual_delta.toFixed(4);
    return `| ${nodeId} | ${g.metric} | champion (${champ}) | ${g.actual?.toFixed(4)} | ${delta} >= ${g.min_delta} | ${status} |`;
  }
  return `| ${nodeId} | ${g.metric} | threshold ${g.expected} | ${g.actual?.toFixed(4)} | - | ${status} |`;
}

function buildBody() {
  const results = loadJson('target/run_results.json');
  const diff = loadJson('target/state_diff.json');
  let body = `${MARKER}\n## mbt build report\n\n`;

  if (!results) {
    return body + 'No `run_results.json` produced - the build failed before execution. Check the workflow logs.\n';
  }

  const nodes = results.results || [];
  const trained = nodes.filter((r) => r.unique_id.startsWith('model.'));
  const failed = nodes.filter((r) => ['error', 'gate_failed', 'test_failed'].includes(r.status));

  body += `**Target:** \`${results.metadata.target}\` · **Command:** \`${results.metadata.command}\``;
  if (results.metadata.selector) body += ` · **Selector:** \`${results.metadata.selector}\``;
  body += `\n\n`;

  if (diff) {
    const mod = (diff.modified || []).map((d) => `\`${d.unique_id}\` (${d.components.join(', ')})`);
    const added = (diff.added || []).map((d) => `\`${d.unique_id}\``);
    body += `### Changed vs production\n`;
    body += mod.length || added.length
      ? [...added.map((a) => `- added: ${a}`), ...mod.map((m) => `- modified: ${m}`)].join('\n') + '\n\n'
      : 'Nothing modified - no retraining needed.\n\n';
    if (diff.env && diff.env.changed) {
      body += '> ⚠️ environment digest changed vs the reference manifest (not treated as modified by default).\n\n';
    }
  }

  if (nodes.length) {
    body += `### Nodes\n| node | status | time |\n|---|---|---|\n`;
    for (const r of nodes) {
      body += `| \`${r.unique_id}\` | ${r.status} | ${r.execution_time_s.toFixed(1)}s |\n`;
    }
    body += '\n';
  }

  const gateRows = nodes.flatMap((r) => (r.gates || []).map((g) => gateRow(r.unique_id, g)));
  if (gateRows.length) {
    body += `### Gates (metrics vs champion)\n| node | metric | gate | actual | delta | result |\n|---|---|---|---|---|---|\n`;
    body += gateRows.join('\n') + '\n\n';
  }

  const totalSeconds = nodes.reduce((s, r) => s + (r.execution_time_s || 0), 0);
  const cost = (totalSeconds / 60) * RUNNER_USD_PER_MINUTE;
  body += `### Cost\nExecution time: **${totalSeconds.toFixed(0)}s** across ${trained.length} model(s) → est. **$${cost.toFixed(3)}** at $${RUNNER_USD_PER_MINUTE}/min.\n`;

  if (failed.length) {
    body += `\n> ❌ ${failed.length} node(s) failed - registration blocked.\n`;
  }
  return body;
}

module.exports = async ({ github, context }) => {
  const body = buildBody();
  const { owner, repo } = context.repo;
  const issue_number = context.issue.number;
  const comments = await github.rest.issues.listComments({ owner, repo, issue_number });
  const existing = comments.data.find((c) => c.body.includes(MARKER));
  if (existing) {
    // update in place instead of stacking (S7-04)
    await github.rest.issues.updateComment({ owner, repo, comment_id: existing.id, body });
  } else {
    await github.rest.issues.createComment({ owner, repo, issue_number, body });
  }
};
